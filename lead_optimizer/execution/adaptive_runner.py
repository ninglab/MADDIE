from pathlib import Path

from ..core import write_optimized_csv
from .direct_optimizer import DirectOptimizer
from ..errors import LeadOptimizerRequestError
from ..utils.logging import get_logger
from ..tools import OptimizationTool, ToolChest


logger = get_logger()


BENCHMARK_ORACLE_BUDGET_MULTIPLIER = 2
BENCHMARK_ORACLE_BUDGET_MINIMUM = 200
TARGET_RANGE_TOLERANCES = {
    "logp": 0.01,
    "mol_wt": 0.1,
    "tpsa": 0.1,
    "hbd": 0.0,
    "hba": 0.0,
    "rot_bonds": 0.0,
}


class AdaptiveOptimizerRunner:
    """Run a selected optimizer and adapt when scored outputs miss the target."""

    def __init__(
        self,
        tool_chest: ToolChest,
        max_attempts: int = 1,
        molopt_fallback_attempts: int = 4,
    ):
        self.tool_chest = tool_chest
        self.max_attempts = max_attempts
        self.molopt_fallback_attempts = molopt_fallback_attempts

    def optimize_csv(
        self,
        selected_tool: OptimizationTool,
        decision_packet: dict,
        input_csv: Path,
        output_csv: Path,
        adjustment: str,
        num_outputs: int,
        objective,
        allow_fallback: bool = True,
    ):
        candidate_tools = self._candidate_tools(
            selected_tool,
            decision_packet,
            allow_fallback,
        )
        best_result = None
        best_progress = None
        best_settings = {}
        best_attempt_index = None
        attempts = []
        debug_dir = output_csv.parent / "debug"
        logger.info(
            "Candidate optimizer tools: %s.",
            ", ".join(tool.name for tool in candidate_tools),
        )

        for index, tool in enumerate(candidate_tools, start=1):
            configured_oracle_budget = getattr(tool, "max_oracle_calls", None)
            settings = self._apply_benchmark_settings(tool, decision_packet)
            attempt_output = output_csv
            if len(candidate_tools) > 1:
                debug_dir.mkdir(parents=True, exist_ok=True)
                attempt_output = (
                    debug_dir
                    / f"{output_csv.stem}_{index}_{tool.name}{output_csv.suffix}"
                )
            logger.info(
                "Running optimizer attempt %s/%s with %s.",
                index,
                len(candidate_tools),
                tool.name,
            )
            if index > 1:
                logger.warning(
                    "Trying fallback optimizer %s after the prior attempt did "
                    "not satisfy the target.",
                    tool.name,
                )
            if settings:
                logger.info(
                    "Oracle budget for %s: using up to %s calls per input "
                    "molecule (%s).",
                    tool.name,
                    settings["max_oracle_calls_per_molecule"],
                    self._format_oracle_budget_source(settings),
                )
                logger.debug("execution settings:\n%s", settings)

            try:
                try:
                    result = DirectOptimizer(
                        tool=tool,
                        num_outputs=num_outputs,
                        max_attempts=self.max_attempts,
                    ).optimize_csv(
                        input_csv=input_csv,
                        output_csv=attempt_output,
                        adjustment=adjustment,
                    )
                finally:
                    if configured_oracle_budget is not None:
                        tool.max_oracle_calls = configured_oracle_budget
            except LeadOptimizerRequestError:
                raise
            except Exception as exc:
                attempts.append({
                    "tool": tool.name,
                    "output_csv": str(attempt_output),
                    "execution_settings": settings,
                    "error": str(exc),
                })
                logger.warning(
                    "Optimizer attempt with %s failed; continuing when a fallback "
                    "is available: %s",
                    tool.name,
                    exc,
                )
                continue

            progress = self._target_progress(result, objective)
            attempts.append({
                "tool": tool.name,
                "output_csv": str(result.output_csv),
                "execution_settings": settings,
                "failed_input_ids": result.failed_input_ids or [],
                "partial": result.partial,
                "row_count": result.row_count,
                "molecule_count": result.molecule_count,
                **progress,
            })
            if progress.get("target_hits") is None:
                logger.info(
                    "Optimizer attempt with %s produced validated SMILES; "
                    "property target scoring is unavailable for this output.",
                    tool.name,
                )
            else:
                logger.info(
                    "Optimizer attempt with %s produced %s/%s property target "
                    "hit(s).",
                    tool.name,
                    progress.get("target_hits", "n/a"),
                    progress.get("scored_outputs", "n/a"),
                )

            if self._is_better_progress(progress, best_progress, objective):
                best_result = result
                best_progress = progress
                best_settings = settings
                best_attempt_index = len(attempts) - 1

            if self._target_satisfied(progress):
                logger.info("Target satisfied by %s.", tool.name)
                break

        if best_result is None:
            errors = [
                f"{attempt['tool']}: {attempt['error']}"
                for attempt in attempts
                if attempt.get("error")
            ]
            detail = "; ".join(errors) if errors else "No optimization attempt was executed."
            raise LeadOptimizerRequestError(
                f"All optimization attempts failed. {detail}",
                status="failed",
                metadata={
                    "error_category": "optimizer_failed",
                    "optimization_attempts": attempts,
                    "action_notes": [
                        "No optimized output was produced because every "
                        "optimizer attempt failed.",
                        *errors,
                    ],
                },
            )
        if best_attempt_index is not None:
            attempts[best_attempt_index]["selected_output"] = True
        if Path(best_result.output_csv) != output_csv:
            write_optimized_csv(output_csv, best_result.results)
            best_result.output_csv = str(output_csv)
        return best_result, best_settings, attempts

    def _target_satisfied(self, progress: dict) -> bool:
        target_hits = progress.get("target_hits")
        scored_outputs = progress.get("scored_outputs")
        return (
            target_hits is not None
            and scored_outputs is not None
            and scored_outputs > 0
            and target_hits >= scored_outputs
        )

    def _candidate_tools(
        self,
        selected_tool: OptimizationTool,
        decision_packet: dict,
        allow_fallback: bool,
    ) -> list[OptimizationTool]:
        if (
            not allow_fallback
            or self.molopt_fallback_attempts <= 0
            or not getattr(selected_tool, "supports_adaptive_fallback", False)
        ):
            return [selected_tool]

        comparison = decision_packet.get("threshold_comparison") or {}
        ordered_results = comparison.get("ordered_results") or []
        available_results = [
            result
            for result in ordered_results
            if result.get("tool") in self.tool_chest.available()
        ]
        selected_result = next(
            (
                result
                for result in available_results
                if result.get("tool") == selected_tool.name
            ),
            None,
        )
        if selected_result is None:
            return [selected_tool]

        alternatives = [
            result
            for result in available_results
            if result.get("tool") != selected_tool.name
        ]
        alternatives.sort(
            key=lambda result: (
                -(result.get("mean_final_top10") or -1.0),
                (
                    result.get("oracle_calls")
                    if result.get("oracle_calls") is not None
                    else float("inf")
                ),
            )
        )
        names = [selected_tool.name] + [
            result["tool"]
            for result in alternatives[:self.molopt_fallback_attempts]
        ]
        return [self.tool_chest.get(name) for name in names]

    def _format_oracle_budget_source(self, settings: dict) -> str:
        if settings.get("oracle_budget_source") != "benchmark":
            return "configured default"

        benchmark_calls = settings["benchmark_calls_to_target"]
        recommended = settings.get("benchmark_recommended_calls_per_molecule")
        configured = settings.get("configured_max_oracle_calls_per_molecule")
        actual = settings["max_oracle_calls_per_molecule"]
        source = (
            f"benchmark reached target in {benchmark_calls} calls; "
            f"2x recommendation={recommended}"
        )
        if configured is not None and actual < recommended:
            source += f"; capped by configured max={configured}"
        return source

    def _apply_benchmark_settings(
        self,
        selected_tool: OptimizationTool,
        decision_packet: dict,
    ) -> dict:
        comparison = decision_packet.get("threshold_comparison") or {}
        ordered_results = comparison.get("ordered_results") or []
        selected = next(
            (
                result
                for result in ordered_results
                if result.get("tool") == selected_tool.name
            ),
            None,
        )
        if not hasattr(selected_tool, "max_oracle_calls"):
            return {}

        configured_budget = int(selected_tool.max_oracle_calls)
        if not selected:
            return {
                "oracle_budget_source": "configured default",
                "max_oracle_calls_per_molecule": configured_budget,
            }

        settings = {
            "oracle_budget_source": "configured default",
            "max_oracle_calls_per_molecule": configured_budget,
        }
        oracle_calls = selected.get("oracle_calls")
        if oracle_calls is not None:
            oracle_calls = int(oracle_calls)
            recommended_budget = max(
                BENCHMARK_ORACLE_BUDGET_MINIMUM,
                oracle_calls * BENCHMARK_ORACLE_BUDGET_MULTIPLIER,
            )
            budget = min(
                configured_budget,
                recommended_budget,
            )
            selected_tool.max_oracle_calls = budget
            settings = {
                "oracle_budget_source": "benchmark",
                "benchmark_calls_to_target": oracle_calls,
                "benchmark_recommended_calls_per_molecule": recommended_budget,
                "configured_max_oracle_calls_per_molecule": configured_budget,
                "max_oracle_calls_per_molecule": budget,
            }
        return settings

    def _target_progress(self, result, objective) -> dict:
        if objective:
            mismatched_properties = sorted({
                output.property
                for output in result.results
                if (
                    output.property is not None
                    and output.property != objective.property
                )
            })
            if mismatched_properties:
                logger.warning(
                    "Ignoring scored outputs for %s because the requested "
                    "objective is %s.",
                    ", ".join(mismatched_properties),
                    objective.property,
                )

        scored_values = [
            output.optimized_property_value
            for output in result.results
            if (
                output.optimized_property_value is not None
                and (
                    not objective
                    or output.property == objective.property
                )
            )
        ]
        if not objective or not scored_values:
            return {
                "target_hits": None,
                "scored_outputs": len(scored_values),
                "validated_outputs": result.row_count,
            }

        if objective.direction == "increase":
            best_value = max(scored_values)
            target_hits = (
                sum(value >= objective.threshold for value in scored_values)
                if objective.threshold is not None
                else None
            )
        elif objective.direction == "decrease":
            best_value = min(scored_values)
            target_hits = (
                sum(value <= objective.threshold for value in scored_values)
                if objective.threshold is not None
                else None
            )
        else:
            midpoint = (objective.minimum + objective.maximum) / 2
            best_value = min(scored_values, key=lambda value: abs(value - midpoint))
            tolerance = TARGET_RANGE_TOLERANCES.get(objective.property, 0.0)
            target_hits = sum(
                objective.minimum - tolerance
                <= value
                <= objective.maximum + tolerance
                for value in scored_values
            )

        return {
            "target_hits": target_hits,
            "scored_outputs": len(scored_values),
            "best_property_value": best_value,
            "validated_outputs": result.row_count,
        }

    def _is_better_progress(self, progress: dict, best: dict | None, objective) -> bool:
        if best is None:
            return True
        progress_hits = progress.get("target_hits")
        best_hits = best.get("target_hits")
        if progress_hits is not None and best_hits is None:
            return True
        if progress_hits is None and best_hits is not None:
            return False
        if progress_hits is not None and best_hits is not None:
            if progress_hits != best_hits:
                return progress_hits > best_hits

        progress_scored = progress.get("scored_outputs", 0)
        best_scored = best.get("scored_outputs", 0)
        if progress_scored != best_scored:
            return progress_scored > best_scored

        progress_valid = progress.get("validated_outputs")
        best_valid = best.get("validated_outputs")
        if progress_valid is not None and best_valid is not None:
            if progress_valid != best_valid:
                return progress_valid > best_valid

        progress_value = progress.get("best_property_value")
        best_value = best.get("best_property_value")
        if progress_value is None or best_value is None or objective is None:
            return False
        if objective.direction == "decrease":
            return progress_value < best_value
        if objective.direction == "target_range":
            midpoint = (objective.minimum + objective.maximum) / 2
            return abs(progress_value - midpoint) < abs(best_value - midpoint)
        return progress_value > best_value
