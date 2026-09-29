import csv
from pathlib import Path
import time
from .utils.logging import get_logger, log_run_exceptions
from .core import (
    extract_optimization_adjustment,
    find_csv_in_prompt,
    find_num_outputs_in_prompt,
    resolve_input_csv,
    write_optimized_csv,
)
from .execution.adaptive_runner import AdaptiveOptimizerRunner
from .strategy import AgentActionPlanner, DecisionController, NextActionAdvisor
from .errors import LeadOptimizerRequestError
from .oracles.specs import format_objective, format_objectives, prompt_objective_fields
from .strategy import StrategyContext, ToolSelectionAdvisor
from .response import build_response_summary, debug_output_paths
from .status import scored_target_status
from .tools import FakeOptimizerTool, OptimizationTool, ToolChest
from .planner import RuleBasedPlanner, SmartPlanner, find_excluded_tools


logger = get_logger()


class LeadOptimizerAgent:
    """
    Unified lead optimization agent: NL prompt → tool selection → execution → output.

    Simple, modular flow:
    1. Parse user prompt with SmartPlanner
    2. Select appropriate tool (GeLLMO, MolOpt, etc.)
    3. Run optimization deterministically
    4. Return results and metadata
    """

    def __init__(
        self,
        tool_chest: ToolChest | None = None,
        tool: OptimizationTool | None = None,
        tools: list[OptimizationTool] | None = None,
        planner=None,
        num_outputs: int = 5,
        max_attempts: int = 1,
        molopt_fallback_attempts: int = 4,
        adaptive_runner: AdaptiveOptimizerRunner | None = None,
        response_llm=None,
        use_llm_response_summary: bool = True,
        tool_selection_llm=None,
        use_llm_tool_selection: bool = False,
        decision_llm=None,
        use_llm_next_action: bool = False,
        execute_next_actions: bool = False,
        max_tool_actions: int = 2,
        output_selection_policy: str = "best_attempt",
    ):
        """
        Args:
            tool_chest: ToolChest with registered optimization tools.
                        If None, uses FakeOptimizerTool for testing.
            planner: Custom planner (default: SmartPlanner with heuristics).
            num_outputs: Default number of outputs per molecule.
            max_tool_actions: Total optimizer/tool executions allowed for one
                              agent.run call, including the initial run. LLM
                              planning, next-action decisions, and final
                              response summarization are tracked separately and
                              do not count as tool actions.
        """
        has_explicit_chest = tool_chest is not None
        has_tool_collection = bool(tools)
        self.tool_chest = tool_chest or ToolChest()

        if tool is not None:
            default_tool = tool
            self.tool_chest.add(tool.name, tool)
        else:
            default_tool = FakeOptimizerTool()
            if not has_explicit_chest and not has_tool_collection:
                self.tool_chest.add(default_tool.name, default_tool)

        for extra_tool in tools or []:
            self.tool_chest.add(extra_tool.name, extra_tool)

        if not self.tool_chest.available():
            # Add a fake tool for testing if chest is empty
            self.tool_chest.add(default_tool.name, default_tool)

        self.num_outputs = num_outputs
        self.max_attempts = max_attempts
        self.adaptive_runner = adaptive_runner or AdaptiveOptimizerRunner(
            self.tool_chest,
            max_attempts=max_attempts,
            molopt_fallback_attempts=molopt_fallback_attempts,
        )

        if planner is None:
            if has_explicit_chest or has_tool_collection:
                # SmartPlanner uses heuristics to select tools based on NL hints.
                self.planner = SmartPlanner(
                    available_tools=self.tool_chest.tool_hints(),
                    default_num_outputs=num_outputs,
                )
            else:
                # Preserve the original deterministic default path.
                self.planner = RuleBasedPlanner(
                    default_tool=default_tool.name,
                    default_num_outputs=num_outputs,
                )
        else:
            self.planner = planner

        if hasattr(self.planner, "with_tools"):
            self.planner.with_tools(self.tool_chest.documentation())
        if hasattr(self.planner, "with_tool_capabilities"):
            self.planner.with_tool_capabilities(
                self.tool_chest.multi_property_tools()
            )
        self.response_llm = response_llm
        self.use_llm_response_summary = use_llm_response_summary
        self.tool_selection_advisor = ToolSelectionAdvisor(
            llm=tool_selection_llm or response_llm,
            enabled=use_llm_tool_selection,
        )
        self.max_tool_actions = max(1, int(max_tool_actions))
        followup_actions = max(0, self.max_tool_actions - 1)
        self.max_decision_rounds = followup_actions + 1
        self.execute_next_actions = execute_next_actions
        self.max_agent_steps = followup_actions
        if output_selection_policy != "best_attempt":
            raise ValueError(
                "Only output_selection_policy='best_attempt' is currently supported."
            )
        self.output_selection_policy = output_selection_policy
        self.next_action_advisor = NextActionAdvisor(
            llm=decision_llm or response_llm,
            enabled=use_llm_next_action,
        )
        self.decision_controller = DecisionController(
            advisor=self.next_action_advisor,
            max_decision_rounds=self.max_decision_rounds,
            execute_next_actions=self.execute_next_actions,
            max_agent_steps=self.max_agent_steps,
            output_selection_policy=self.output_selection_policy,
        )
        self.action_planner = AgentActionPlanner(
            self.tool_chest,
            max_attempts=self.max_attempts,
        )

    @log_run_exceptions
    def run(self, prompt: str, path_to_env: str, iterations: int = 1) -> tuple[str, dict]:
        """
        Execute lead optimization from prompt.

        Args:
            prompt: Natural language instruction (e.g., "Optimize sample.csv with GeLLMO
                    to increase QED above 0.85, give me 3 variants").
            path_to_env: Path to working directory with input/, output/ subdirs.
            iterations: Run optimization N times, using outputs of round i as inputs
                       to round i+1 (for refinement). Default=1 (single pass).

        Returns:
            (response_text, metadata_dict)
                response_text: User-friendly summary
                metadata_dict: Tool name, paths, molecule counts, etc.
        """
        env = Path(path_to_env)
        run_started = time.perf_counter()
        self._reset_llm_responses()
        out_dir = env / "output" / "lead_optimizer"
        logger.info("'LeadOptimizer' run started.")
        logger.debug("user prompt:\n%s", prompt)
        logger.info("Planning optimization request.")

        # Plan: parse prompt to determine tool, CSV file, oracle, etc.
        plan = self.planner.plan(prompt)
        # requested_count = find_num_outputs_in_prompt(prompt)
        prompt_file = find_csv_in_prompt(prompt)
        plan.adjustment = extract_optimization_adjustment(prompt)
        plan.num_outputs = self.num_outputs  # always use the configured value, not prompt-parsed count
        plan.input_file = prompt_file or plan.input_file
        decision_packet = getattr(self.planner, "last_decision_packet", {})
        requested_properties = self._requested_properties(plan, decision_packet)
        plan.requested_properties = requested_properties
        plan.objective_mode = (
            "multi_property" if len(requested_properties) > 1 else "single_property"
        )
        prompt_fields = prompt_objective_fields(prompt)
        plan.objectives = prompt_fields["objectives"] or self._objectives(plan)
        scoring_objective = (
            prompt_fields["objective"]
            if plan.objective_mode == "multi_property"
            else (plan.objectives[0] if plan.objectives else None)
        )
        plan.objective = scoring_objective
        objective_text = self._format_objective(
            scoring_objective,
            objectives=plan.objectives,
            requested_properties=requested_properties,
            objective_mode=plan.objective_mode,
        )
        logger.info(
            "Parsed plan: tool=%s, input=%s, outputs=%s, objective=%s.",
            plan.tool,
            plan.input_file or "(auto)",
            plan.num_outputs,
            objective_text,
        )
        logger.debug("parsed plan:\n%s", plan.to_dict())

        if not plan.objectives:
            raise LeadOptimizerRequestError(
                "I could not confidently identify which property to optimize. "
                "Please name a property such as QED, SAScore, LogP, molecular "
                "weight, TPSA, hERG, DILI, ClinTox, mutagenicity, or "
                "carcinogenicity, and include the desired direction or target.",
                status="needs_clarification",
                metadata={"error_category": "objective_missing"},
            )

        tool_selection = self._select_initial_tool(
            prompt=prompt,
            plan=plan,
            decision_packet=decision_packet,
        )
        if tool_selection.tool != plan.tool:
            logger.info(
                "Tool-selection advisor changed tool from %s to %s.",
                plan.tool,
                tool_selection.tool,
            )
            plan.tool = tool_selection.tool
            plan.reason = tool_selection.rationale or plan.reason

        # Validate final tool exists after optional advisor review.
        if plan.tool not in self.tool_chest.available():
            available = ", ".join(self.tool_chest.available())
            raise LeadOptimizerRequestError(
                f"Tool '{plan.tool}' is not configured. Available tools: "
                f"{available}. Request an available tool or update the tool chest.",
                metadata={
                    "error_category": "tool_unavailable",
                    "requested_tool": plan.tool,
                    "available_tools": self.tool_chest.available(),
                },
            )

        selected_tool = self.tool_chest.get(plan.tool)
        logger.info("Selected tool: %s.", selected_tool.name)
        logger.debug("decision packet:\n%s", decision_packet)
        logger.debug("tool-selection decision:\n%s", tool_selection.to_dict())

        # Resolve input/output paths, repairing a missing filename when there is
        # exactly one unambiguous CSV in the environment.
        input_csv, path_warning = resolve_input_csv(env, plan.input_file)
        plan.input_file = input_csv.name
        output_csv = out_dir / f"{input_csv.stem}_optimized.csv"
        if input_csv.parent == env:
            output_csv = env / f"{input_csv.stem}_output.csv"
        logger.info("Input CSV: %s.", input_csv)
        logger.info("Output CSV: %s.", output_csv)
        if path_warning:
            logger.warning("Input cleanup: %s", path_warning)

        # Execute optimization, with optional iteration
        current_input = input_csv
        final_result = None
        input_warnings = [path_warning] if path_warning else []
        failed_input_ids = []
        execution_settings = {}
        optimization_attempts = []
        incomplete_run = False

        for iteration in range(iterations):
            logger.info(
                "Starting optimization iteration %s/%s with %s.",
                iteration + 1,
                iterations,
                selected_tool.name,
            )
            result, execution_settings, optimization_attempts = (
                self.adaptive_runner.optimize_csv(
                    selected_tool=selected_tool,
                    decision_packet=decision_packet,
                    input_csv=current_input,
                    output_csv=output_csv,
                    adjustment=plan.adjustment,
                    num_outputs=plan.num_outputs,
                    objective=scoring_objective,
                    allow_fallback=(
                        iteration == 0
                        and iterations == 1
                        and not (
                            self.execute_next_actions
                            and self.max_decision_rounds > 0
                            and self.next_action_advisor.enabled
                        )
                    ),
                )
            )
            final_result = result
            input_warnings.extend(result.input_warnings or [])
            failed_input_ids.extend(result.failed_input_ids or [])
            incomplete_run = incomplete_run or result.partial
            logger.info(
                "Iteration %s complete: tool=%s, rows=%s.",
                iteration + 1,
                result.tool,
                result.row_count,
            )

            # If iterating, use this round's output as next round's input
            if iterations > 1 and iteration < iterations - 1:
                next_input = out_dir / f"{input_csv.stem}_iter{iteration+2}_input.csv"
                self._write_iteration_input(output_csv, next_input)
                current_input = next_input
                # Rename output for next iteration to avoid overwrite
                output_csv = out_dir / f"{input_csv.stem}_optimized_iter{iteration+2}.csv"

        # Build response
        final_result.input_warnings = list(dict.fromkeys(input_warnings))
        final_result.failed_input_ids = list(dict.fromkeys(failed_input_ids))
        has_partial_output = incomplete_run and final_result.row_count > 0
        status = "partial" if has_partial_output else "completed"
        if incomplete_run and final_result.row_count == 0:
            status = "failed"
        execution_reason = self._build_execution_reason(
            plan_reason=plan.reason,
            plan_tool=plan.tool,
            final_tool=final_result.tool,
            attempts=optimization_attempts,
        )
        stopping_reason = self._build_stopping_reason(
            status=status,
            result=final_result,
            attempts=optimization_attempts,
        )
        logger.info("Execution reason: %s", execution_reason)
        logger.info("Stopping reason: %s", stopping_reason["message"])
        logger.debug(
            "optimization attempts:\n%s",
            self._format_optimization_attempts(optimization_attempts),
        )
        self._log_scored_outputs(final_result.results)
        metadata = {
            "tool": final_result.tool,
            "available_tools": self.tool_chest.available(),
            "planner": getattr(self.planner, "name", self.planner.__class__.__name__),
            "plan": plan.to_dict(),
            "execution_reason": execution_reason,
            "stopping_reason": stopping_reason["reason"],
            "stopping_reason_message": stopping_reason["message"],
            "benchmark_evidence": getattr(
                self.planner,
                "last_benchmark_evidence",
                {},
            ),
            "decision_packet": getattr(
                self.planner,
                "last_decision_packet",
                {},
            ),
            "tool_selection": tool_selection.to_dict(),
            "requested_properties": requested_properties,
            "objective_mode": plan.objective_mode,
            "objectives": [
                objective.to_dict() for objective in plan.objectives or []
            ],
            "scoring_objective": (
                scoring_objective.to_dict()
                if scoring_objective is not None
                else None
            ),
            "status": status,
            "partial_output": has_partial_output,
            "input_type": "csv",
            "iterations": iterations,
            "molecule_count": final_result.molecule_count,
            "row_count": final_result.row_count,
            "expected_output_rows": (
                final_result.molecule_count * plan.num_outputs
                if final_result.molecule_count and plan.num_outputs
                else None
            ),
            "failed_input_ids": final_result.failed_input_ids or [],
            "input_warnings": final_result.input_warnings or [],
            "skipped_input_rows": final_result.skipped_input_rows,
            "request_schema": "OptimizationRequest",
            "output_selection_policy": self.output_selection_policy,
            "execution_settings": execution_settings,
            "optimization_attempts": optimization_attempts,
            "runtime": self._runtime_summary(
                run_started,
                optimization_attempts,
                final_result.molecule_count,
            ),
            "outputs": {
                "csv": str(final_result.output_csv),
                "debug_csvs": debug_output_paths(
                    optimization_attempts,
                    Path(final_result.output_csv),
                ),
                "log": None,
            },
            "error_category": "optimization_failed" if status == "failed" else None,
        }
        metadata["strategy_trace"] = [
            {
                "phase": "pre_run",
                "source": tool_selection.source,
                "action_type": "optimize_csv",
                "tool": tool_selection.tool,
                "reason": tool_selection.rationale,
                "confidence": tool_selection.confidence,
                "warnings": tool_selection.warnings,
                "executed": True,
            }
        ]

        def refresh_metadata():
            nonlocal status, execution_reason, stopping_reason
            final_result.input_warnings = list(dict.fromkeys(input_warnings))
            final_result.failed_input_ids = list(dict.fromkeys(failed_input_ids))
            has_partial = incomplete_run and final_result.row_count > 0
            status = "partial" if has_partial else "completed"
            if incomplete_run and final_result.row_count == 0:
                status = "failed"
            execution_reason = self._build_execution_reason(
                plan_reason=plan.reason,
                plan_tool=plan.tool,
                final_tool=final_result.tool,
                attempts=optimization_attempts,
            )
            stopping_reason = self._build_stopping_reason(
                status=status,
                result=final_result,
                attempts=optimization_attempts,
            )
            metadata.update({
                "tool": final_result.tool,
                "execution_reason": execution_reason,
                "stopping_reason": stopping_reason["reason"],
                "stopping_reason_message": stopping_reason["message"],
                "status": status,
                "partial_output": has_partial,
                "molecule_count": final_result.molecule_count,
                "row_count": final_result.row_count,
                "requested_outputs_per_input": plan.num_outputs,
                "expected_total_outputs": (
                    final_result.molecule_count * plan.num_outputs
                    if final_result.molecule_count and plan.num_outputs
                    else None
                ),
                "failed_input_ids": final_result.failed_input_ids or [],
                "input_warnings": final_result.input_warnings or [],
                "skipped_input_rows": final_result.skipped_input_rows,
                "execution_settings": execution_settings,
                "optimization_attempts": optimization_attempts,
                "runtime": self._runtime_summary(
                    run_started,
                    optimization_attempts,
                    final_result.molecule_count,
                ),
                "outputs": {
                    "csv": str(final_result.output_csv),
                    "debug_csvs": debug_output_paths(
                        optimization_attempts,
                        Path(final_result.output_csv),
                    ),
                    "log": None,
                },
            })

        def decision_context():
            return self._next_action_context(
                prompt=prompt,
                plan=plan,
                status=status,
                result=final_result,
                iterations=iterations,
                execution_reason=execution_reason,
                optimization_attempts=optimization_attempts,
                metadata=metadata,
            )

        def execute_decision_action(decision: dict, step: int) -> dict:
            nonlocal final_result, execution_settings, incomplete_run
            policy_action = decision.get("policy_action")
            if policy_action == "retry_same_tool":
                action_tool, action_strategy = self.action_planner.prepare_retry_tool(
                    self.tool_chest.get(final_result.tool),
                    step,
                    attempts=optimization_attempts,
                    decision=decision,
                )
            elif policy_action == "switch_tool":
                action_tool, action_strategy = self.action_planner.prepare_switch_tool(
                    decision=decision,
                    decision_packet=decision_packet,
                    attempts=optimization_attempts,
                )
                if action_tool is None:
                    return {
                        "executed": False,
                        "action": policy_action,
                        "reason": action_strategy["reason"],
                    }
            else:
                return {
                    "executed": False,
                    "reason": f"Policy action {policy_action} is not executable yet.",
                }

            final_output = Path(metadata["outputs"]["csv"])
            action_output = (
                final_output.parent
                / "debug"
                / f"{final_output.stem}_agent_step{step}_{action_tool.name}{final_output.suffix}"
            )
            action_adjustment = plan.adjustment
            if (
                policy_action == "retry_same_tool"
                and action_strategy.get("retry_instruction")
            ):
                action_adjustment = (
                    f"{plan.adjustment}\n\n"
                    "Agent retry strategy: "
                    f"{action_strategy['retry_instruction']}"
                )

            try:
                action_result, action_settings, action_attempts = self.adaptive_runner.optimize_csv(
                    selected_tool=action_tool,
                    decision_packet=decision_packet,
                    input_csv=current_input,
                    output_csv=action_output,
                    adjustment=action_adjustment,
                    num_outputs=plan.num_outputs,
                    objective=plan.objective,
                    allow_fallback=False,
                )
            except Exception as exc:
                return {
                    "executed": False,
                    "action": policy_action,
                    "tool": action_tool.name,
                    "error": str(exc),
                }
            for attempt in action_attempts:
                attempt["agent_step"] = step
                attempt["controller_action"] = policy_action
                attempt["agent_action_strategy"] = action_strategy

            current_attempt = self.action_planner.selected_attempt(
                optimization_attempts
            )
            action_attempt = self.action_planner.selected_attempt(action_attempts)
            action_selected = self.action_planner.is_better_attempt(
                action_attempt,
                current_attempt,
                plan.objective,
            )
            if action_selected:
                for attempt in optimization_attempts:
                    attempt["selected_output"] = False
                write_optimized_csv(final_output, action_result.results)
                action_result.output_csv = str(final_output)
                final_result = action_result
                execution_settings = action_settings
            else:
                for attempt in action_attempts:
                    attempt["selected_output"] = False

            optimization_attempts.extend(action_attempts)
            input_warnings.extend(action_result.input_warnings or [])
            failed_input_ids.extend(action_result.failed_input_ids or [])
            incomplete_run = incomplete_run or action_result.partial
            refresh_metadata()
            return {
                "executed": True,
                "action": policy_action,
                "tool": action_tool.name,
                "selected_as_final": action_selected,
                "output_csv": str(action_output),
                "action_strategy": action_strategy,
                "target_hits": action_attempt.get("target_hits"),
                "scored_outputs": action_attempt.get("scored_outputs"),
                "validated_outputs": action_attempt.get("validated_outputs")
                or action_attempt.get("row_count"),
                "row_count": action_attempt.get("row_count"),
            }

        decision_trace = self.decision_controller.run(
            context_builder=decision_context,
            action_executor=execute_decision_action,
        )
        refresh_metadata()
        metadata["decision_trace"] = decision_trace
        metadata["runtime"] = self._runtime_summary(
            run_started,
            optimization_attempts,
            final_result.molecule_count,
        )
        metadata["reasoning_limits"] = self._reasoning_limits_summary(decision_trace)
        metadata["strategy_trace"] = [
            metadata["strategy_trace"][0],
            *[
                {
                    "phase": item.get("phase", "post_run"),
                    "source": item.get("source"),
                    "action_type": item.get("policy_action"),
                    "tool": item.get("tool"),
                    "reason": item.get("policy_reason"),
                    "confidence": item.get("confidence"),
                    "warnings": [
                        item.get("error")
                    ] if item.get("error") else [],
                    "executed": item.get("executed", False),
                    "execution_result": item.get("execution_result"),
                    "llm_recommendation": item.get("llm_recommendation"),
                }
                for item in decision_trace
            ],
        ]
        next_action = decision_trace[-1] if decision_trace else None
        metadata["next_action"] = next_action
        if next_action and next_action.get("source") != "disabled":
            logger.info(
                "Next-action advisor: %s (%s confidence).",
                next_action.get("action"),
                next_action.get("confidence"),
            )
            logger.debug("next-action recommendation:\n%s", next_action)
        response_text = build_response_summary(
            prompt=prompt,
            result=final_result,
            output_csv=output_csv,
            env=env,
            iterations=iterations,
            status=status,
            stopping_reason=stopping_reason["reason"],
            stopping_reason_message=stopping_reason["message"],
            objective=scoring_objective,
            objectives=plan.objectives,
            requested_properties=requested_properties,
            objective_mode=plan.objective_mode,
            requested_outputs_per_input=plan.num_outputs,
            execution_reason=execution_reason,
            optimization_attempts=optimization_attempts,
            reasoning_limits=metadata["reasoning_limits"],
            metadata=metadata,
            planner=self.planner,
            response_llm=self.response_llm,
            use_llm=self.use_llm_response_summary,
        )
        metadata["responses"] = self._collect_llm_responses()
        logger.info(
            "'LeadOptimizer' run completed in %.1f seconds: wrote %s rows to %s.",
            metadata["runtime"]["elapsed_seconds"],
            final_result.row_count,
            final_result.output_csv,
        )

        return response_text, metadata

    def _runtime_summary(
        self,
        run_started: float,
        optimization_attempts: list[dict],
        molecule_count: int,
    ) -> dict:
        elapsed_seconds = time.perf_counter() - run_started
        molopt_budgets = []
        for attempt in optimization_attempts:
            settings = attempt.get("execution_settings") or {}
            budget = settings.get("max_oracle_calls_per_molecule")
            if budget is not None:
                molopt_budgets.append(int(budget))

        max_budget = max(molopt_budgets) if molopt_budgets else None
        total_budget = (
            sum(molopt_budgets) * molecule_count
            if molopt_budgets and molecule_count
            else None
        )
        note = None
        if max_budget is not None:
            note = (
                "MolOpt attempt runtime can scale with oracle budget; this run "
                f"used up to {max_budget} oracle calls per input molecule."
            )
            if total_budget is not None:
                note += (
                    f" Across {molecule_count} input molecule(s) and "
                    f"{len(molopt_budgets)} MolOpt attempt(s), the configured "
                    f"upper-bound oracle-call budget was {total_budget}."
                )

        return {
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_minutes": round(elapsed_seconds / 60, 2),
            "optimizer_attempt_count": len(optimization_attempts),
            "molopt_attempt_count": len(molopt_budgets),
            "max_oracle_calls_per_molecule": max_budget,
            "estimated_total_oracle_call_budget": total_budget,
            "note": note,
        }

    def _reasoning_limits_summary(self, decision_trace: list[dict]) -> dict:
        followup_actions_executed = sum(
            1 for decision in decision_trace if decision.get("executed")
        )
        tool_actions_executed = 1 + followup_actions_executed
        last_decision = decision_trace[-1] if decision_trace else {}
        stopped_by = "disabled"
        if self.next_action_advisor.enabled:
            stopped_by = "policy_stop"
            if not decision_trace:
                stopped_by = "no_decision_trace"
            elif (
                last_decision.get("executed")
                and len(decision_trace) >= self.max_decision_rounds
            ):
                stopped_by = "decision_round_limit"
            elif (
                self.execute_next_actions
                and not last_decision.get("executed")
                and last_decision.get("policy_action") in {"retry_same_tool", "switch_tool"}
                and followup_actions_executed >= self.max_agent_steps
            ):
                stopped_by = "tool_action_limit"
            elif (
                self.execute_next_actions
                and followup_actions_executed >= self.max_agent_steps
                and last_decision.get("legal_actions", {}).get("can_execute_more") is False
            ):
                stopped_by = "tool_action_limit"

        return {
            "reasoning_mode": (
                "bounded_llm" if self.next_action_advisor.enabled else "disabled"
            ),
            "llm_next_action_enabled": self.next_action_advisor.enabled,
            "execute_next_actions": self.execute_next_actions,
            "max_decision_rounds": self.max_decision_rounds,
            "max_tool_actions": self.max_tool_actions,
            "max_followup_tool_actions": self.max_agent_steps,
            "decision_rounds_used": len(decision_trace),
            "tool_actions_executed": tool_actions_executed,
            "followup_tool_actions_executed": followup_actions_executed,
            "remaining_decision_rounds": max(
                0,
                self.max_decision_rounds - len(decision_trace),
            ),
            "remaining_tool_actions": max(
                0,
                self.max_tool_actions - tool_actions_executed,
            ),
            "stopped_by": stopped_by,
        }

    def _llm_response_sources(self) -> list:
        sources = []
        for llm in (
            getattr(self.planner, "llm", None),
            self.response_llm,
            getattr(self.tool_selection_advisor, "llm", None),
            getattr(self.next_action_advisor, "llm", None),
        ):
            if llm is None or not hasattr(llm, "responses"):
                continue
            if not any(source is llm for source in sources):
                sources.append(llm)
        return sources

    def _reset_llm_responses(self) -> None:
        for llm in self._llm_response_sources():
            llm.responses = []

    def _collect_llm_responses(self) -> list:
        responses = []
        for llm in self._llm_response_sources():
            responses.extend(getattr(llm, "responses", []))
        return responses

    def _select_initial_tool(
        self,
        prompt: str,
        plan,
        decision_packet: dict,
    ):
        context = self._strategy_context(
            prompt=prompt,
            phase="pre_run",
            plan=plan,
            decision_packet=decision_packet,
            remaining_strategy_steps=self.max_decision_rounds,
            remaining_tool_steps=self.max_agent_steps,
        )
        return self.tool_selection_advisor.recommend(context)

    def _strategy_context(
        self,
        prompt: str,
        phase: str,
        plan,
        decision_packet: dict,
        prior_attempts: list[dict] | None = None,
        target_status: str | None = None,
        target_hits: int | None = None,
        scored_outputs: int | None = None,
        best_property_value: float | None = None,
        remaining_strategy_steps: int | None = None,
        remaining_tool_steps: int | None = None,
    ) -> StrategyContext:
        return StrategyContext(
            prompt=prompt,
            phase=phase,
            input_file=plan.input_file,
            objective=plan.objective.to_dict() if plan.objective else None,
            adjustment=plan.adjustment,
            available_tools=self.tool_chest.available(),
            excluded_tools=self._excluded_tools(prompt),
            tool_docs=self.tool_chest.documentation(),
            deterministic_tool=plan.tool,
            decision_packet=decision_packet,
            benchmark_evidence=getattr(self.planner, "last_benchmark_evidence", {}),
            benchmark_is_imperfect=True,
            prior_attempts=prior_attempts or [],
            target_status=target_status,
            target_hits=target_hits,
            scored_outputs=scored_outputs,
            best_property_value=best_property_value,
            num_outputs=plan.num_outputs,
            remaining_strategy_steps=(
                self.max_decision_rounds
                if remaining_strategy_steps is None
                else remaining_strategy_steps
            ),
            remaining_tool_steps=(
                self.max_agent_steps
                if remaining_tool_steps is None
                else remaining_tool_steps
            ),
            constraints=self._tool_selection_constraints(decision_packet),
        )

    def _excluded_tools(self, prompt: str) -> list[str]:
        return find_excluded_tools(prompt, self.tool_chest.available())

    def _tool_selection_constraints(self, decision_packet: dict) -> list[str]:
        constraints = []
        if decision_packet.get("selection_rule"):
            constraints.append(decision_packet["selection_rule"])
        if decision_packet.get("selection_reason"):
            constraints.append(
                f"Baseline selection reason: {decision_packet['selection_reason']}"
            )
        constraints.extend(decision_packet.get("tool_preference_notes") or [])

        request = decision_packet.get("request") or {}
        tool_constraint = decision_packet.get("tool_constraint") or {}
        if tool_constraint.get("tool"):
            if tool_constraint.get("mode") == "exclusive":
                constraints.append(
                    "User made an exclusive tool constraint: use only "
                    f"{tool_constraint['tool']}; do not switch tools."
                )
            else:
                constraints.append(
                    "User requested "
                    f"{tool_constraint['tool']} as the starting/preferred tool; "
                    "switching may still be allowed if the run fails and no "
                    "exclusive wording was used."
                )
        if request.get("mode") == "multi_property":
            constraints.append(
                "Request is simultaneous multi-property optimization; choose "
                "only tools that support that mode."
            )
        eligible_tools = decision_packet.get("eligible_tools") or []
        if len(eligible_tools) == 1:
            constraints.append(
                "Phase 1 currently lists one preferred/baseline tool "
                f"({eligible_tools[0]}), but this is imperfect evidence; "
                "other configured non-excluded tools may still be reasonable "
                "if the context supports them."
            )
        if decision_packet.get("threshold_comparison"):
            constraints.append(
                "Benchmark threshold comparison is available but may be "
                "incomplete; use it as evidence, not as a guarantee."
            )
        return constraints

    def _next_action_context(
        self,
        prompt: str,
        plan,
        status: str,
        result,
        iterations: int,
        execution_reason: str,
        optimization_attempts: list[dict],
        metadata: dict,
    ) -> dict:
        selected_attempt = next(
            (
                attempt
                for attempt in optimization_attempts
                if attempt.get("selected_output")
            ),
            optimization_attempts[-1] if optimization_attempts else {},
        )
        target_hits = selected_attempt.get("target_hits")
        scored_outputs = selected_attempt.get("scored_outputs")
        expected_outputs = (
            result.molecule_count * plan.num_outputs
            if result.molecule_count and plan.num_outputs
            else None
        )
        target_status = self._target_status(target_hits, scored_outputs)
        strategy_context = self._strategy_context(
            prompt=prompt,
            phase="post_run",
            plan=plan,
            decision_packet=metadata.get("decision_packet", {}),
            prior_attempts=optimization_attempts,
            target_status=target_status,
            target_hits=target_hits,
            scored_outputs=scored_outputs,
            remaining_strategy_steps=max(
                0,
                self.max_decision_rounds
                - len(metadata.get("decision_trace", [])),
            ),
            remaining_tool_steps=self.max_agent_steps,
        ).to_dict()
        strategy_context.update({
            "prompt": prompt,
            "status": status,
            "tool": result.tool,
            "available_tools": self.tool_chest.available(),
            "property_scoring_tools": self.tool_chest.property_scoring_tools(),
            "plan": plan.to_dict(),
            "iterations": iterations,
            "molecule_count": result.molecule_count,
            "row_count": result.row_count,
            "requested_outputs_per_input": plan.num_outputs,
            "expected_total_outputs": expected_outputs,
            "final_target_hits": target_hits,
            "final_scored_outputs": scored_outputs,
            "final_validated_outputs": result.row_count,
            "final_target_status": target_status,
            "target_success_criterion": (
                "met only when all scored optimized outputs satisfy the "
                "requested property target"
                if scored_outputs is not None
                else "not numerically scored against a property target"
            ),
            "current_tool_retryable": self._tool_supports_agent_retry(result.tool),
            "failed_input_ids": result.failed_input_ids or [],
            "input_warnings": result.input_warnings or [],
            "execution_reason": execution_reason,
            "execution_settings": metadata.get("execution_settings", {}),
            "decision_packet": metadata.get("decision_packet", {}),
            "benchmark_evidence": metadata.get("benchmark_evidence", {}),
            "benchmark_is_imperfect": True,
            "excluded_tools": self._excluded_tools(prompt),
            "tool_docs": self.tool_chest.documentation(),
            "remaining_strategy_steps": max(
                0,
                self.max_decision_rounds
                - len(metadata.get("decision_trace", [])),
            ),
            "remaining_tool_steps": self.max_agent_steps,
            "optimization_attempts": optimization_attempts,
            "outputs": metadata.get("outputs", {}),
        })
        return strategy_context

    def _tool_supports_agent_retry(self, tool_name: str) -> bool:
        if tool_name not in self.tool_chest.available():
            return False
        return bool(
            getattr(
                self.tool_chest.get(tool_name),
                "supports_agent_retry",
                False,
            )
        )

    def _target_status(self, target_hits, scored_outputs) -> str | None:
        return scored_target_status(target_hits, scored_outputs)

    def _log_scored_outputs(self, results: list, max_rows: int = 25) -> None:
        scored_results = [
            result
            for result in results
            if result.property
            and result.input_property_value is not None
            and result.optimized_property_value is not None
        ]
        if not scored_results:
            return

        shown = min(len(scored_results), max_rows)
        logger.info(
            "Scored optimized molecules: showing %s/%s.",
            shown,
            len(scored_results),
        )
        current_input = None
        for result in scored_results[:max_rows]:
            if result.input_id != current_input:
                current_input = result.input_id
                logger.info(
                    "Input %s: %s (%s=%s)",
                    result.input_id,
                    result.input_smiles,
                    result.property,
                    self._format_score(result.input_property_value),
                )
            logger.info(
                "  %s. %s (%s=%s) -> %s (%s=%s)",
                result.output_id,
                result.input_smiles,
                result.property,
                self._format_score(result.input_property_value),
                result.optimized_smiles,
                result.property,
                self._format_score(result.optimized_property_value),
            )

        if len(scored_results) > max_rows:
            logger.info(
                "Additional scored outputs omitted from stdout; full rows are "
                "in the output CSV."
            )

    @staticmethod
    def _format_score(value: float) -> str:
        return f"{value:.4g}"

    def _format_objective(
        self,
        objective,
        objectives: list | None = None,
        requested_properties: list[str] | None = None,
        objective_mode: str = "single_property",
    ) -> str:
        if objective is None:
            primary = "none"
        else:
            primary = format_objective(objective)
        if objective_mode == "multi_property" and objectives:
            requested = format_objectives(objectives)
            if objective is None:
                return f"multi_property {requested}; no scoring priority"
            return f"multi_property {requested}; scoring_priority={primary}"
        if objective_mode == "multi_property" and requested_properties:
            if objective is None:
                return (
                    f"multi_property {', '.join(requested_properties)}; "
                    "no scoring priority"
                )
            return (
                f"multi_property {', '.join(requested_properties)}; "
                f"scoring_priority={primary}"
            )
        return primary

    @staticmethod
    def _objectives(plan) -> list:
        objectives = getattr(plan, "objectives", None) or []
        if objectives:
            return list(objectives)
        objective = getattr(plan, "objective", None)
        return [objective] if objective is not None else []

    @staticmethod
    def _requested_properties(plan, decision_packet: dict) -> list[str]:
        request = decision_packet.get("request") or {}
        properties = request.get("properties") or getattr(
            plan,
            "requested_properties",
            None,
        )
        return list(dict.fromkeys(properties or []))

    def _build_execution_reason(
        self,
        plan_reason: str,
        plan_tool: str,
        final_tool: str,
        attempts: list[dict],
    ) -> str:
        """Explain the final executed tool, including adaptive fallback."""
        if len(attempts) <= 1:
            return plan_reason or "No reason returned."

        first_attempt = attempts[0]
        final_attempt = next(
            (
                attempt
                for attempt in attempts
                if attempt.get("selected_output")
            ),
            next(
                (
                    attempt
                    for attempt in attempts
                    if attempt.get("tool") == final_tool
                ),
                attempts[-1],
            ),
        )
        first_hits = first_attempt.get("target_hits", "n/a")
        first_scored = first_attempt.get("scored_outputs", "n/a")
        final_hits = final_attempt.get("target_hits", "n/a")
        final_scored = final_attempt.get("scored_outputs", "n/a")
        final_rows = final_attempt.get("row_count") or final_attempt.get("validated_outputs")
        if self.action_planner.attempt_satisfied(final_attempt):
            if final_tool == plan_tool:
                return (
                    f"{final_tool} met the target for "
                    f"{final_hits}/{final_scored} candidates."
                )
            return (
                f"{plan_tool} reached {first_hits}/{first_scored} candidates; "
                f"switched to {final_tool}, which reached {final_hits}/{final_scored}."
            )

        if final_attempt.get("target_hits") is None:
            if final_rows is not None:
                return (
                    f"Ran {len(attempts)} optimizer attempt(s); {final_tool} "
                    f"produced the selected validated output with {final_rows} "
                    "valid unique row(s). Property target scoring was not available."
                )
            return (
                f"Ran {len(attempts)} optimizer attempt(s); {final_tool} produced "
                "validated output, but property target scoring was not available."
            )

        return (
            f"Ran {len(attempts)} optimizer attempt(s); {final_tool} returned the "
            f"best partial result ({final_hits}/{final_scored} target hits)."
        )

    @staticmethod
    def _build_stopping_reason(
        status: str,
        result,
        attempts: list[dict],
    ) -> dict:
        selected = next(
            (attempt for attempt in attempts if attempt.get("selected_output")),
            attempts[-1] if attempts else {},
        )
        if status == "partial":
            failed = result.failed_input_ids or []
            if failed:
                return {
                    "reason": "partial_valid_outputs",
                    "message": (
                        f"Stopped after producing partial output; "
                        f"{len(failed)} input molecule(s) did not produce "
                        "valid optimized candidates."
                    ),
                }
            return {
                "reason": "partial_output",
                "message": "Stopped after producing partial optimized output.",
            }
        if status == "failed":
            failed = result.failed_input_ids or []
            if failed:
                return {
                    "reason": "no_valid_outputs",
                    "message": (
                        "Stopped because no valid optimized output rows were "
                        f"produced for {len(failed)} failed input molecule(s)."
                    ),
                }
            if selected.get("error"):
                return {
                    "reason": "optimizer_error",
                    "message": (
                        "Stopped because the selected optimizer attempt failed: "
                        f"{selected['error']}"
                    ),
                }
            return {
                "reason": "optimization_failed",
                "message": "Stopped because optimization produced no output rows.",
            }
        if selected.get("target_hits") is not None:
            target_hits = selected.get("target_hits")
            scored_outputs = selected.get("scored_outputs")
            if scored_outputs and target_hits >= scored_outputs:
                return {
                    "reason": "target_satisfied",
                    "message": (
                        "Stopped because all scored candidates met the requested "
                        "target criterion."
                    ),
                }
            return {
                "reason": "best_available_output",
                "message": (
                    "Stopped after selecting the best available optimizer output "
                    "from scored attempts."
                ),
            }
        return {
            "reason": "validated_outputs_written",
            "message": (
                "Stopped after writing validated optimized SMILES; property "
                "target scoring was unavailable."
            ),
        }

    def _format_optimization_attempts(self, attempts: list[dict]) -> str:
        lines = []
        for index, attempt in enumerate(attempts, start=1):
            settings = attempt.get("execution_settings") or {}
            budget = settings.get("max_oracle_calls_per_molecule")
            budget_text = f", oracle_budget={budget}/molecule" if budget else ""
            if attempt.get("error"):
                result = f"error={attempt['error']}"
            elif attempt.get("failed_input_ids"):
                failed = ",".join(attempt["failed_input_ids"])
                result = f"failed_inputs={failed}"
            elif attempt.get("target_hits") is None:
                result = "target scoring unavailable"
            else:
                result = (
                    f"target_hits={attempt['target_hits']}/"
                    f"{attempt.get('scored_outputs', 0)}"
                )
            lines.append(
                f"{index}. tool={attempt['tool']}{budget_text}, {result}"
                f"{' [selected]' if attempt.get('selected_output') else ''}"
            )
        return "\n".join(lines) or "(none)"

    def _write_iteration_input(self, output_csv: Path, next_input: Path) -> None:
        """Convert optimizer output rows into the next iteration's input CSV."""
        next_input.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open(newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            rows = [
                {
                    "id": f"{row['input_id']}_{row['output_id']}",
                    "smiles": row["optimized_smiles"],
                }
                for row in reader
            ]

        with next_input.open("w", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=["id", "smiles"])
            writer.writeheader()
            writer.writerows(rows)
