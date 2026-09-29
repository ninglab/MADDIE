import json
from pathlib import Path

from ..execution.adaptive_runner import TARGET_RANGE_TOLERANCES
from ..oracles.specs import format_objective, format_objectives
from ..status import (
    RESPONSE_TARGET_ALL_MET,
    TARGET_NOT_SCORED,
    response_target_status_label,
    scored_target_status,
)
from ..utils.logging import get_logger


logger = get_logger()


SUMMARY_SYSTEM_PROMPT = (
    "You are the communication interface for LeadOptimizer. "
    "Write a concise final response using only the provided JSON facts. "
    "State what was attempted, whether the run completed/partially "
    "completed/failed, what tool was used, whether the target was met "
    "when final_target_hits is available, and where the output CSV is. "
    "Use target_status to describe property-target outcome: all_met, partial, "
    "none_met, scored_no_target, validated_smiles_only, or failed. "
    "When requested_outputs_per_input and expected_total_outputs are present, "
    "describe requested candidates as per input molecule and do not confuse "
    "that with the total output row count. "
    "Do not treat valid-SMILES filtering as evidence that the requested "
    "property target was met. If score_summary is available but "
    "final_target_hits is not, say the property was scored but no numeric "
    "target/pass-fail criterion was available. If both score_summary and "
    "final_target_hits are unavailable, say the outputs were filtered to "
    "valid unique SMILES but the property target was not evaluated. "
    "For validated_smiles_only, do not say no target hits were achieved; say "
    "numeric target hits were not evaluated. "
    "When score_summary is available, include the optimized property, "
    "input/optimized value ranges, best optimized value, and mean change. "
    "If action_notes is non-empty, include a brief note describing the "
    "interpretation, cleanup, warnings, or repairs the agent applied. "
    "If target_feasibility_note is present, include it as a cautious practical "
    "note about the target under the current run/tool/settings; do not claim "
    "the target is globally impossible. "
    "If reasoning_limit_note is present, use only that note to describe why "
    "bounded reasoning stopped; say stopped at the named limit, not stopped "
    "early. Do not mention decision-round, tool-action, or budget limits that "
    "are not named in reasoning_limit_note. "
    "If runtime_note is present, include it briefly when it helps explain "
    "runtime, expensive follow-up actions, or user-facing expectations. "
    "If debug_output_csvs is non-empty, mention that these are intermediate "
    "debug artifacts from optimizer attempts and that output_csv is the final "
    "selected output. "
    "Use final_target_hits as the only authoritative target-hit count. "
    "Do not report target-hit counts from optimizer attempts. "
    "Do not call a none_met result partial, partially successful, or a best "
    "partial result; for none_met, say the selected/best available output did "
    "not meet the target. "
    "If partial_output_note is present, mention the partial output clearly. "
    "When selected_output_counts is present, use it for per-input output "
    "counts in the final selected output. Do not infer final per-input counts "
    "from warning text or intermediate debug attempts. "
    "If final_target_hits.target_status is partial, say the run completed but "
    "only partially met the target; do not call that fully successful. "
    "Mention tolerance only when final_target_hits.target_tolerance is present; "
    "otherwise do not use the word tolerance. "
    "For target-range objectives, describe score changes as changes toward or "
    "within a range, not as improvement unless the user explicitly requested "
    "increase or decrease. "
    "Do not invent molecules, scores, paths, tools, or reasons. "
    "Keep the response to one compact paragraph."
)


def build_response_summary(
    *,
    prompt: str,
    result,
    output_csv: Path,
    env: Path,
    iterations: int,
    status: str,
    stopping_reason: str | None = None,
    stopping_reason_message: str | None = None,
    objective=None,
    objectives: list | None = None,
    requested_properties: list[str] | None = None,
    objective_mode: str = "single_property",
    requested_outputs_per_input: int | None = None,
    execution_reason: str,
    optimization_attempts: list[dict],
    metadata: dict,
    reasoning_limits: dict | None = None,
    planner=None,
    response_llm=None,
    use_llm: bool = True,
) -> str:
    """Build the final user response, using an LLM when available."""
    deterministic_response = build_deterministic_response(
        result=result,
        output_csv=output_csv,
        env=env,
        iterations=iterations,
        objective=objective,
        objectives=objectives,
        requested_properties=requested_properties,
        objective_mode=objective_mode,
        execution_reason=execution_reason,
        attempts=optimization_attempts,
        reasoning_limits=reasoning_limits,
        runtime_info=metadata.get("runtime"),
    )
    metadata["response_summary_source"] = "deterministic"

    context = response_summary_context(
        prompt=prompt,
        result=result,
        output_csv=output_csv,
        iterations=iterations,
        status=status,
        stopping_reason=stopping_reason,
        stopping_reason_message=stopping_reason_message,
        objective=objective,
        objectives=objectives,
        requested_properties=requested_properties,
        objective_mode=objective_mode,
        requested_outputs_per_input=requested_outputs_per_input,
        execution_reason=execution_reason,
        optimization_attempts=optimization_attempts,
        reasoning_limits=reasoning_limits,
        runtime_info=metadata.get("runtime"),
        deterministic_response=deterministic_response,
    )
    metadata.update(response_metadata(context))

    llm = response_llm
    if llm is None and use_llm:
        llm = getattr(planner, "llm", None)
    if llm is None or not use_llm or status in {"failed", "planner_failed", "needs_input"}:
        return deterministic_response

    try:
        response = llm(
            SUMMARY_SYSTEM_PROMPT,
            json.dumps(context, indent=2, default=str),
        ).strip()
    except Exception as exc:
        metadata["response_summary_source"] = "deterministic_fallback"
        metadata["response_summary_error"] = str(exc)
        logger.warning(
            "LLM response summary failed; using deterministic summary: %s",
            exc,
        )
        return deterministic_response

    if not response:
        metadata["response_summary_source"] = "deterministic_fallback"
        metadata["response_summary_error"] = "empty LLM response"
        return deterministic_response

    metadata["response_summary_source"] = "llm"
    return response


def response_metadata(context: dict) -> dict:
    """Small machine-readable response facts for orchestrator callers."""
    final_hits = context.get("final_target_hits") or {}
    return {
        "target_status": context["target_status"],
        "stopping_reason": context["stopping_reason"],
        "stopping_reason_message": context["stopping_reason_message"],
        "validation_status": context["validation_status"],
        "property_scored": context["property_scored"],
        "property_target_evaluated": context["property_target_evaluated"],
        "target_hits": final_hits.get("target_hits"),
        "scored_output_count": context["scored_output_count"],
        "partial_output_note": context["partial_output_note"],
        "target_feasibility_note": context["target_feasibility_note"],
        "reasoning_limits": context["reasoning_limits"],
        "reasoning_limit_note": context["reasoning_limit_note"],
        "runtime": context["runtime"],
        "runtime_note": context["runtime_note"],
        "action_notes": context["action_notes"],
        "debug_output_csvs": context["debug_output_csvs"],
    }


def build_deterministic_response(
    *,
    result,
    output_csv: Path,
    env: Path,
    iterations: int,
    objective=None,
    objectives: list | None = None,
    requested_properties: list[str] | None = None,
    objective_mode: str = "single_property",
    execution_reason: str = "",
    attempts: list[dict] | None = None,
    reasoning_limits: dict | None = None,
    runtime_info: dict | None = None,
) -> str:
    """Build a simple fallback response without another LLM call."""
    attempts = attempts or []
    final_attempt = attempts[-1] if attempts else {}
    target_hits = final_attempt.get("target_hits")
    scored_outputs = final_attempt.get("scored_outputs")
    property_scored = any(
        row.property
        and row.input_property_value is not None
        and row.optimized_property_value is not None
        for row in result.results
    )

    if result.row_count == 0:
        response = (
            f"No optimized molecules were produced with {result.tool}. "
            f"No optimized rows were written to {output_csv.name}."
        )
    elif iterations > 1:
        response = (
            f"Lead optimization completed with {result.tool}. Wrote "
            f"{result.row_count} candidate row(s) for {result.molecule_count} "
            f"input molecule(s) to {output_csv.name} after {iterations} iterations."
        )
    else:
        response = (
            f"Lead optimization completed with {result.tool}. Wrote "
            f"{result.row_count} candidate row(s) for {result.molecule_count} "
            f"input molecule(s) to {output_csv.name}."
        )

    if target_hits is not None and scored_outputs is not None:
        response += (
            f" Target check: {target_hits}/{scored_outputs} scored candidate(s) "
            "met the requested criterion."
        )
    elif property_scored:
        response += (
            " Property scores were calculated, but no numeric pass/fail target "
            "was available."
        )
    elif result.row_count:
        response += (
            " Candidates were filtered to valid unique SMILES, but property "
            "scoring was not available for this output."
        )
    if objective_mode == "multi_property" and objectives:
        response += (
            f" Requested objectives: {format_objectives(objectives)}. "
            "The full multi-property instruction was sent to the optimizer."
        )
        if objective is not None:
            response += (
                f" The request marked {format_objective(objective)} as the "
                "scoring priority."
            )
    elif objective_mode == "multi_property" and requested_properties:
        response += (
            f" Requested properties: {', '.join(requested_properties)}. "
            "The full multi-property instruction was sent to the optimizer."
        )
    if execution_reason and execution_reason != "No reason returned.":
        response += f" Reason: {execution_reason}"

    final_hits = final_target_hits(attempts, objective)
    summary = score_summary(
        [
            row for row in result.results
            if row.property
            and row.input_property_value is not None
            and row.optimized_property_value is not None
        ],
        objective,
    )
    feasibility_note = target_feasibility_note(final_hits, summary, objective)
    if feasibility_note:
        response += f" {feasibility_note}"
    limit_note = reasoning_limit_note(reasoning_limits, final_hits)
    if limit_note:
        response += f" {limit_note}"
    run_time_note = runtime_note(runtime_info)
    if run_time_note:
        response += f" {run_time_note}"

    debug_outputs = debug_output_paths(attempts, output_csv)
    if debug_outputs:
        response += (
            " Intermediate attempt CSVs were saved in the debug folder; the "
            "main output CSV is the selected result."
        )

    notable_warnings = [
        warning
        for warning in result.input_warnings or []
        if warning.startswith(("Used '", "Renamed duplicate", "Skipped "))
    ]
    if notable_warnings:
        response += f" Input cleanup: {' '.join(notable_warnings)}"

    partial_warnings = [
        warning
        for warning in result.input_warnings or []
        if (
            "stopped after" in warning
            and "valid unique molecules" in warning
        )
        or " failed for " in warning
        or warning.startswith("Could not score")
    ]
    if partial_warnings:
        label = "Partial output" if result.row_count else "Failure"
        response += f" {label}: {' '.join(partial_warnings)}"
    return response


def response_summary_context(
    *,
    prompt: str,
    result,
    output_csv: Path,
    iterations: int,
    status: str,
    stopping_reason: str | None = None,
    stopping_reason_message: str | None = None,
    objective=None,
    objectives: list | None,
    requested_properties: list[str] | None,
    objective_mode: str,
    requested_outputs_per_input: int | None,
    execution_reason: str,
    optimization_attempts: list[dict],
    reasoning_limits: dict | None,
    runtime_info: dict | None,
    deterministic_response: str,
) -> dict:
    scored_results = [
        row for row in result.results
        if row.property
        and row.input_property_value is not None
        and row.optimized_property_value is not None
    ]
    final_hits = final_target_hits(optimization_attempts, objective)
    debug_outputs = debug_output_paths(optimization_attempts, output_csv)
    validation_status = (
        "valid_unique_smiles" if result.row_count else "no_valid_outputs"
    )
    property_scored = bool(scored_results)
    summary = score_summary(scored_results, objective)
    target_status_value = response_target_status(
        final_hits=final_hits,
        row_count=result.row_count,
        property_scored=property_scored,
        status=status,
    )
    examples = [
        {
            "input_id": row.input_id,
            "input_smiles": row.input_smiles,
            "optimized_smiles": row.optimized_smiles,
            "property": row.property,
            "input_property_value": row.input_property_value,
            "optimized_property_value": row.optimized_property_value,
        }
        for row in result.results[:5]
    ]
    return {
        "agent": "LeadOptimizer",
        "user_prompt": prompt,
        "status": status,
        "stopping_reason": stopping_reason,
        "stopping_reason_message": stopping_reason_message,
        "objective": objective.to_dict() if objective is not None else None,
        "objectives": [
            item.to_dict() for item in objectives or []
        ],
        "requested_properties": requested_properties or [],
        "objective_mode": objective_mode,
        "tool": result.tool,
        "iterations": iterations,
        "input_molecule_count": result.molecule_count,
        "requested_outputs_per_input": requested_outputs_per_input,
        "expected_total_outputs": expected_total_outputs(
            result.molecule_count,
            requested_outputs_per_input,
        ),
        "optimized_row_count": result.row_count,
        "selected_output_counts": selected_output_counts(
            result,
            requested_outputs_per_input,
        ),
        "output_csv": str(output_csv),
        "debug_output_csvs": debug_outputs,
        "execution_reason": execution_reason,
        "action_notes": action_notes(
            objective,
            result.input_warnings or [],
            objectives=objectives,
            requested_properties=requested_properties,
            objective_mode=objective_mode,
        ),
        "validation_note": validation_note(final_hits, result.row_count),
        "validation_status": validation_status,
        "property_scored": property_scored,
        "property_target_evaluated": final_hits is not None,
        "target_status": target_status_value,
        "final_target_hits": final_hits,
        "scored_output_count": len(scored_results),
        "score_summary": summary,
        "target_feasibility_note": target_feasibility_note(
            final_hits,
            summary,
            objective,
        ),
        "reasoning_limits": reasoning_limits or {},
        "reasoning_limit_note": reasoning_limit_note(
            reasoning_limits,
            final_hits,
        ),
        "runtime": runtime_info or {},
        "runtime_note": runtime_note(runtime_info),
        "failed_input_ids": result.failed_input_ids or [],
        "warnings": result.input_warnings or [],
        "partial_output_note": partial_output_note(
            result,
            requested_outputs_per_input,
        ),
        "skipped_input_rows": result.skipped_input_rows,
        "attempt_summary": attempt_summary(optimization_attempts),
        "example_outputs": examples,
        "deterministic_fallback_response": deterministic_response,
    }


def response_target_status(
    *,
    final_hits: dict | None,
    row_count: int,
    property_scored: bool,
    status: str,
) -> str:
    if final_hits is not None:
        return final_hits["target_status"]
    if property_scored:
        return "scored_no_target"
    if row_count:
        return "validated_smiles_only"
    if status == "unsupported":
        return "unsupported"
    return "failed"


def action_notes(
    objective,
    warnings: list[str],
    objectives: list | None = None,
    requested_properties: list[str] | None = None,
    objective_mode: str = "single_property",
) -> list[str]:
    notes = []
    if objective_mode == "multi_property" and requested_properties:
        notes.append(
            "Interpreted the request as simultaneous multi-property "
            f"optimization for {', '.join(requested_properties)}."
        )
    if objective is not None:
        if objective_mode == "multi_property":
            notes.append(
                f"Used {format_objective(objective)} as the scoring "
                "priority because the request explicitly marked it."
            )
        else:
            notes.append(
                f"Interpreted the request as {format_objective(objective)}."
            )
    notes.extend(
        warning for warning in warnings
        if not (
            "stopped after" in warning
            and "valid unique molecules" in warning
        )
    )
    return list(dict.fromkeys(note for note in notes if note))


def expected_total_outputs(
    molecule_count: int,
    requested_outputs_per_input: int | None,
) -> int | None:
    if molecule_count <= 0 or requested_outputs_per_input is None:
        return None
    return molecule_count * requested_outputs_per_input


def final_target_hits(optimization_attempts: list[dict], objective=None) -> dict | None:
    if not optimization_attempts:
        return None
    final_attempt = selected_attempt(optimization_attempts)
    target_hits = final_attempt.get("target_hits")
    scored_outputs = final_attempt.get("scored_outputs")
    if target_hits is None or scored_outputs is None:
        return None
    hits = {
        "target_hits": target_hits,
        "scored_outputs": scored_outputs,
        "target_status": target_status(target_hits, scored_outputs),
    }
    if objective is not None and objective.direction == "target_range":
        tolerance = TARGET_RANGE_TOLERANCES.get(objective.property, 0.0)
        if tolerance:
            hits["target_tolerance"] = tolerance
            hits["target_note"] = (
                f"Target-range hits allow +/-{tolerance:g} tolerance around "
                "the requested bounds."
            )
    return hits


def validation_note(final_hits: dict | None, row_count: int) -> str:
    if final_hits is None and row_count:
        return (
            "Output candidates were filtered to valid unique SMILES, but "
            "the requested property target was not evaluated for this output."
        )
    if final_hits is not None:
        return (
            "Output candidates were filtered to valid unique SMILES and "
            "then checked against the requested property target."
        )
    return "No optimized output candidates were produced."


def target_feasibility_note(
    final_hits: dict | None,
    summary: dict | None,
    objective=None,
) -> str | None:
    """Return a cautious note when observed scores miss a numeric target."""
    if final_hits is None or summary is None or objective is None:
        return None
    if final_hits.get("target_status") == RESPONSE_TARGET_ALL_MET:
        return None
    if not objective.property or not objective.direction:
        return None

    best = summary.get("best_optimized_value")
    if best is None:
        return None
    prop = summary.get("property") or objective.property
    hits = final_hits.get("target_hits")
    scored = final_hits.get("scored_outputs")
    hit_note = ""
    if hits is not None and scored is not None:
        hit_note = f" ({hits}/{scored} target hits)"

    if objective.direction == "increase" and objective.threshold is not None:
        if best < objective.threshold:
            return (
                f"Target feasibility note: under the current tool/settings, "
                f"the best observed {prop} was {best}, below the requested "
                f"cutoff of {objective.threshold}{hit_note}; consider relaxing "
                "the cutoff, allowing another tool, or increasing budget in "
                "the next request."
            )
    elif objective.direction == "decrease" and objective.threshold is not None:
        if best > objective.threshold:
            return (
                f"Target feasibility note: under the current tool/settings, "
                f"the best observed {prop} was {best}, above the requested "
                f"cutoff of {objective.threshold}{hit_note}; consider relaxing "
                "the cutoff, allowing another tool, or increasing budget in "
                "the next request."
            )
    elif (
        objective.direction == "target_range"
        and objective.minimum is not None
        and objective.maximum is not None
        and not objective.minimum <= best <= objective.maximum
    ):
        return (
            f"Target feasibility note: under the current tool/settings, the "
            f"best observed {prop} was {best}, outside the requested "
            f"{objective.minimum}-{objective.maximum} range{hit_note}; "
            "consider relaxing the range, allowing another tool, or increasing "
            "budget in the next request."
        )
    return None


def reasoning_limit_note(
    reasoning_limits: dict | None,
    final_hits: dict | None,
) -> str | None:
    if not reasoning_limits:
        return None
    if (final_hits or {}).get("target_status") == RESPONSE_TARGET_ALL_MET:
        return None

    stopped_by = reasoning_limits.get("stopped_by")
    if stopped_by not in {"decision_round_limit", "tool_action_limit"}:
        return None

    rounds = reasoning_limits.get("decision_rounds_used")
    actions = reasoning_limits.get("tool_actions_executed")
    if stopped_by == "decision_round_limit":
        reason = "decision-round limit"
    else:
        reason = "tool-action limit"
    return (
        f"Bounded reasoning note: automatic follow-up stopped at the configured {reason} "
        f"after {rounds} decision round(s) and {actions} executed tool "
        "action(s); increase the tool-action budget, request numeric scoring, "
        "or allow another compatible tool in the next request if more "
        "exploration is desired."
    )


def runtime_note(runtime_info: dict | None) -> str | None:
    if not runtime_info:
        return None

    elapsed = runtime_info.get("elapsed_seconds")
    if elapsed is None:
        return None

    parts = [f"Runtime note: this run took {elapsed} seconds"]
    budget = runtime_info.get("max_oracle_calls_per_molecule")
    total_budget = runtime_info.get("estimated_total_oracle_call_budget")
    if budget is not None:
        parts.append(
            f"and used MolOpt oracle budgets up to {budget} calls per input molecule"
        )
    if total_budget is not None:
        parts.append(
            f"(upper-bound total oracle-call budget {total_budget})"
        )
    return " ".join(parts) + "."


def selected_output_counts(result, requested_outputs_per_input: int | None) -> list[dict]:
    counts = {}
    for row in result.results:
        counts[row.input_id] = counts.get(row.input_id, 0) + 1
    return [
        {
            "input_id": input_id,
            "accepted_outputs": count,
            "requested_outputs": requested_outputs_per_input,
        }
        for input_id, count in sorted(counts.items())
    ]


def partial_output_note(result, requested_outputs_per_input: int | None = None) -> str | None:
    counts = selected_output_counts(result, requested_outputs_per_input)
    short_counts = [
        count
        for count in counts
        if count["requested_outputs"] is not None
        and count["accepted_outputs"] < count["requested_outputs"]
    ]
    if short_counts:
        details = ", ".join(
            f"{item['input_id']} produced {item['accepted_outputs']}/"
            f"{item['requested_outputs']} requested valid unique output(s)"
            for item in short_counts
        )
        return f"Selected output is partial: {details}."

    notes = [
        warning
        for warning in result.input_warnings or []
        if (
            "stopped after" in warning
            and "valid unique molecules" in warning
        )
        or " failed for " in warning
        or warning.startswith("Could not score")
    ]
    if notes:
        return " ".join(notes)
    if result.partial and result.row_count:
        return "The run produced partial output; see warnings for details."
    return None


def attempt_summary(optimization_attempts: list[dict]) -> str:
    if not optimization_attempts:
        return ""
    if len(optimization_attempts) == 1:
        return f"One optimizer attempt ran with {optimization_attempts[0].get('tool')}."
    final_tool = selected_attempt(optimization_attempts).get("tool", "the final tool")
    return (
        f"{len(optimization_attempts)} optimizer attempt(s) ran; "
        f"{final_tool} produced the selected output."
    )


def selected_attempt(optimization_attempts: list[dict]) -> dict:
    """Return the attempt that produced the final selected output."""
    return next(
        (
            attempt
            for attempt in optimization_attempts
            if attempt.get("selected_output")
        ),
        optimization_attempts[-1],
    )


def debug_output_paths(
    optimization_attempts: list[dict],
    final_output_csv: Path,
) -> list[str]:
    """Return non-final attempt CSVs for orchestrator/debug traceability."""
    final_path = Path(final_output_csv)
    paths = []
    for attempt in optimization_attempts:
        output_csv = attempt.get("output_csv")
        if output_csv and Path(output_csv) != final_path:
            paths.append(str(output_csv))
    return list(dict.fromkeys(paths))


def target_status(target_hits: int, scored_outputs: int) -> str:
    return response_target_status_label(
        scored_target_status(target_hits, scored_outputs)
    ) or TARGET_NOT_SCORED


def score_summary(scored_results: list, objective=None) -> dict | None:
    if not scored_results:
        return None

    property_name = scored_results[0].property
    input_values = [row.input_property_value for row in scored_results]
    optimized_values = [row.optimized_property_value for row in scored_results]
    deltas = [
        optimized - input_value
        for input_value, optimized in zip(input_values, optimized_values)
    ]
    mean_delta = sum(deltas) / len(deltas)
    mean_change_value, mean_change_label = mean_change_summary(
        mean_delta,
        objective,
    )
    return {
        "property": property_name,
        "count": len(scored_results),
        "input": summarize_values(input_values),
        "optimized": summarize_values(optimized_values),
        "best_optimized_value": best_value(optimized_values, objective),
        "mean_change": mean_change_value,
        "mean_change_label": mean_change_label,
    }


def summarize_values(values: list[float]) -> dict:
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(sum(values) / len(values), 4),
    }


def best_value(values: list[float], objective=None) -> float:
    if objective is not None and objective.direction == "decrease":
        return round(min(values), 4)
    if objective is not None and objective.direction == "target_range":
        midpoint = (objective.minimum + objective.maximum) / 2
        return round(min(values, key=lambda value: abs(value - midpoint)), 4)
    return round(max(values), 4)


def mean_change_summary(mean_delta: float, objective=None) -> tuple[float, str]:
    if objective is not None and objective.direction == "target_range":
        return round(mean_delta, 4), "mean change"
    if mean_delta >= 0:
        return round(mean_delta, 4), "mean increase"
    return round(abs(mean_delta), 4), "mean decrease"
