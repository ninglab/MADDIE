"""Load compact measured evidence for benchmark-aware tool selection."""

import csv
import json
from pathlib import Path
from statistics import mean

from .schemas import OptimizationObjective


ADMET_ORACLES = {"herg", "dili", "clintox", "mutagenicity", "carcinogens"}


def load_benchmark_evidence(
    knowledge_path: str | Path,
    objective: OptimizationObjective | None,
    legal_tools: list[str],
) -> dict:
    if objective is None:
        return {}

    path = Path(knowledge_path)
    knowledge = json.loads(path.read_text())
    profile = next(
        (
            item
            for item in knowledge["objective_profiles"]
            if objective.property in item["applies_to"]
        ),
        None,
    )
    if profile is None:
        return {}

    legal_algorithms = {
        tool.removeprefix("molopt_")
        for tool in legal_tools
        if tool.startswith("molopt_")
    }
    run_rows = []
    threshold_rows = []
    evidence = profile.get("evidence", [{}])[0]
    for source in evidence.get("source_tables", []):
        table_path = path.parent.parent / source
        if not table_path.exists():
            continue
        with table_path.open(newline="") as handle:
            table_rows = list(csv.DictReader(handle))
        columns = table_rows[0] if table_rows else {}
        if {"oracle", "algorithm", "seed"}.issubset(columns):
            run_rows = [
                row
                for row in table_rows
                if row["oracle"] == objective.property
                and row["algorithm"] in legal_algorithms
            ]
        elif {"oracle", "algorithm", "threshold", "call_to_threshold"}.issubset(columns):
            threshold_rows = [
                row
                for row in table_rows
                if row["oracle"] == objective.property
                and row["algorithm"] in legal_algorithms
                and row.get("top_k") == "10"
            ]

    if not threshold_rows and objective.property in ADMET_ORACLES:
        threshold_path = path.parent.parent / "tables/small_admet/small_admet_top10_thresholds.csv"
        if threshold_path.exists():
            with threshold_path.open(newline="") as handle:
                threshold_rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row["oracle"] == objective.property
                    and row["algorithm"] in legal_algorithms
                    and row.get("top_k") == "10"
                ]

    run_calls = {}
    final_top1 = {}
    final_top10 = {}
    for row in run_rows:
        algorithm = row["algorithm"]
        run_calls.setdefault(algorithm, []).append(float(row["max_call"]))
        if row.get("final_top1"):
            final_top1.setdefault(algorithm, []).append(float(row["final_top1"]))
        if row.get("final_top10"):
            final_top10.setdefault(algorithm, []).append(float(row["final_top10"]))

    runtime_by_algorithm = {}
    runtime_path = path.parent.parent / "tables/molopt_generation_probe.csv"
    if runtime_path.exists():
        with runtime_path.open(newline="") as handle:
            runtime_by_algorithm = {
                row["algorithm"]: float(row["seconds_per_molecule"])
                for row in csv.DictReader(handle)
            }

    oracle_seconds_per_call = None
    oracle_runtime_path = path.parent.parent / "tables/oracle_wall_time_estimates.csv"
    if oracle_runtime_path.exists():
        with oracle_runtime_path.open(newline="") as handle:
            runtime_row = next(
                (
                    row
                    for row in csv.DictReader(handle)
                    if row["oracle"] == objective.property
                ),
                None,
            )
        if runtime_row:
            oracle_seconds_per_call = float(runtime_row["estimated_seconds_per_call"])
    if oracle_seconds_per_call is None and objective.property in ADMET_ORACLES:
        admet_runtime_path = (
            path.parent.parent
            / "tables/admet_replicates/runtime_admet_replicates_by_oracle.csv"
        )
        if admet_runtime_path.exists():
            with admet_runtime_path.open(newline="") as handle:
                runtime_row = next(
                    (
                        row
                        for row in csv.DictReader(handle)
                        if row["oracle"] == objective.property
                    ),
                    None,
                )
            if runtime_row:
                oracle_seconds_per_call = float(runtime_row["seconds_per_call"])

    grouped_thresholds = {}
    for row in threshold_rows:
        grouped_thresholds.setdefault(float(row["threshold"]), []).append(row)

    comparisons = []
    algorithms_with_evidence = set()
    for threshold, rows_at_threshold in sorted(grouped_thresholds.items()):
        results = []
        for row in rows_at_threshold:
            algorithm = row["algorithm"]
            algorithms_with_evidence.add(algorithm)
            reached = row["reached"].lower() == "true"
            calls = int(row["call_to_threshold"]) if row["call_to_threshold"] else None
            result = {
                "tool": f"molopt_{algorithm}",
                "reached": reached,
                "oracle_calls": calls,
            }
            if calls is not None and oracle_seconds_per_call is not None:
                result["estimated_oracle_seconds"] = round(
                    calls * oracle_seconds_per_call,
                    1,
                )
            if algorithm in run_calls:
                result["mean_run_oracle_calls"] = round(mean(run_calls[algorithm]), 1)
            if algorithm in final_top1:
                result["mean_final_top1"] = round(mean(final_top1[algorithm]), 4)
            if algorithm in final_top10:
                result["mean_final_top10"] = round(mean(final_top10[algorithm]), 4)
            if algorithm in runtime_by_algorithm:
                result["search_overhead_seconds_per_molecule"] = runtime_by_algorithm[
                    algorithm
                ]
            results.append(result)
        results.sort(
            key=lambda item: (
                not item["reached"],
                item["oracle_calls"] if item["oracle_calls"] is not None else float("inf"),
            )
        )
        comparisons.append({"score_threshold": threshold, "results": results})

    comparable_thresholds = [
        comparison["score_threshold"]
        for comparison in comparisons
        if sum(result["reached"] for result in comparison["results"]) >= 2
    ]
    default_comparison_threshold = (
        max(comparable_thresholds) if comparable_thresholds else None
    )
    requested_threshold = objective.threshold
    requested_benchmark_threshold = benchmark_threshold(objective)
    comparison_threshold = default_comparison_threshold
    threshold_match = "default"
    if requested_benchmark_threshold is not None:
        satisfying_thresholds = [
            comparison["score_threshold"]
            for comparison in comparisons
            if comparison["score_threshold"] >= requested_benchmark_threshold
        ]
        if satisfying_thresholds:
            comparison_threshold = min(satisfying_thresholds)
            threshold_match = (
                "exact"
                if comparison_threshold == requested_benchmark_threshold
                else "next_tested_at_or_above"
            )
        else:
            comparison_threshold = None
            threshold_match = "no_tested_threshold_at_or_above"
    routing_comparison = None
    if comparison_threshold is not None and comparisons:
        routing_comparison = min(
            comparisons,
            key=lambda item: abs(item["score_threshold"] - comparison_threshold),
        )

    other_tool_measurements = []
    comparison_path = path.parent.parent / "tables/tool_runtime_comparison.csv"
    if "gellmo_c" in legal_tools and comparison_path.exists():
        with comparison_path.open(newline="") as handle:
            other_tool_measurements = [
                {
                    "tool": "gellmo_c",
                    "operation": row["operation"],
                    "unit": row["unit"],
                    "wall_seconds": float(row["wall_seconds"]),
                }
                for row in csv.DictReader(handle)
                if row["tool"] == "GeLLMO-C"
            ]

    return {
        "property": objective.property,
        "score_meaning": profile["molopt_score"],
        "requested_threshold": objective.threshold,
        "requested_benchmark_threshold": requested_benchmark_threshold,
        "threshold_match": threshold_match,
        "routing_priority": (
            "Compare oracle calls and estimated time needed to reach useful "
            "top-10 score thresholds. Final score differences are secondary."
        ),
        "molopt_property_evidence_strength": evidence.get("strength"),
        "benchmark_call_budget_per_run": evidence.get("calls_per_run"),
        "estimated_oracle_seconds_per_call": oracle_seconds_per_call,
        "comparison_threshold_when_unspecified": default_comparison_threshold,
        "available_tested_thresholds": [
            comparison["score_threshold"] for comparison in comparisons
        ],
        "routing_threshold_comparison": routing_comparison,
        "tools_without_threshold_evidence": sorted(
            set(legal_tools)
            - {f"molopt_{algorithm}" for algorithm in algorithms_with_evidence}
        ),
        "other_tool_runtime_measurements": other_tool_measurements,
    }


def benchmark_threshold(objective: OptimizationObjective) -> float | None:
    """Map raw user-facing thresholds to benchmark score thresholds."""
    if objective.threshold is None:
        return None
    if objective.property == "sascore" and objective.direction == "decrease":
        return 10.0 - objective.threshold
    if objective.property in ADMET_ORACLES and objective.direction == "decrease":
        return 1.0 - objective.threshold
    return objective.threshold
