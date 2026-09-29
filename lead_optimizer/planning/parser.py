import json
from json import JSONDecodeError

from ..errors import LeadOptimizerRequestError
from ..oracles.specs import (
    parse_optimization_objective,
    prompt_objective_fields,
)
from ..schemas import OptimizationObjective
from .schemas import OptimizationPlan


def parse_json_plan(raw_text: str) -> OptimizationPlan:
    """Convert LLM JSON text into the internal OptimizationPlan object."""
    try:
        data = json.loads(extract_json_object(raw_text))
    except (JSONDecodeError, ValueError) as exc:
        raise planner_error("planner_malformed_json", exc, raw_text) from exc
    input_file = data.get("input_file")

    try:
        objectives = parse_objective_list(data)
        objective = first_objective(
            objectives,
            parse_optimization_objective(data["adjustment"]),
        )
        return OptimizationPlan(
            task_type=data["task_type"],
            input_type="csv",
            input_file=input_file,
            adjustment=data["adjustment"],
            tool=data["tool"],
            num_outputs=int(data["num_outputs"]),
            objective=objective,
            objectives=objectives or ([objective] if objective is not None else None),
            reason=data.get("reason", ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise planner_error("planner_invalid_schema", exc, raw_text) from exc


def extract_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in planner response.")
    return raw_text[start:end + 1]


def compact_excerpt(text: str, limit: int = 600) -> str:
    """Keep metadata useful without dumping a full planner response."""
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit - 3] + "..."


def planner_error(category: str, exc: Exception, raw_text: str) -> LeadOptimizerRequestError:
    if category == "planner_malformed_json":
        reason = "the planner returned invalid JSON"
    else:
        reason = "the planner returned an invalid plan schema"
    return LeadOptimizerRequestError(
        "I could not plan the lead optimization request because "
        f"{reason}. No optimization was run and no output CSV was written.",
        status="planner_failed",
        metadata={
            "error_category": category,
            "planner_error": str(exc),
            "planner_response_excerpt": compact_excerpt(raw_text),
            "outputs": {"csv": None, "debug_csvs": [], "log": None},
        },
    )


def parse_objective_dict(data) -> OptimizationObjective | None:
    if not isinstance(data, dict):
        return None
    property_name = data.get("property")
    direction = data.get("direction")
    if property_name is None or direction is None:
        return None
    if direction == "target_range":
        minimum = data.get("minimum")
        maximum = data.get("maximum")
        if minimum is None or maximum is None:
            return None
        return OptimizationObjective(
            property=property_name,
            direction=direction,
            minimum=float(minimum),
            maximum=float(maximum),
        )
    return OptimizationObjective(
        property=property_name,
        direction=direction,
        threshold=optional_float(data.get("threshold")),
    )


def parse_objective_list(data: dict) -> list[OptimizationObjective]:
    raw_objectives = data.get("objectives", data.get("objective"))
    if raw_objectives is None:
        return []
    if isinstance(raw_objectives, dict):
        objective = parse_objective_dict(raw_objectives)
        return [objective] if objective is not None else []
    if not isinstance(raw_objectives, list):
        raise ValueError("objective must be an object or a list of objects.")

    objectives = []
    for index, raw_objective in enumerate(raw_objectives, start=1):
        objective = parse_objective_dict(raw_objective)
        if objective is None:
            raise ValueError(f"objective item {index} is missing property or direction.")
        objectives.append(objective)
    return objectives


def first_objective(
    objectives: list[OptimizationObjective],
    fallback: OptimizationObjective | None = None,
) -> OptimizationObjective | None:
    return objectives[0] if objectives else fallback


def objective_kwargs_from_prompt(prompt: str) -> dict:
    fields = prompt_objective_fields(prompt)
    return {
        "objective": fields["objective"],
        "objectives": fields["objectives"] or None,
        "requested_properties": fields["requested_properties"],
        "objective_mode": fields["objective_mode"],
    }


def apply_prompt_objectives(plan: OptimizationPlan, prompt: str) -> OptimizationPlan:
    fields = prompt_objective_fields(prompt)
    if fields["objectives"]:
        plan.objectives = fields["objectives"]
        plan.objective = fields["objective"]
    elif not plan.objectives and plan.objective is not None:
        plan.objectives = [plan.objective]
    elif plan.objectives and len(plan.objectives) > 1:
        plan.objective = None
    plan.requested_properties = fields["requested_properties"]
    plan.objective_mode = fields["objective_mode"]
    return plan


def optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)

