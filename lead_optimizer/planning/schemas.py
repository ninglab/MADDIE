from dataclasses import asdict, dataclass

from ..schemas import OptimizationObjective


@dataclass
class OptimizationPlan:
    task_type: str
    input_type: str
    input_file: str | None
    adjustment: str
    tool: str
    num_outputs: int
    objective: OptimizationObjective | None = None
    objectives: list[OptimizationObjective] | None = None
    reason: str = ""
    requested_properties: list[str] | None = None
    objective_mode: str = "single_property"

    def to_dict(self) -> dict:
        return asdict(self)

