from dataclasses import asdict, dataclass


@dataclass
class OptimizationObjective:
    """Structured optimization goal parsed from the user's request."""

    property: str
    direction: str
    threshold: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if self.direction not in {"increase", "decrease", "target_range"}:
            raise ValueError(
                "Objective direction must be increase, decrease, or target_range."
            )
        if self.direction == "target_range":
            if self.minimum is None or self.maximum is None:
                raise ValueError("Target-range objectives require minimum and maximum.")
            if self.minimum > self.maximum:
                raise ValueError("Objective minimum cannot be greater than maximum.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationRequest:
    input_id: str
    smiles: str
    adjustment: str
    num_outputs: int

    thresholds: dict[str, float] | None = None
    stable: list[str] | None = None
    attempt: int = 1

    def to_gellmo_text(self) -> str:
        return f"{self.adjustment} <SMILES> {self.smiles} </SMILES>"


@dataclass
class OptimizationResult:
    input_id: str
    input_smiles: str
    optimized_smiles: str
    output_id: int
    adjustment: str
    tool: str
    property: str | None = None
    input_property_value: float | None = None
    optimized_property_value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationBatchResult:
    input_csv: str
    output_csv: str
    molecule_count: int
    row_count: int
    tool: str
    results: list[OptimizationResult]
    input_warnings: list[str] | None = None
    skipped_input_rows: int = 0
    failed_input_ids: list[str] | None = None
    partial: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["results"] = [result.to_dict() for result in self.results]
        return data
