from dataclasses import asdict, dataclass


@dataclass
class StrategyContext:
    prompt: str
    phase: str
    input_file: str | None
    objective: dict | None
    adjustment: str
    available_tools: list[str]
    excluded_tools: list[str]
    tool_docs: str
    deterministic_tool: str
    decision_packet: dict
    benchmark_evidence: dict
    benchmark_is_imperfect: bool
    prior_attempts: list[dict]
    target_status: str | None
    target_hits: int | None
    scored_outputs: int | None
    best_property_value: float | None
    num_outputs: int
    remaining_strategy_steps: int
    remaining_tool_steps: int
    constraints: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyAction:
    action_type: str
    tool: str | None
    adjustment: str
    num_outputs: int
    reason: str
    confidence: float | None
    source: str
    alternatives: list[str]
    warnings: list[str]
    raw: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolSelectionContext:
    prompt: str
    objective: dict | None
    deterministic_tool: str
    eligible_tools: list[str]
    available_tools: list[str]
    decision_packet: dict
    benchmark_evidence: dict
    tool_docs: str
    num_outputs: int
    constraints: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolSelectionDecision:
    tool: str
    confidence: str | None
    rationale: str
    alternatives: list[str]
    warnings: list[str]
    source: str
    raw: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
