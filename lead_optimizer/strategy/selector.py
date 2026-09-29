import json
import re
from dataclasses import replace

from .schemas import (
    StrategyAction,
    StrategyContext,
    ToolSelectionContext,
    ToolSelectionDecision,
)


TOOL_SELECTION_SYSTEM_PROMPT = """You advise the LeadOptimizer tool selector.
Use only the structured context. Return one JSON object with:
tool: one tool name from available_tools
confidence: low, medium, or high
rationale: concise explanation grounded in the context
alternatives: ordered list of other reasonable eligible tools
warnings: list of caveats, uncertainties, or user-facing assumptions

Do not invent benchmark evidence. Do not recommend tools outside available_tools
or excluded_tools. Treat decision_packet and benchmark_evidence as useful but
imperfect evidence, not ground truth. The deterministic_tool is the safe
baseline; override it only when the broader context supports a better choice.
"""


STRATEGY_SYSTEM_PROMPT = """You advise the LeadOptimizer strategy layer.
Use only the structured context. Return one JSON object with:
action_type: optimize_csv, retry_with_revised_instruction, switch_tool, stop, or ask_clarification
tool: tool name for optimize_csv/switch_tool, otherwise null
adjustment: concise optimization instruction to execute
num_outputs: requested outputs per input molecule
reason: concise explanation grounded in the context
confidence: number from 0.0 to 1.0
alternatives: optional ordered list of other reasonable tools
warnings: optional caveats or uncertainty

Do not execute tools. Do not invent benchmark evidence. Tools must come from
available_tools and must not come from excluded_tools. Treat decision_packet and
benchmark_evidence as useful but imperfect evidence, not ground truth.
"""


STRATEGY_ACTIONS = {
    "optimize_csv",
    "retry_with_revised_instruction",
    "switch_tool",
    "stop",
    "ask_clarification",
}


def legal_tools(context) -> list[str]:
    """Return configured tools that are not excluded by the user."""
    excluded = set(getattr(context, "excluded_tools", []) or [])
    return [
        tool for tool in getattr(context, "available_tools", []) or []
        if tool not in excluded
    ]


def fallback_tool(context, allowed_tools: list[str]) -> str | None:
    if context.deterministic_tool in allowed_tools:
        return context.deterministic_tool
    return allowed_tools[0] if allowed_tools else None


class StrategyPolicy:
    """Validate LLM strategy actions before the agent can execute them."""

    def validate(
        self,
        action: StrategyAction,
        context: StrategyContext,
    ) -> StrategyAction:
        action_type = self._normalize_action_type(action.action_type)
        warnings = list(action.warnings)

        if action_type not in STRATEGY_ACTIONS:
            warnings.append(f"Unsupported action_type '{action.action_type}'.")
            action_type = "optimize_csv" if context.phase == "pre_run" else "stop"

        allowed_tools = legal_tools(context)
        if not allowed_tools:
            return replace(
                action,
                action_type="ask_clarification",
                tool=None,
                confidence=0.0,
                reason="No configured tool remains after user constraints.",
                warnings=[
                    *warnings,
                    "No non-excluded tool was available.",
                ],
            )

        if context.phase == "pre_run" and action_type in {"stop", "ask_clarification"}:
            warnings.append(
                f"Pre-run action '{action_type}' was replaced with optimize_csv."
            )
            action_type = "optimize_csv"

        if action_type in {"optimize_csv", "switch_tool"}:
            tool = action.tool
            if tool not in allowed_tools:
                fallback = fallback_tool(context, allowed_tools)
                warnings.append(
                    f"Rejected ineligible tool recommendation: {tool}"
                )
                tool = fallback
        elif action_type == "retry_with_revised_instruction":
            tool = action.tool or context.deterministic_tool
            if tool not in allowed_tools:
                tool = fallback_tool(context, allowed_tools)
        else:
            tool = None

        num_outputs = action.num_outputs
        if not isinstance(num_outputs, int):
            try:
                num_outputs = int(num_outputs)
            except (TypeError, ValueError):
                num_outputs = context.num_outputs
        if num_outputs < 1:
            warnings.append("num_outputs was below 1; using context value.")
            num_outputs = context.num_outputs
        if num_outputs > 50:
            warnings.append("num_outputs was above 50; capped at 50.")
            num_outputs = 50

        adjustment = (action.adjustment or context.adjustment or "").strip()
        if action_type in {"optimize_csv", "retry_with_revised_instruction", "switch_tool"}:
            if not adjustment:
                adjustment = context.prompt.strip()
                warnings.append("Empty adjustment was replaced with the user prompt.")

        if action_type in {"retry_with_revised_instruction", "switch_tool"}:
            if context.remaining_tool_steps <= 0:
                warnings.append("No tool-execution budget remains.")
                action_type = "stop"
                tool = None
        alternatives = [
            candidate for candidate in action.alternatives
            if candidate in allowed_tools and candidate != tool
        ]

        return replace(
            action,
            action_type=action_type,
            tool=tool,
            adjustment=adjustment,
            num_outputs=num_outputs,
            alternatives=alternatives,
            warnings=warnings,
        )

    def _normalize_action_type(self, action_type: str) -> str:
        action_type = str(action_type or "").strip().lower().replace("-", "_")
        aliases = {
            "retry_same_tool": "retry_with_revised_instruction",
            "run_tool": "optimize_csv",
            "optimize": "optimize_csv",
        }
        return aliases.get(action_type, action_type)


class StrategyAdvisor:
    """Optional LLM-backed strategy recommender with deterministic validation."""

    def __init__(self, llm=None, enabled: bool = False, policy=None):
        self.llm = llm
        self.enabled = enabled
        self.policy = policy or StrategyPolicy()

    def recommend(self, context: StrategyContext) -> StrategyAction:
        if not self.enabled or self.llm is None:
            return self.policy.validate(
                StrategyAction(
                    action_type="optimize_csv" if context.phase == "pre_run" else "stop",
                    tool=context.deterministic_tool,
                    adjustment=context.adjustment,
                    num_outputs=context.num_outputs,
                    reason="LLM strategy advisor is disabled.",
                    confidence=None,
                    alternatives=[],
                    warnings=[],
                    source="disabled",
                ),
                context,
            )

        try:
            raw = self.llm(
                STRATEGY_SYSTEM_PROMPT,
                json.dumps(context.to_dict(), indent=2),
            )
            action = self._normalize(self._parse_json(raw), raw)
        except Exception as exc:
            action = StrategyAction(
                action_type="optimize_csv" if context.phase == "pre_run" else "stop",
                tool=context.deterministic_tool,
                adjustment=context.adjustment,
                num_outputs=context.num_outputs,
                reason="Strategy advisor failed; using deterministic fallback.",
                confidence=0.0,
                alternatives=[],
                warnings=["Fell back to deterministic strategy."],
                source="fallback",
                error=str(exc),
            )

        return self.policy.validate(action, context)

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize(self, data: dict, raw: str) -> StrategyAction:
        confidence = data.get("confidence")
        if isinstance(confidence, str):
            confidence = {
                "low": 0.25,
                "medium": 0.5,
                "high": 0.9,
            }.get(confidence.lower(), confidence)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        alternatives = data.get("alternatives") or []
        if not isinstance(alternatives, list):
            alternatives = []
        warnings = data.get("warnings") or []
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        action_type = data.get("action_type") or data.get("action")
        if not action_type and data.get("tool"):
            action_type = "optimize_csv"

        return StrategyAction(
            action_type=str(action_type or ""),
            tool=data.get("tool"),
            adjustment=str(data.get("adjustment") or ""),
            num_outputs=data.get("num_outputs"),
            reason=str(data.get("reason") or data.get("rationale") or "").strip(),
            confidence=confidence,
            alternatives=[str(item) for item in alternatives],
            warnings=[str(item) for item in warnings],
            source="llm",
            raw=raw,
        )


class ToolSelectionPolicy:
    """Validate LLM tool choices against deterministic legal options."""

    def validate(
        self,
        decision: ToolSelectionDecision,
        context: ToolSelectionContext,
    ) -> ToolSelectionDecision:
        allowed = legal_tools(context)
        if not allowed:
            return replace(
                decision,
                tool=context.deterministic_tool,
                confidence="low",
                warnings=[
                    *decision.warnings,
                    "No eligible tool list was available; kept deterministic tool.",
                ],
            )

        if decision.tool in allowed:
            return decision

        fallback = fallback_tool(context, allowed)
        return replace(
            decision,
            tool=fallback,
            confidence="low",
            rationale=(
                f"Rejected recommended tool '{decision.tool}' because it was "
                f"not eligible; using '{fallback}' instead."
            ),
            warnings=[
                *decision.warnings,
                f"Rejected ineligible tool recommendation: {decision.tool}",
            ],
        )


class ToolSelectionAdvisor:
    """Optional LLM-backed reviewer for initial tool selection."""

    def __init__(self, llm=None, enabled: bool = False, policy=None):
        self.llm = llm
        self.enabled = enabled
        self.policy = policy or ToolSelectionPolicy()
        self.strategy_advisor = StrategyAdvisor(
            llm=llm,
            enabled=enabled,
        )

    def recommend(self, context: ToolSelectionContext) -> ToolSelectionDecision:
        if isinstance(context, StrategyContext):
            action = self.strategy_advisor.recommend(context)
            return ToolSelectionDecision(
                tool=action.tool or context.deterministic_tool,
                confidence=(
                    None if action.confidence is None else str(action.confidence)
                ),
                rationale=action.reason,
                alternatives=action.alternatives,
                warnings=action.warnings,
                source=action.source,
                raw=action.raw,
                error=action.error,
            )

        if not self.enabled or self.llm is None:
            return ToolSelectionDecision(
                tool=context.deterministic_tool,
                confidence=None,
                rationale="LLM tool-selection advisor is disabled.",
                alternatives=[],
                warnings=[],
                source="disabled",
            )

        try:
            raw = self.llm(
                TOOL_SELECTION_SYSTEM_PROMPT,
                json.dumps(context.to_dict(), indent=2),
            )
            decision = self._normalize(self._parse_json(raw), raw)
        except Exception as exc:
            decision = ToolSelectionDecision(
                tool=context.deterministic_tool,
                confidence="low",
                rationale=(
                    "Tool-selection advisor failed; using deterministic tool."
                ),
                alternatives=[],
                warnings=["Fell back to deterministic tool selection."],
                source="fallback",
                error=str(exc),
            )

        return self.policy.validate(decision, context)

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize(self, data: dict, raw: str) -> ToolSelectionDecision:
        confidence = str(data.get("confidence", "low")).lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"

        alternatives = data.get("alternatives") or []
        if not isinstance(alternatives, list):
            alternatives = []
        warnings = data.get("warnings") or []
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        return ToolSelectionDecision(
            tool=str(data.get("tool", "")).strip(),
            confidence=confidence,
            rationale=str(data.get("rationale") or data.get("reason") or "").strip(),
            alternatives=[str(item) for item in alternatives],
            warnings=[str(item) for item in warnings],
            source="llm",
            raw=raw,
        )
