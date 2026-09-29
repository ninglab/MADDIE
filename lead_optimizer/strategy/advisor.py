import json
import re


ADVISOR_SYSTEM_PROMPT = """You advise the LeadOptimizer agent after a tool run.
Use the structured run facts only. Return one JSON object with:
action: one of stop, retry_same_tool, switch_tool, ask_clarification
tool: tool name when action is switch_tool, otherwise null
confidence: low, medium, or high
rationale: concise reasoning about what should happen next
retry_instruction: optional concise instruction for how a retry should differ
sampling_temperature: optional number from 0.1 to 1.5 for stochastic retries

Do not write the final user-facing summary. The agent will generate factual
user messages from structured run facts. Do not recommend excluded or
unavailable tools.

Use final_target_status as authoritative. If it is partial, none_met, failed,
or validated_smiles_only, do not describe the property target as fully met.
If final_target_status is validated_smiles_only or final_scored_outputs is 0,
the property target was not evaluated; do not infer that zero molecules met or
failed the target.
requested_outputs_per_input is per input molecule; compare target hits against
final_scored_outputs/expected_total_outputs, not just against the requested
per-input count. If follow-up actions are available and the target is not met,
consider switch_tool or retry_same_tool before stop; retry_same_tool is only
useful when current_tool_retryable is true. For retry_same_tool, explain the
material variation in retry_instruction; do not just say "try again."
"""

LLM_ACTIONS = {
    "stop",
    "retry_same_tool",
    "switch_tool",
    "ask_clarification",
}


class DecisionMessageBuilder:
    """Build short policy reasons/messages from one shared vocabulary."""

    def fallback_recommendation(self, context: dict) -> dict:
        status = context.get("final_target_status")
        if status == "met":
            return self._recommend(
                "stop",
                "The selected output met the scored property target.",
            )
        if status == "partial":
            return self._recommend(
                "stop",
                "The selected output partially met the property target.",
            )
        if status == "none_met":
            return self._recommend(
                "ask_clarification",
                "No scored outputs met the target; ask whether to relax the "
                "target, try another tool, or spend more compute.",
            )
        return self._recommend(
            "stop",
            "The output was not numerically scored against the property target, "
            "so report validated outputs.",
        )

    def policy_reason(
        self,
        context: dict,
        policy_action: str,
        tool: str | None,
    ) -> str:
        hits, scored = self._hits(context)
        if policy_action == "stop_success":
            return f"All scored outputs met the property target ({hits}/{scored})."
        if policy_action == "report_partial":
            return f"Only some scored outputs met the property target ({hits}/{scored})."
        if policy_action == "retry_same_tool":
            return "The target was not fully met, and retrying the same tool is allowed."
        if policy_action == "switch_tool":
            if context.get("final_target_status") is None:
                return (
                    "Outputs were valid SMILES, but numeric property status was "
                    f"not scored; switching to {tool} is allowed."
                )
            return f"The target was not fully met, and switching to {tool} is allowed."
        if policy_action == "ask_clarification":
            return "The next safe step requires user clarification."
        if policy_action == "report_failure":
            return "No optimized output was produced."
        if policy_action == "report_validated_only":
            return "Outputs were validated, but no numeric property target was scored."
        return "Report the current LeadOptimizer result."

    def user_message(
        self,
        context: dict,
        policy_action: str,
        tool: str | None,
    ) -> str:
        hits, scored = self._hits(context)
        rows = context.get("row_count")
        expected = context.get("expected_total_outputs")

        if policy_action == "stop_success":
            return (
                f"Property target met: {hits}/{scored} scored outputs met the "
                "target. Report the final output CSV."
            )
        if policy_action == "report_partial":
            return (
                f"Partial property-target success: {hits}/{scored} scored outputs "
                "met the target. Report the current output rather than continue "
                "automatically."
            )
        if policy_action == "retry_same_tool":
            if self._is_validated_only_context(context):
                count_note = f"{rows} valid unique output row(s)"
                if expected is not None:
                    count_note += f" out of {expected} requested"
                return (
                    f"Validated output count is incomplete ({count_note}); "
                    "retrying the same tool with a materially different strategy "
                    "is allowed."
                )
            return (
                f"Target status is {context.get('final_target_status')}; consider "
                "retrying the same tool if more compute is acceptable."
            )
        if policy_action == "switch_tool":
            status = context.get("final_target_status") or "validated_smiles_only"
            return f"Target status is {status}; consider switching to {tool}."
        if policy_action == "ask_clarification":
            return (
                "Ask the user whether to relax the target, try another tool, or "
                "spend more compute."
            )
        if policy_action == "report_failure":
            return "No optimized output was produced; report the failure details."
        if policy_action == "report_validated_only":
            count_note = f"{rows} output row(s)"
            if expected is not None:
                count_note += f" out of {expected} expected"
            return (
                f"Outputs were validated as SMILES ({count_note}), but no numeric "
                "property target status is available."
            )
        return "Report the current LeadOptimizer result."

    def _recommend(self, action: str, rationale: str) -> dict:
        return {
            "action": action,
            "tool": None,
            "confidence": "medium",
            "rationale": rationale,
        }

    def _hits(self, context: dict) -> tuple:
        return context.get("final_target_hits"), context.get("final_scored_outputs")

    def _is_validated_only_context(self, context: dict) -> bool:
        if context.get("final_target_status") == "validated_smiles_only":
            return True
        return (
            context.get("row_count", 0) > 0
            and not context.get("final_scored_outputs")
            and context.get("final_target_hits") is None
        )


class AllowedActionPolicy:
    """Compute legal next actions from status, budget, and tool capability."""

    def actions(self, context: dict, target_status: str) -> list[str]:
        return self.describe(context, target_status)["allowed_actions"]

    def describe(self, context: dict, target_status: str) -> dict:
        available_tools = set(context.get("available_tools") or [])
        excluded_tools = set(context.get("excluded_tools") or [])
        tried_tools = {
            attempt.get("tool")
            for attempt in context.get("optimization_attempts") or []
            if attempt.get("tool")
        }
        current_tool = context.get("tool")
        tool_retryable = bool(context.get("current_tool_retryable"))
        can_execute_more = (
            not context.get("execution_enabled")
            or context.get("remaining_tool_steps", 1) > 0
        )
        incomplete_validated_output = self._has_incomplete_validated_output(context)
        complete_validated_output = self._has_complete_validated_output(context)
        current_tool_requested = self._current_tool_was_user_requested(context)
        has_numeric_objective = self._has_numeric_objective(context)
        has_scoring_tool_context = "property_scoring_tools" in context
        requires_scoring_tool = (
            target_status in {"partial", "none_met"}
            and has_numeric_objective
            and has_scoring_tool_context
        )
        property_scoring_tools = set(context.get("property_scoring_tools") or [])
        switch_candidates = [
            tool for tool in available_tools
            if (
                tool not in excluded_tools
                and tool != current_tool
                and tool not in tried_tools
            )
        ]
        alternatives = [
            tool for tool in switch_candidates
            if not requires_scoring_tool or tool in property_scoring_tools
        ]
        retry_rule = self._retry_rule(
            target_status=target_status,
            tool_retryable=tool_retryable,
            can_execute_more=can_execute_more,
            incomplete_validated_output=incomplete_validated_output,
        )
        switch_rule = self._switch_rule(
            target_status=target_status,
            current_tool=current_tool,
            can_execute_more=can_execute_more,
            alternatives=alternatives,
            switch_candidates=switch_candidates,
            current_tool_requested=current_tool_requested,
            complete_validated_output=complete_validated_output,
            allow_expensive_scoring_followup=(
                self._allows_expensive_scoring_followup(context)
            ),
            has_numeric_objective=has_numeric_objective,
            requires_scoring_tool=requires_scoring_tool,
        )
        retryable = retry_rule["allowed"]
        if not switch_rule["allowed"]:
            alternatives = []

        has_alternatives = bool(alternatives)

        if target_status == "met":
            actions = ["stop_success"]
        elif target_status == "partial":
            actions = ["report_partial", "ask_clarification"]
            if retryable:
                actions.insert(1, "retry_same_tool")
            if has_alternatives:
                actions.insert(2 if retryable else 1, "switch_tool")
        elif target_status == "none_met":
            actions = ["ask_clarification"]
            if retryable:
                actions.insert(0, "retry_same_tool")
            if has_alternatives:
                actions.insert(1 if retryable else 0, "switch_tool")
        elif target_status == "failed":
            actions = ["report_failure", "ask_clarification"]
            if has_alternatives:
                actions.insert(1, "switch_tool")
        elif target_status == "validated_smiles_only":
            actions = ["report_validated_only", "ask_clarification"]
            if retryable:
                actions.insert(0, "retry_same_tool")
            if has_alternatives:
                insert_at = 2 if retryable else 1
                actions.insert(insert_at, "switch_tool")
        else:
            actions = ["report_validated_only", "ask_clarification"]

        return {
            "target_status": target_status,
            "allowed_actions": actions,
            "legal_switch_tools": sorted(alternatives),
            "can_retry_same_tool": retryable,
            "can_execute_more": can_execute_more,
            "current_tool": current_tool,
            "tried_tools": sorted(tried_tools),
            "excluded_tools": sorted(excluded_tools),
            "retry_rule": retry_rule,
            "switch_rule": {
                **switch_rule,
                "legal_tools": sorted(alternatives),
                "candidate_tools": sorted(switch_candidates),
                "requires_property_scoring_tool": requires_scoring_tool,
            },
        }

    def _retry_rule(
        self,
        target_status: str,
        tool_retryable: bool,
        can_execute_more: bool,
        incomplete_validated_output: bool,
    ) -> dict:
        if target_status == "met":
            return {
                "allowed": False,
                "reason": "target already met",
                "requires_material_variation": False,
            }
        if target_status not in {"partial", "none_met", "validated_smiles_only"}:
            return {
                "allowed": False,
                "reason": f"retry is not useful for target status {target_status}",
                "requires_material_variation": False,
            }
        if not can_execute_more:
            return {
                "allowed": False,
                "reason": "no remaining tool-execution budget",
                "requires_material_variation": True,
            }
        if not tool_retryable:
            return {
                "allowed": False,
                "reason": "current tool does not declare a useful retry strategy",
                "requires_material_variation": True,
            }
        if target_status == "validated_smiles_only" and not incomplete_validated_output:
            return {
                "allowed": False,
                "reason": (
                    "validated output count is complete; report valid SMILES "
                    "instead of retrying"
                ),
                "requires_material_variation": True,
            }
        return {
            "allowed": True,
            "reason": "target not fully met and current tool can vary on retry",
            "requires_material_variation": True,
        }

    def _switch_rule(
        self,
        target_status: str,
        current_tool: str | None,
        can_execute_more: bool,
        alternatives: list[str],
        switch_candidates: list[str],
        current_tool_requested: bool,
        complete_validated_output: bool,
        allow_expensive_scoring_followup: bool,
        has_numeric_objective: bool,
        requires_scoring_tool: bool,
    ) -> dict:
        if target_status == "met":
            return {"allowed": False, "reason": "target already met"}
        if target_status not in {"partial", "none_met", "failed", "validated_smiles_only"}:
            return {
                "allowed": False,
                "reason": f"switch is not useful for target status {target_status}",
            }
        if not can_execute_more:
            return {"allowed": False, "reason": "no remaining tool-execution budget"}
        if target_status == "validated_smiles_only":
            if current_tool_requested:
                return {
                    "allowed": False,
                    "reason": (
                        "current tool was explicitly requested; report validated "
                        "outputs instead of switching automatically"
                    ),
                }
            if (
                current_tool == "gellmo_c"
                and complete_validated_output
                and not allow_expensive_scoring_followup
            ):
                return {
                    "allowed": False,
                    "reason": (
                        "gellmo_c returned the requested number of valid outputs; "
                        "automatic MolOpt scoring follow-up is disabled because it "
                        "can be slow. Ask for numeric scoring or allow more compute "
                        "to continue."
                    ),
                }
            if not has_numeric_objective:
                return {
                    "allowed": False,
                    "reason": "no numeric objective was parsed for a scoring-tool switch",
                }
        if not alternatives:
            reason = "no legal untried compatible tool is available"
            if requires_scoring_tool and switch_candidates:
                reason = "untried alternatives exist, but none can score the numeric target"
            return {"allowed": False, "reason": reason}
        return {
            "allowed": True,
            "reason": "legal untried compatible tool is available",
        }

    def _has_numeric_objective(self, context: dict) -> bool:
        objective = (context.get("plan") or {}).get("objective") or {}
        return bool(objective.get("property") and objective.get("direction"))

    def _has_incomplete_validated_output(self, context: dict) -> bool:
        expected = context.get("expected_total_outputs")
        rows = context.get("row_count")
        return (
            expected is not None
            and rows is not None
            and rows > 0
            and rows < expected
        )

    def _has_complete_validated_output(self, context: dict) -> bool:
        expected = context.get("expected_total_outputs")
        rows = context.get("row_count")
        return (
            expected is not None
            and rows is not None
            and expected > 0
            and rows >= expected
        )

    def _allows_expensive_scoring_followup(self, context: dict) -> bool:
        prompt = str(context.get("prompt") or "").lower()
        phrases = (
            "score",
            "scoring",
            "evaluate",
            "evaluation",
            "verify",
            "confirm",
            "numeric",
            "property value",
            "property score",
        )
        return any(phrase in prompt for phrase in phrases)

    def _current_tool_was_user_requested(self, context: dict) -> bool:
        decision_packet = context.get("decision_packet") or {}
        tool_constraint = decision_packet.get("tool_constraint") or {}
        return (
            tool_constraint.get("tool") == context.get("tool")
            and tool_constraint.get("mode") in {"preferred", "exclusive"}
        )


class NextActionAdvisor:
    """Optional LLM advisor for next-step recommendations after a run.

    The LLM recommends an action. Deterministic policy normalizes that
    recommendation into a stable decision contract for the agent/orchestrator.
    """

    def __init__(
        self,
        llm=None,
        enabled: bool = False,
        action_policy: AllowedActionPolicy | None = None,
        message_builder: DecisionMessageBuilder | None = None,
    ):
        self.llm = llm
        self.enabled = enabled
        self.action_policy = action_policy or AllowedActionPolicy()
        self.message_builder = message_builder or DecisionMessageBuilder()

    def recommend(self, context: dict) -> dict:
        context = self._with_legal_actions(context)
        if not self.enabled or self.llm is None:
            return self._policy_decision(
                context=context,
                recommendation={
                    "source": "disabled",
                    "action": "none",
                    "tool": None,
                    "confidence": None,
                    "rationale": "LLM next-action advisor is disabled.",
                },
            )

        try:
            raw = self.llm(ADVISOR_SYSTEM_PROMPT, json.dumps(context, indent=2))
            recommendation = self._normalize_recommendation(
                self._parse_json(raw),
                context,
            )
            recommendation["source"] = "llm"
            recommendation["raw"] = raw
        except Exception as exc:
            recommendation = self._fallback_recommendation(context)
            recommendation["source"] = "fallback"
            recommendation["error"] = str(exc)

        return self._policy_decision(context, recommendation)

    def _with_legal_actions(self, context: dict) -> dict:
        context = dict(context)
        target_status = self._target_status(context)
        context["legal_actions"] = self.action_policy.describe(
            context,
            target_status,
        )
        return context

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _normalize_recommendation(self, data: dict, context: dict) -> dict:
        action = str(data.get("action", "")).strip().lower().replace("-", "_")
        if action not in LLM_ACTIONS:
            raise ValueError(f"Unsupported next action: {action}")

        tool = data.get("tool")
        available_tools = set(context.get("available_tools") or [])
        excluded_tools = set(context.get("excluded_tools") or [])
        if action == "switch_tool":
            if not tool:
                raise ValueError("switch_tool recommendations require a tool.")
            if available_tools and tool not in available_tools:
                raise ValueError(f"Recommended tool is unavailable: {tool}")
            if tool in excluded_tools:
                raise ValueError(f"Recommended tool was excluded by the user: {tool}")
        else:
            tool = None

        confidence = data.get("confidence")
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"

        return {
            "action": action,
            "tool": tool,
            "confidence": confidence,
            "rationale": str(
                data.get("rationale")
                or data.get("reason")
                or data.get("user_message")
                or ""
            ).strip(),
            "retry_instruction": self._retry_instruction(data),
            "sampling_temperature": self._sampling_temperature(data),
        }

    def _fallback_recommendation(self, context: dict) -> dict:
        return self.message_builder.fallback_recommendation(context)

    def _policy_decision(self, context: dict, recommendation: dict) -> dict:
        target_status = self._target_status(context)
        legal_actions = context.get("legal_actions") or self.action_policy.describe(
            context,
            target_status,
        )
        allowed_actions = legal_actions["allowed_actions"]
        policy_action = self._policy_action(
            recommendation=recommendation,
            allowed_actions=allowed_actions,
            target_status=target_status,
        )
        tool = None
        if policy_action == "switch_tool":
            legal_switch_tools = legal_actions.get("legal_switch_tools") or []
            recommended_tool = recommendation.get("tool")
            if recommended_tool in legal_switch_tools:
                tool = recommended_tool
            elif legal_switch_tools:
                tool = legal_switch_tools[0]

        retry_instruction = None
        sampling_temperature = None
        if policy_action == "retry_same_tool":
            retry_instruction = recommendation.get("retry_instruction")
            sampling_temperature = recommendation.get("sampling_temperature")

        return {
            "source": recommendation.get("source"),
            "action": policy_action,
            "policy_action": policy_action,
            "allowed_actions": allowed_actions,
            "legal_actions": legal_actions,
            "excluded_tools": context.get("excluded_tools") or [],
            "target_status": target_status,
            "target_hits": context.get("final_target_hits"),
            "scored_outputs": context.get("final_scored_outputs"),
            "validated_outputs": context.get("final_validated_outputs")
            or context.get("row_count"),
            "expected_total_outputs": context.get("expected_total_outputs"),
            "tool": tool,
            "confidence": recommendation.get("confidence"),
            "policy_reason": self.message_builder.policy_reason(
                context,
                policy_action,
                tool,
            ),
            "llm_recommendation": {
                key: recommendation.get(key)
                for key in (
                    "action",
                    "tool",
                    "confidence",
                    "rationale",
                    "retry_instruction",
                    "sampling_temperature",
                )
            },
            "retry_instruction": retry_instruction,
            "sampling_temperature": sampling_temperature,
            "user_message": self.message_builder.user_message(
                context,
                policy_action,
                tool,
            ),
            "raw": recommendation.get("raw"),
            "error": recommendation.get("error"),
        }

    def _target_status(self, context: dict) -> str:
        if context.get("status") == "failed":
            return "failed"
        status = context.get("final_target_status")
        if status:
            return status
        if context.get("row_count", 0) > 0:
            return "validated_smiles_only"
        return "failed"

    def _allowed_policy_actions(self, context: dict, target_status: str) -> list[str]:
        return self.action_policy.actions(context, target_status)

    def _policy_action(
        self,
        recommendation: dict,
        allowed_actions: list[str],
        target_status: str,
    ) -> str:
        llm_action = recommendation.get("action")
        if target_status == "met":
            return "stop_success"
        if target_status == "partial" and llm_action == "stop":
            return "report_partial"
        if target_status == "none_met" and llm_action == "stop":
            return allowed_actions[0]
        if target_status == "failed" and llm_action == "stop":
            return "report_failure"
        if target_status == "validated_smiles_only":
            if llm_action == "retry_same_tool" and "retry_same_tool" in allowed_actions:
                return "retry_same_tool"
            if llm_action == "switch_tool" and "switch_tool" in allowed_actions:
                return "switch_tool"
            return "report_validated_only"

        if llm_action in {"retry_same_tool", "ask_clarification"}:
            return llm_action if llm_action in allowed_actions else allowed_actions[0]
        if llm_action == "switch_tool":
            return "switch_tool" if "switch_tool" in allowed_actions else allowed_actions[0]
        return allowed_actions[0]

    def _retry_instruction(self, data: dict) -> str | None:
        instruction = str(
            data.get("retry_instruction")
            or data.get("revised_instruction")
            or data.get("instruction")
            or ""
        ).strip()
        if not instruction:
            return None
        return instruction[:600]

    def _sampling_temperature(self, data: dict) -> float | None:
        value = data.get("sampling_temperature")
        if value is None:
            value = data.get("temperature")
        if value is None:
            return None
        try:
            temperature = float(value)
        except (TypeError, ValueError):
            return None
        if 0.1 <= temperature <= 1.5:
            return temperature
        return None
