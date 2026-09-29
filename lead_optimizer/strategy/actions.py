from copy import copy

from ..tools import OptimizationTool, ToolChest


class AgentActionPlanner:
    """Prepare bounded follow-up tool actions without running tools."""

    def __init__(self, tool_chest: ToolChest, max_attempts: int):
        self.tool_chest = tool_chest
        self.max_attempts = max_attempts

    @staticmethod
    def attempt_satisfied(attempt: dict) -> bool:
        target_hits = attempt.get("target_hits")
        scored_outputs = attempt.get("scored_outputs")
        return (
            target_hits is not None
            and scored_outputs is not None
            and scored_outputs > 0
            and target_hits >= scored_outputs
        )

    @staticmethod
    def selected_attempt(attempts: list[dict]) -> dict:
        return next(
            (
                attempt
                for attempt in attempts
                if attempt.get("selected_output")
            ),
            attempts[-1] if attempts else {},
        )

    @staticmethod
    def is_better_attempt(candidate: dict, current: dict, objective) -> bool:
        candidate_hits = candidate.get("target_hits")
        current_hits = current.get("target_hits")
        if candidate_hits is not None and current_hits is None:
            return True
        if candidate_hits is None and current_hits is not None:
            return False
        if candidate_hits is not None and current_hits is not None:
            if candidate_hits != current_hits:
                return candidate_hits > current_hits

        candidate_scored = candidate.get("scored_outputs", 0)
        current_scored = current.get("scored_outputs", 0)
        if candidate_scored != current_scored:
            return candidate_scored > current_scored

        candidate_valid = (
            candidate.get("validated_outputs")
            if candidate.get("validated_outputs") is not None
            else candidate.get("row_count")
        )
        current_valid = (
            current.get("validated_outputs")
            if current.get("validated_outputs") is not None
            else current.get("row_count")
        )
        if candidate_valid is not None and current_valid is not None:
            if candidate_valid != current_valid:
                return candidate_valid > current_valid

        candidate_value = candidate.get("best_property_value")
        current_value = current.get("best_property_value")
        if candidate_value is None or current_value is None or objective is None:
            return False
        if objective.direction == "decrease":
            return candidate_value < current_value
        if objective.direction == "increase":
            return candidate_value > current_value
        if objective.direction == "target_range":
            midpoint = (objective.minimum + objective.maximum) / 2
            return abs(candidate_value - midpoint) < abs(current_value - midpoint)
        return False

    def prepare_retry_tool(
        self,
        tool: OptimizationTool,
        step: int,
        attempts: list[dict] | None = None,
        decision: dict | None = None,
    ) -> tuple[OptimizationTool, dict]:
        """Return a retry tool copy plus the config change used for this retry."""
        retry_tool = tool
        changes = []
        decision = decision or {}
        retry_instruction = decision.get("retry_instruction")
        selected_attempt = self.selected_attempt(attempts or [])
        budget_context = self._retry_budget_context(selected_attempt)

        if hasattr(tool, "seed"):
            retry_tool = copy(tool)
            original_seed = int(getattr(tool, "seed"))
            retry_seed = original_seed + step
            retry_tool.seed = retry_seed
            changes.append({
                "parameter": "seed",
                "from": original_seed,
                "to": retry_seed,
                "reason": "vary stochastic optimizer trajectory",
            })

        if hasattr(tool, "retry_attempt_offset"):
            if retry_tool is tool:
                retry_tool = copy(tool)
            attempt_block = self.max_attempts
            original_offset = int(getattr(tool, "retry_attempt_offset"))
            retry_offset = original_offset + (step * attempt_block)
            retry_tool.retry_attempt_offset = retry_offset
            changes.append({
                "parameter": "retry_attempt_offset",
                "from": original_offset,
                "to": retry_offset,
                "reason": (
                    "advance GeLLMO instruction/temperature schedule for a "
                    "fresh agent-level retry"
                ),
            })

        sampling_temperature = decision.get("sampling_temperature")
        if sampling_temperature is not None and hasattr(tool, "retry_temperatures"):
            if retry_tool is tool:
                retry_tool = copy(tool)
            original_temperature = getattr(tool, "sampling_temperature_override", None)
            retry_tool.sampling_temperature_override = sampling_temperature
            changes.append({
                "parameter": "sampling_temperature_override",
                "from": original_temperature,
                "to": sampling_temperature,
                "reason": "use LLM-advised stochastic retry temperature",
            })

        if changes and retry_instruction:
            strategy = "vary_configuration_and_instruction"
        elif changes:
            strategy = "vary_configuration"
        elif retry_instruction:
            strategy = "revise_instruction"
        else:
            strategy = "tool_internal_retry"
        metadata = {
            "strategy": strategy,
            "tool": tool.name,
            "agent_step": step,
            "changes": changes,
            "budget_context": budget_context,
            "retry_instruction": retry_instruction,
            "sampling_temperature": sampling_temperature,
        }
        if not changes:
            metadata["note"] = (
                "No generic configuration change was available; relying on "
                "tool-internal retry behavior."
            )
        return retry_tool, metadata

    def prepare_switch_tool(
        self,
        decision: dict,
        decision_packet: dict,
        attempts: list[dict],
    ) -> tuple[OptimizationTool | None, dict]:
        tried_tools = {
            attempt.get("tool")
            for attempt in attempts
            if attempt.get("tool")
        }
        available_tools = set(self.tool_chest.available())
        excluded_tools = set(decision.get("excluded_tools") or [])
        legal_action_summary = decision.get("legal_actions") or {}
        legal_switch_tools = set(legal_action_summary.get("legal_switch_tools") or [])
        ordered_candidates = self._ordered_switch_tool_names(decision_packet)
        for tool_name in sorted(available_tools):
            if tool_name not in ordered_candidates:
                ordered_candidates.append(tool_name)
        if "legal_switch_tools" not in legal_action_summary:
            legal_switch_tools = {
                tool_name for tool_name in ordered_candidates
                if (
                    tool_name in available_tools
                    and tool_name not in excluded_tools
                    and tool_name not in tried_tools
                )
            }
        legal_candidates = [
            tool_name for tool_name in ordered_candidates
            if tool_name in legal_switch_tools
        ]

        selected_tool = decision.get("tool")
        llm_requested_tool = (
            (decision.get("llm_recommendation") or {}).get("tool")
            or selected_tool
        )
        if selected_tool and selected_tool in legal_switch_tools:
            llm_tool_was_legal = selected_tool == llm_requested_tool
            return self.tool_chest.get(selected_tool), {
                "strategy": (
                    "switch_to_llm_recommended_tool"
                    if llm_tool_was_legal
                    else "switch_to_policy_fallback_tool"
                ),
                "tool": selected_tool,
                "reason": (
                    "LLM recommended a legal untried compatible tool."
                    if llm_tool_was_legal
                    else "LLM requested an illegal switch tool; policy selected "
                    "the next legal untried compatible tool."
                ),
                "llm_requested_tool": llm_requested_tool,
                "policy_selected_tool": selected_tool,
                "candidate_tools": legal_candidates,
                "tried_tools": sorted(tried_tools),
                "excluded_tools": sorted(excluded_tools),
                "legal_switch_tools": sorted(legal_switch_tools),
                "available_tools": sorted(available_tools),
            }

        if legal_candidates:
            tool_name = legal_candidates[0]
            return self.tool_chest.get(tool_name), {
                "strategy": "switch_to_next_compatible_tool",
                "tool": tool_name,
                "reason": (
                    "Selected the next legal untried tool from decision-packet "
                    "ordering plus tool-chest fallback candidates."
                ),
                "llm_requested_tool": llm_requested_tool,
                "policy_selected_tool": tool_name,
                "candidate_tools": legal_candidates,
                "tried_tools": sorted(tried_tools),
                "excluded_tools": sorted(excluded_tools),
                "legal_switch_tools": sorted(legal_switch_tools),
                "available_tools": sorted(available_tools),
            }

        return None, {
            "strategy": "no_untried_switch_tool",
            "reason": (
                "No legal untried switch tool was available from the decision "
                "packet or tool chest."
            ),
            "llm_requested_tool": llm_requested_tool,
            "candidate_tools": [],
            "tried_tools": sorted(tried_tools),
            "excluded_tools": sorted(excluded_tools),
            "legal_switch_tools": sorted(legal_switch_tools),
            "available_tools": sorted(available_tools),
        }

    def _retry_budget_context(self, attempt: dict) -> dict:
        settings = attempt.get("execution_settings") or {}
        keys = [
            "oracle_budget_source",
            "benchmark_calls_to_target",
            "benchmark_recommended_calls_per_molecule",
            "configured_max_oracle_calls_per_molecule",
            "max_oracle_calls_per_molecule",
        ]
        context = {
            key: settings[key]
            for key in keys
            if key in settings
        }
        recommended = context.get("benchmark_recommended_calls_per_molecule")
        actual = context.get("max_oracle_calls_per_molecule")
        configured = context.get("configured_max_oracle_calls_per_molecule")
        if recommended and actual and actual < recommended:
            context["budget_note"] = (
                "Initial run used less than the benchmark-derived recommendation."
            )
            if configured and actual >= configured:
                context["budget_limit"] = "configured_max_oracle_calls"
        return context

    @staticmethod
    def _ordered_switch_tool_names(decision_packet: dict) -> list[str]:
        comparison = decision_packet.get("threshold_comparison") or {}
        ordered_results = comparison.get("ordered_results") or []
        names = [
            result.get("tool")
            for result in ordered_results
            if result.get("tool")
        ]
        if names:
            return names
        return list(decision_packet.get("eligible_tools") or [])
