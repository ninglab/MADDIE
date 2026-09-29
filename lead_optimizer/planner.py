import json
import re

from .config.benchmark import load_benchmark_evidence
from .core import find_csv_in_prompt
from .errors import LeadOptimizerRequestError
from .oracles.specs import (
    find_property_names,
    prompt_objective_fields,
)
from .planning.parser import (
    apply_prompt_objectives,
    compact_excerpt,
    extract_json_object,
    first_objective,
    objective_kwargs_from_prompt,
    optional_float,
    parse_json_plan,
    parse_objective_dict,
    parse_objective_list,
    planner_error,
)
from .planning.prompts import PLANNER_SYSTEM_PROMPT
from .planning.schemas import OptimizationPlan
from .planning.selection import (
    build_decision_packet,
    concise_selection_reason,
    find_excluded_tools,
    find_explicit_tool,
    find_tool_request,
    is_exclusive_tool_request,
    multi_property_unavailable_message,
    observed_tool_preference,
    order_measured_results,
)


class Planner:
    """Small interface shared by deterministic and LLM-backed planners."""

    name = "planner"

    def with_tools(self, tool_docs: str):
        return self

    def with_tool_capabilities(self, multi_property_tools: list[str]):
        return self

    def plan(self, prompt: str) -> OptimizationPlan:
        raise NotImplementedError


class RuleBasedPlanner(Planner):
    name = "rule_based_planner"

    def __init__(self, default_tool: str = "fake_optimizer", default_num_outputs: int = 3):
        self.default_tool = default_tool
        self.default_num_outputs = default_num_outputs

    def plan(self, prompt: str) -> OptimizationPlan:
        # Current local baseline: grab the CSV filename and use the full prompt
        # as the optimization instruction.
        input_file = find_csv_in_prompt(prompt)

        return OptimizationPlan(
            task_type="optimize",
            input_type="csv",
            input_file=input_file,
            adjustment=prompt,
            tool=self.default_tool,
            num_outputs=self.default_num_outputs,
            **objective_kwargs_from_prompt(prompt),
        )


class JSONPlanner(Planner):
    name = "json_planner"

    def __init__(self, llm, benchmark_path: str | None = None):
        self.llm = llm
        self.tool_docs = ""
        self.benchmark_path = benchmark_path
        self.last_benchmark_evidence = {}
        self.last_decision_packet = {}
        self.multi_property_tools = []

    def with_tools(self, tool_docs: str):
        self.tool_docs = tool_docs
        return self

    def with_tool_capabilities(self, multi_property_tools: list[str]):
        self.multi_property_tools = multi_property_tools
        return self

    def plan(self, prompt: str) -> OptimizationPlan:
        # JSONPlanner lets a small instruction model produce the same plan shape
        # as RuleBasedPlanner, while execution stays deterministic downstream.
        system_prompt = PLANNER_SYSTEM_PROMPT
        legal_tools = [
            line[2:].split(":", 1)[0]
            for line in self.tool_docs.splitlines()
            if line.startswith("- ") and ":" in line
        ]
        self.last_benchmark_evidence = {}
        self.last_decision_packet = {}
        if self.benchmark_path:
            objective = prompt_objective_fields(prompt)["objective"]
            self.last_benchmark_evidence = load_benchmark_evidence(
                self.benchmark_path,
                objective,
                legal_tools,
            )
            if self.last_benchmark_evidence:
                requested_properties = find_property_names(prompt)
                self.last_benchmark_evidence["requested_properties"] = (
                    requested_properties
                )
                self.last_benchmark_evidence["property_count"] = len(
                    requested_properties
                )
        if legal_tools:
            self.last_decision_packet = build_decision_packet(
                prompt,
                self.last_benchmark_evidence,
                legal_tools,
                self.multi_property_tools,
            )
            if not self.last_decision_packet.get("eligible_tools"):
                excluded = find_excluded_tools(prompt, legal_tools)
                if self.last_decision_packet["request"]["mode"] == "multi_property":
                    raise LeadOptimizerRequestError(
                        multi_property_unavailable_message(
                            excluded,
                            self.multi_property_tools,
                        ),
                        metadata={
                            "error_category": "multi_property_tool_unavailable",
                            "decision_packet": self.last_decision_packet,
                            "excluded_tools": excluded,
                            "multi_property_tools": self.multi_property_tools,
                        },
                    )
                exclusion_text = (
                    f" after excluding: {', '.join(excluded)}"
                    if excluded
                    else ""
                )
                raise LeadOptimizerRequestError(
                    "No available optimization tool satisfies the request"
                    f"{exclusion_text}.",
                    metadata={
                        "error_category": "tool_unavailable",
                        "decision_packet": self.last_decision_packet,
                        "excluded_tools": excluded,
                    },
                )
            system_prompt += (
                "\n\nDecision packet:\n"
                f"{json.dumps(self.last_decision_packet, indent=2)}"
            )
            if (
                self.last_decision_packet.get("eligible_tools") == legal_tools
                and "threshold_comparison" not in self.last_decision_packet
            ):
                system_prompt += f"\n\nAvailable tools:\n{self.tool_docs}"
        if not self.last_decision_packet and self.tool_docs:
            system_prompt += f"\n\nAvailable tools:\n{self.tool_docs}"
        try:
            raw_text = self.llm(system_prompt, prompt)
        except Exception as exc:
            requested_properties = find_property_names(prompt)
            raise LeadOptimizerRequestError(
                "I could not plan the lead optimization request because the "
                "planner LLM call failed. No optimization was run and no output "
                "CSV was written.",
                status="planner_failed",
                metadata={
                    "error_category": "planner_llm_failed",
                    "planner_error": str(exc),
                    "decision_packet": self.last_decision_packet,
                    "requested_properties": requested_properties,
                    "objective_mode": (
                        "multi_property"
                        if len(requested_properties) > 1
                        else "single_property"
                    ),
                    "outputs": {"csv": None, "debug_csvs": [], "log": None},
                },
            ) from exc
        try:
            plan = parse_json_plan(raw_text)
        except LeadOptimizerRequestError as exc:
            requested_properties = find_property_names(prompt)
            exc.metadata.update({
                "decision_packet": self.last_decision_packet,
                "requested_properties": requested_properties,
                "objective_mode": (
                    "multi_property"
                    if len(requested_properties) > 1
                    else "single_property"
                ),
            })
            raise
        # Tool choice comes from the LLM, but execution must preserve the user's
        # exact properties, directions, thresholds, and ranges when the
        # deterministic parser can recover them. If it cannot, keep the LLM's
        # structured objective interpretation.
        plan.adjustment = prompt
        plan = apply_prompt_objectives(plan, prompt)
        eligible_tools = self.last_decision_packet.get("eligible_tools", [])
        if eligible_tools and plan.tool not in eligible_tools:
            plan.tool = eligible_tools[0]
        if eligible_tools:
            plan.reason = concise_selection_reason(
                self.last_decision_packet,
                plan.tool,
            )
        return plan


class LLMPlanner(Planner):
    name = "llm_planner"

    def __init__(self, llm):
        self.llm = llm

    def plan(self, prompt: str) -> OptimizationPlan:
        raise NotImplementedError("LLMPlanner is not implemented yet.")


class SmartPlanner(Planner):
    """Heuristic planner that selects tool based on NL hints."""

    name = "smart_planner"

    def __init__(
        self,
        available_tools: dict[str, str],
        default_num_outputs: int = 3,
        multi_property_tools: list[str] | None = None,
    ):
        """
        Args:
            available_tools: Dict mapping tool name → short description.
                Example: {"gellmo_c": "generate molecules", "molopt_screening": "screen"}
            default_num_outputs: Default number of outputs if not specified.
        """
        self.available_tools = available_tools
        self.default_num_outputs = default_num_outputs
        self.multi_property_tools = multi_property_tools or []

    def with_tools(self, tool_docs: str):
        # Parse tool docs if provided (e.g., from ToolChest.documentation())
        return self

    def with_tool_capabilities(self, multi_property_tools: list[str]):
        self.multi_property_tools = multi_property_tools
        return self

    def plan(self, prompt: str) -> OptimizationPlan:
        """
        Parse prompt for tool hints and oracle targets.

        Heuristics:
        - If prompt mentions "gellmo", "generate", "invent" → gellmo_c
        - If prompt mentions "screen", "rank", "evaluate" → molopt_screening
        - If prompt mentions algorithm name (graph_ga, smiles_ga, etc.) → molopt_<algorithm>
        - Otherwise, default to first available tool
        """
        prompt_lower = prompt.lower()
        input_file = find_csv_in_prompt(prompt)

        # Tool selection heuristics
        selected_tool = self._select_tool(prompt_lower)
        decision_packet = build_decision_packet(
            prompt,
            {},
            list(self.available_tools),
            self.multi_property_tools,
        )

        # Extract num_outputs if specified (e.g., "give me 5 variants")
        num_outputs = self._extract_num_outputs(prompt_lower) or self.default_num_outputs

        return OptimizationPlan(
            task_type="optimize",
            input_type="csv",
            input_file=input_file,
            adjustment=prompt,
            tool=selected_tool,
            num_outputs=num_outputs,
            reason=concise_selection_reason(decision_packet, selected_tool),
            **objective_kwargs_from_prompt(prompt),
        )

    def _select_tool(self, prompt_lower: str) -> str:
        """Select tool based on keyword matching."""
        excluded = set(
            find_excluded_tools(prompt_lower, list(self.available_tools))
        )
        available_tools = {
            name: description
            for name, description in self.available_tools.items()
            if name not in excluded
        }
        if not available_tools:
            raise LeadOptimizerRequestError(
                "The request excluded every available optimization tool. "
                "Allow at least one configured tool and try again.",
                metadata={
                    "error_category": "tool_unavailable",
                    "excluded_tools": sorted(excluded),
                },
            )

        explicit_tool = find_explicit_tool(prompt_lower, list(available_tools))
        if explicit_tool:
            return explicit_tool

        properties = find_property_names(prompt_lower)
        if len(properties) > 1:
            capable_tools = [
                tool
                for tool in self.multi_property_tools
                if tool in available_tools
            ]
            if capable_tools:
                return capable_tools[0]
            raise LeadOptimizerRequestError(
                multi_property_unavailable_message(
                    excluded,
                    self.multi_property_tools,
                ),
                metadata={
                    "error_category": "multi_property_tool_unavailable",
                    "excluded_tools": sorted(excluded),
                    "multi_property_tools": self.multi_property_tools,
                },
            )

        if "gellmo_c" in available_tools and len(properties) == 1:
            preference = observed_tool_preference(
                prompt_lower,
                properties,
                list(available_tools),
            )
            if preference:
                return "gellmo_c"

        keywords = {
            "gellmo_c": ["gellmo", "generate", "invent", "create"],
            "molopt_screening": ["screen", "rank", "filter", "select"],
            "molopt_graph_ga": ["graph_ga"],
            "molopt_smiles_ga": ["smiles_ga", "string-level"],
            "molopt_selfies_ga": ["selfies", "selfies_ga"],
            "molopt_stoned": ["stoned"],
            "molopt_gpbo": ["gpbo", "bayesian optimization"],
            "molopt_graph_mcts": ["graph mcts", "graph_mcts", "monte carlo tree"],
        }

        for tool_name, hint_words in keywords.items():
            if tool_name in available_tools:
                if any(word in prompt_lower for word in hint_words):
                    return tool_name

        if "molopt" in prompt_lower:
            molopt_tools = [
                name for name in available_tools if name.startswith("molopt_")
            ]
            if "molopt_screening" in molopt_tools:
                return "molopt_screening"
            if molopt_tools:
                return molopt_tools[0]

        # Fallback: return first available tool
        return next(iter(available_tools))

    def _extract_num_outputs(self, prompt_lower: str) -> int | None:
        """Extract number of outputs from phrases like 'give me 5 variants' or 'top 3'."""
        match = re.search(r"(?:give me|top|return|generate)\s+(\d+)", prompt_lower)
        if match:
            return int(match.group(1))
        return None
