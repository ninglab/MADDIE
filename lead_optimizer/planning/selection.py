import re

from ..oracles.specs import find_property_names


QED_QUALITY_TOLERANCE = 0.01
QED_RELIABILITY_ORDER = {
    "molopt_graph_ga": 0,
    "molopt_stoned": 1,
    "molopt_smiles_ga": 2,
    "molopt_selfies_ga": 3,
    "molopt_gpbo": 4,
    "molopt_screening": 5,
}
GELLMO_PREFERRED_SINGLE_PROPERTIES = {
    "herg",
    "clintox",
    "mutagenicity",
    "carcinogens",
}
CHEM_REASONABLE_HINTS = (
    "chemically reasonable",
    "plausible",
    "close analog",
    "close analogue",
    "analog-like",
    "analogue-like",
)
TOOL_ALIASES = {
    "gellmo_c": ("gellmo", "gellmo c"),
    "molopt_graph_ga": ("graph ga", "graph genetic algorithm"),
    "molopt_graph_mcts": ("graph mcts", "graph monte carlo tree search"),
    "molopt_gpbo": ("gpbo", "bayesian optimization"),
    "molopt_mimosa": ("mimosa",),
    "molopt_moldqn": ("moldqn", "mol dqn"),
    "molopt_reinvent": ("reinvent",),
    "molopt_reinvent_selfies": ("reinvent selfies",),
    "molopt_screening": ("molopt screening",),
    "molopt_smiles_ga": ("smiles ga",),
    "molopt_selfies_ga": ("selfies ga",),
    "molopt_stoned": ("stoned",),
}
TOOL_EXCLUSION_PHRASES = (
    "anything other than",
    "anything but",
    "other than",
    "do not use",
    "don't use",
    "excluding",
    "except",
    "avoid",
    "without",
)
TOOL_EXCLUSIVE_PHRASES = (
    "only use",
    "use only",
    "exclusively use",
    "use exclusively",
    "nothing but",
    "no tool except",
    "no tools except",
    "do not use anything except",
    "don't use anything except",
)


def multi_property_unavailable_message(
    excluded_tools=(),
    multi_property_tools=(),
) -> str:
    excluded_capable_tools = [
        tool for tool in multi_property_tools if tool in excluded_tools
    ]
    if excluded_capable_tools:
        names = ", ".join(excluded_capable_tools)
        return (
            "This request asks for simultaneous multi-property optimization, "
            "but all configured tools that support it were excluded "
            f"({names}). Allow one of those tools or request each property separately."
        )
    return (
        "This request asks for simultaneous multi-property optimization, but "
        "no available tool supports it. Configure a multi-property optimization "
        "tool or request each property separately."
    )


def order_measured_results(measured_results: list[dict], evidence: dict) -> list[dict]:
    if evidence.get("property") != "qed" or not measured_results:
        return measured_results

    top10_scores = [
        result["mean_final_top10"]
        for result in measured_results
        if result.get("mean_final_top10") is not None
    ]
    if not top10_scores:
        return measured_results

    best_top10 = max(top10_scores)

    def sort_key(result: dict) -> tuple:
        top10 = result.get("mean_final_top10")
        quality_close = (
            top10 is not None
            and best_top10 - top10 <= QED_QUALITY_TOLERANCE
        )
        oracle_calls = (
            result.get("oracle_calls")
            if result.get("oracle_calls") is not None
            else float("inf")
        )
        return (
            0 if quality_close else 1,
            QED_RELIABILITY_ORDER.get(result["tool"], 99),
            -(top10 or -1.0),
            oracle_calls,
        )

    return sorted(measured_results, key=sort_key)


def build_decision_packet(
    prompt: str,
    evidence: dict,
    legal_tools: list[str],
    multi_property_tools: list[str] | None = None,
) -> dict:
    multi_property_tools = multi_property_tools or []
    properties = find_property_names(prompt)
    property_count = len(properties)
    mode = "multi_property" if property_count > 1 else "single_property"
    comparison = evidence.get("routing_threshold_comparison")
    excluded_tools = find_excluded_tools(prompt, legal_tools)
    legal_tools = [
        tool for tool in legal_tools if tool not in excluded_tools
    ]
    tool_request = find_tool_request(prompt, legal_tools)

    if mode == "multi_property":
        eligible_tools = [
            tool for tool in legal_tools if tool in multi_property_tools
        ]
        return {
            "request": {
                "properties": properties,
                "property_count": property_count,
                "mode": mode,
            },
            "eligible_tools": eligible_tools,
            "selection_rule": (
                "Choose a tool that supports simultaneous multi-property optimization."
            ),
            "selection_reason": "it supports multi-property optimization.",
            "tool_constraint": tool_request,
        }

    explicit_tool = tool_request.get("tool")
    if explicit_tool:
        exclusive = tool_request.get("mode") == "exclusive"
        return {
            "request": {
                "properties": properties,
                "property_count": property_count,
                "mode": mode,
            },
            "eligible_tools": [explicit_tool],
            "selection_rule": (
                f"The user requested only {explicit_tool}. Do not switch tools."
                if exclusive
                else f"The user explicitly requested {explicit_tool}. Use that tool first."
            ),
            "selection_reason": (
                "the user made it an exclusive tool constraint."
                if exclusive
                else "the user requested it."
            ),
            "tool_constraint": tool_request,
            "excluded_tools": excluded_tools,
        }

    observed_preference = observed_tool_preference(prompt, properties, legal_tools)
    if observed_preference:
        return observed_preference

    if comparison:
        measured_results = []
        for result in comparison["results"]:
            if not result["reached"] or result["tool"] not in legal_tools:
                continue
            measured_result = {
                "tool": result["tool"],
                "oracle_calls": result["oracle_calls"],
            }
            if "estimated_oracle_seconds" in result:
                measured_result["estimated_oracle_seconds"] = result[
                    "estimated_oracle_seconds"
                ]
            if "mean_final_top1" in result:
                measured_result["mean_final_top1"] = result["mean_final_top1"]
            if "mean_final_top10" in result:
                measured_result["mean_final_top10"] = result["mean_final_top10"]
            measured_results.append(measured_result)
        measured_results = order_measured_results(measured_results, evidence)
        if not measured_results:
            comparison = None

    if comparison:
        selection_rule = (
            "Results are ordered by fewest oracle calls. Prefer the first "
            "tool unless the user states another priority."
        )
        if evidence.get("property") == "qed":
            selection_rule = (
                "For QED, tools with comparable benchmark top-10 quality are "
                "ordered by reliability preference before oracle-call count. "
                "Prefer the first tool unless the user states another priority."
            )
        return {
            "request": {
                "properties": properties,
                "property_count": property_count,
                "mode": mode,
                "requested_threshold": evidence.get("requested_threshold"),
            },
            "eligible_tools": [result["tool"] for result in measured_results],
            "selection_rule": selection_rule,
            "selection_reason": (
                "benchmark evidence supports it for the requested property."
            ),
            "tool_constraint": tool_request,
            "threshold_comparison": {
                "score_threshold": comparison["score_threshold"],
                "ordered_results": measured_results,
            },
        }

    selection_rule = "Choose the tool that best matches the user's request."
    if excluded_tools:
        selection_rule = (
            "Choose the tool that best matches the request without using any "
            "excluded tool."
        )
    return {
        "request": {
            "properties": properties,
            "property_count": property_count,
            "mode": mode,
        },
        "eligible_tools": legal_tools,
        "selection_rule": selection_rule,
        "selection_reason": (
            "it best matched the request among the allowed tools."
            if excluded_tools
            else "it best matched the request."
        ),
        "tool_constraint": tool_request,
        "excluded_tools": excluded_tools,
    }


def observed_tool_preference(
    prompt: str,
    properties: list[str],
    legal_tools: list[str],
) -> dict | None:
    if "gellmo_c" not in legal_tools or len(properties) != 1:
        return None

    property_name = properties[0]
    prompt_lower = prompt.lower()
    if property_name in GELLMO_PREFERRED_SINGLE_PROPERTIES:
        selection_rule = (
            "Use gellmo_c. Local lead-optimizer smoke tests observed stronger "
            "valid, chemically plausible tox/ADMET outputs for this endpoint "
            "than the current MolOpt routing."
        )
        selection_reason = f"local tests favor it for {property_name} optimization."
    elif (
        property_name in {"logp", "qed", "sascore", "tpsa", "mol_wt"}
        and any(hint in prompt_lower for hint in CHEM_REASONABLE_HINTS)
    ):
        selection_rule = (
            "Use gellmo_c because the user emphasized chemically plausible or "
            "close analog generation; local smoke tests observed more reasonable "
            "analogs from GeLLMO than MolOpt for this style of request."
        )
        selection_reason = "it is preferred for chemically plausible analog generation."
    else:
        return None

    return {
        "request": {
            "properties": properties,
            "property_count": len(properties),
            "mode": "single_property",
        },
        "eligible_tools": ["gellmo_c"],
        "selection_rule": selection_rule,
        "selection_reason": selection_reason,
        "tool_preference_notes": [
            "Observed locally, not a benchmark guarantee.",
            "DILI is not included in the GeLLMO-preferred tox list yet because "
            "initial GeLLMO DILI tests did not meet target hits.",
            "For benchmarked physicochemical threshold optimization without "
            "a chemistry-reasonableness emphasis, keep using MolOpt evidence.",
        ],
    }


def concise_selection_reason(decision_packet: dict, tool: str) -> str:
    reason = decision_packet.get("selection_reason", "it best matched the request.")
    return f"Selected {tool} because {reason}"


def find_explicit_tool(prompt: str, legal_tools: list[str]) -> str | None:
    request = find_tool_request(prompt, legal_tools)
    return request.get("tool")


def find_tool_request(prompt: str, legal_tools: list[str]) -> dict:
    """Return explicit tool mention plus whether it is a preference or constraint."""
    normalized = re.sub(r"[_-]+", " ", prompt.lower())
    matches = []
    for tool in legal_tools:
        for alias in TOOL_ALIASES.get(tool, (tool.replace("_", " "),)):
            position = normalized.find(alias)
            if position >= 0:
                matches.append((position, -len(alias), tool, alias))
    if not matches:
        return {"tool": None, "mode": None}

    _position, _length, tool, alias = sorted(matches)[0]
    return {
        "tool": tool,
        "mode": (
            "exclusive"
            if is_exclusive_tool_request(normalized, alias)
            else "preferred"
        ),
    }


def is_exclusive_tool_request(normalized_prompt: str, alias: str) -> bool:
    phrase_pattern = "|".join(
        re.escape(phrase) for phrase in TOOL_EXCLUSIVE_PHRASES
    )
    alias_variants = [alias]
    if not alias.startswith(("gellmo", "molopt", "mol opt")):
        alias_variants.extend([f"molopt {alias}", f"mol opt {alias}"])

    return any(
        re.search(
            rf"\b(?:{phrase_pattern})\s+(?:the\s+)?{re.escape(candidate)}\b",
            normalized_prompt,
        )
        or re.search(
            rf"\buse\s+(?:the\s+)?{re.escape(candidate)}\s+only\b",
            normalized_prompt,
        )
        or re.search(
            rf"\b{re.escape(candidate)}\s+only\b",
            normalized_prompt,
        )
        for candidate in alias_variants
    )


def find_excluded_tools(prompt: str, legal_tools: list[str]) -> list[str]:
    """Return tools excluded by phrases such as 'anything other than GeLLMO'."""
    normalized = re.sub(r"[_-]+", " ", prompt.lower())
    phrase_pattern = "|".join(
        re.escape(phrase) for phrase in TOOL_EXCLUSION_PHRASES
    )
    excluded = []
    for tool in legal_tools:
        aliases = TOOL_ALIASES.get(tool, (tool.replace("_", " "),))
        if any(
            re.search(
                rf"\b(?:{phrase_pattern})\s+(?:the\s+)?{re.escape(alias)}\b",
                normalized,
            )
            for alias in aliases
        ):
            excluded.append(tool)

    request = find_tool_request(prompt, legal_tools)
    if request.get("mode") == "exclusive" and request.get("tool"):
        excluded.extend(
            tool for tool in legal_tools if tool != request["tool"]
        )
    return sorted(set(excluded))
