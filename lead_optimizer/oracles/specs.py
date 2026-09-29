import re

from ..schemas import OptimizationObjective


ORACLE_ALIASES = {
    "drd2": ("drd2", "drd2 inhibition"),
    "qed": ("qed", "drug likeness", "drug-likeness", "drug like", "drug-like"),
    "sascore": (
        "sascore",
        "sa score",
        "synthetic accessibility",
        "easier to synthesize",
        "easy to synthesize",
        "less synthetically difficult",
    ),
    "mol_wt": ("mol_wt", "mol wt", "molecular weight", "molecular mass"),
    "logp": ("logp", "log p", "penalized logp", "penalized log p"),
    "tpsa": ("tpsa", "polar surface area"),
    "hbd": ("hbd", "hydrogen bond donors", "h bond donors"),
    "hba": ("hba", "hydrogen bond acceptors", "h bond acceptors"),
    "rot_bonds": ("rot_bonds", "rotatable bonds"),
    "herg": ("herg", "hERG", "cardiotoxicity", "cardiotoxic"),
    "dili": ("dili",),
    "clintox": ("clintox", "clinical toxicity"),
    "mutagenicity": ("mutagenicity",),
    "carcinogens": ("carcinogens", "carcinogenicity"),
}


INCREASE_WORDS = (
    "increase",
    "increasing",
    "maximize",
    "maximise",
    "maximizing",
    "maximising",
    "raise",
    "raising",
    "higher",
    "above",
    "over",
    "greater",
    "no less than",
    "at least",
)


DECREASE_WORDS = (
    "decrease",
    "decreasing",
    "minimize",
    "minimise",
    "minimizing",
    "minimising",
    "reduce",
    "reducing",
    "lower",
    "lowering",
    "easier to synthesize",
    "easy to synthesize",
    "below",
    "under",
    "less",
    "no more than",
    "no greater than",
    "at most",
)


IMPROVE_WORDS = ("improve", "improving", "optimize", "optimizing")
EXPLICIT_INCREASE_WORDS = tuple(
    word for word in INCREASE_WORDS if word not in IMPROVE_WORDS
)


DEFAULT_IMPROVEMENT_OBJECTIVES = {
    "qed": {"direction": "increase"},
    "sascore": {"direction": "decrease"},
    "herg": {"direction": "decrease"},
    "dili": {"direction": "decrease"},
    "clintox": {"direction": "decrease"},
    "mutagenicity": {"direction": "decrease"},
    "carcinogens": {"direction": "decrease"},
    "tpsa": {"direction": "target_range", "minimum": 60.0, "maximum": 100.0},
    "mol_wt": {"direction": "target_range", "minimum": 250.0, "maximum": 450.0},
    "logp": {"direction": "target_range", "minimum": 1.0, "maximum": 4.0},
    "hbd": {"direction": "target_range", "minimum": 0.0, "maximum": 3.0},
    "hba": {"direction": "target_range", "minimum": 2.0, "maximum": 8.0},
    "rot_bonds": {"direction": "target_range", "minimum": 0.0, "maximum": 8.0},
}


TARGET_RANGE_PROPERTIES = {
    name
    for name, objective in DEFAULT_IMPROVEMENT_OBJECTIVES.items()
    if objective["direction"] == "target_range"
}


def parse_optimization_objective(
    text: str,
    properties=(),
    default_property: str | None = None,
) -> OptimizationObjective | None:
    """Parse one property, direction, and optional threshold/range from text."""
    property_name = default_property or find_property_name(text, properties)
    if property_name is None:
        return None

    return parse_objective_for_property(text, property_name)


def parse_optimization_objectives(
    text: str,
    properties=(),
) -> list[OptimizationObjective]:
    """Parse one objective per property mentioned in the request."""
    property_names = find_property_names(text, properties)
    objectives = [
        objective
        for property_name in property_names
        if (objective := parse_objective_for_property(text, property_name))
        is not None
    ]
    return list({objective.property: objective for objective in objectives}.values())


def prompt_objective_fields(text: str) -> dict:
    """Return parsed properties/objectives plus any explicit scoring objective."""
    properties = find_property_names(text)
    objectives = parse_optimization_objectives(text)
    if not objectives:
        objective = parse_optimization_objective(text)
        objectives = [objective] if objective is not None else []
    mode = "multi_property" if len(properties) > 1 else "single_property"
    objective = objectives[0] if mode == "single_property" and objectives else None
    if mode == "multi_property":
        primary = find_primary_property(text)
        objective = next(
            (item for item in objectives if item.property == primary),
            None,
        )
    return {
        "requested_properties": properties,
        "objective_mode": mode,
        "objectives": objectives,
        "objective": objective,
    }


def find_primary_property(text: str, properties=()) -> str | None:
    """Return a property only when the user explicitly marks it as primary."""
    property_names = find_property_names(text, properties)
    if not property_names:
        return None

    normalized = normalize_text(text)
    priority_patterns = [
        r"(?:primary|main|top|first|highest priority)\s+(?:objective|goal|target|property)?\s*(?:is|:)?\s*{alias}",
        r"(?:prioritize|prioritise|prefer|rank by|score by|select by)\s+{alias}",
        r"{alias}\s+(?:is|as)?\s*(?:the\s+)?(?:primary|main|top|first|highest priority)",
    ]
    for property_name in property_names:
        aliases = ORACLE_ALIASES.get(property_name, (property_name,))
        for alias in aliases:
            alias_pattern = re.escape(normalize_text(alias))
            for pattern in priority_patterns:
                if re.search(
                    pattern.format(alias=alias_pattern),
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    return property_name
    return None


def parse_objective_for_property(
    text: str,
    property_name: str,
) -> OptimizationObjective | None:
    """Parse direction and numeric target for a specific property."""
    range_match = find_range(text, property_name)
    if range_match:
        return OptimizationObjective(
            property=property_name,
            direction="target_range",
            minimum=range_match[0],
            maximum=range_match[1],
        )

    target_value = find_single_value_target(text, property_name)
    if target_value is not None:
        return OptimizationObjective(
            property=property_name,
            direction="target_range",
            minimum=target_value,
            maximum=target_value,
        )

    improvement_objective = default_improvement_objective(text, property_name)
    threshold = find_threshold(text, property_name)
    if threshold is not None:
        if (
            improvement_objective is not None
            and improvement_objective.direction != "target_range"
        ):
            direction = improvement_objective.direction
        else:
            direction = find_direction(text, property_name)
        return OptimizationObjective(
            property=property_name,
            direction=direction,
            threshold=threshold,
        )

    if improvement_objective is not None:
        return improvement_objective

    return OptimizationObjective(
        property=property_name,
        direction=find_direction(text, property_name),
        threshold=None,
    )


def find_range(text: str, oracle_name: str | None = None) -> tuple[float, float] | None:
    window = relevant_text_window(text, oracle_name)
    number = r"(-?\d+(?:\.\d+)?)"
    range_patterns = [
        rf"(?:between|from)\s*{number}\s*(?:and|to|-)\s*{number}",
        rf"(?:range|window)\D{{0,30}}{number}\s*(?:and|to|-)\s*{number}",
        rf"{number}\s*(?:and|to|-)\s*{number}\s*(?:range|window)",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, window, flags=re.IGNORECASE)
        if match:
            first = float(match.group(1))
            second = float(match.group(2))
            return (min(first, second), max(first, second))
    return None


def find_single_value_target(
    text: str,
    oracle_name: str | None = None,
) -> float | None:
    if oracle_name not in TARGET_RANGE_PROPERTIES:
        return None

    window = relevant_text_window(text, oracle_name)
    number = r"(-?\d+(?:\.\d+)?)"
    target_patterns = [
        rf"(?:target(?:ing)?|aim(?:ing)? for|set|keep(?:ing)?|land(?:s|ing)?)"
        rf"[^.;,]{{0,80}}\b(?:at|to|near|around)\s*{number}",
        rf"\b{number}\s*(?:target|goal)\b",
    ]
    for pattern in target_patterns:
        match = re.search(pattern, window, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def find_property_name(text: str, properties=()) -> str | None:
    """Return the first known property mentioned in the request."""
    names = find_property_names(text, properties)
    return names[0] if names else None


def format_objective(objective) -> str:
    if objective.direction == "target_range":
        return (
            f"{objective.property} target range "
            f"{objective.minimum:g}-{objective.maximum:g}"
        )
    if objective.threshold is not None:
        return f"{objective.property} {objective.direction} {objective.threshold:g}"
    return f"{objective.property} {objective.direction}"


def format_objectives(objectives: list) -> str:
    return "; ".join(format_objective(objective) for objective in objectives)


def find_property_names(text: str, properties=()) -> list[str]:
    """Return distinct known properties in the order they appear."""
    normalized = normalize_text(text)
    matches = []
    aliases_by_name = dict(ORACLE_ALIASES)
    for name in properties:
        aliases_by_name.setdefault(name, (name, name.replace("_", " ")))

    for name, aliases in aliases_by_name.items():
        for alias in aliases:
            position = phrase_position(normalized, alias)
            if position is not None:
                matches.append((position, name))
                break
    if not matches:
        return []
    ordered_names = []
    for _position, name in sorted(matches):
        if name not in ordered_names:
            ordered_names.append(name)
    return ordered_names


def find_direction(text: str, oracle_name: str | None = None) -> str:
    window = relevant_text_window(text, oracle_name)
    normalized = normalize_text(window)
    decrease_position = first_position(normalized, DECREASE_WORDS)
    increase_position = first_position(normalized, INCREASE_WORDS)

    if decrease_position is None:
        return "increase"
    if increase_position is None:
        return "decrease"
    return "decrease" if decrease_position < increase_position else "increase"


def default_improvement_objective(
    text: str,
    property_name: str,
) -> OptimizationObjective | None:
    window = relevant_text_window(text, property_name)
    normalized = normalize_text(window)
    if (
        first_position(normalized, IMPROVE_WORDS) is None
        and first_position(normalize_text(text), IMPROVE_WORDS) is None
    ):
        return None
    if first_position(normalized, DECREASE_WORDS + EXPLICIT_INCREASE_WORDS) is not None:
        return None

    default = DEFAULT_IMPROVEMENT_OBJECTIVES.get(property_name)
    if default is None:
        return None
    return OptimizationObjective(property=property_name, **default)


def find_threshold(text: str, oracle_name: str | None = None) -> float | None:
    window = relevant_text_window(text, oracle_name)

    tagged = re.search(
        r"<THRESHOLD>\s*(-?\d+(?:\.\d+)?)\s*</THRESHOLD>",
        window,
        flags=re.IGNORECASE,
    )
    if tagged:
        return float(tagged.group(1))

    threshold_patterns = [
        r"(?:above|over|below|under|greater than|less than|at least|at most|no more than|no less than|no greater than|at or below|at or above|up to|at|to|target(?:ing)?|threshold(?: of)?)\s*(-?\d+(?:\.\d+)?)",
        r"(?:>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in threshold_patterns:
        match = re.search(pattern, window, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def relevant_text_window(text: str, oracle_name: str | None) -> str:
    if not oracle_name:
        return text

    normalized = normalize_text(text)
    spans = alias_spans(normalized, oracle_name)
    if not spans:
        return text

    position, _alias_end = min(spans)
    other_spans = [
        span
        for name in ORACLE_ALIASES
        if name != oracle_name
        for span in alias_spans(normalized, name)
    ]
    previous_ends = [end for start, end in other_spans if start < position]
    next_starts = [start for start, _end in other_spans if start > position]

    start = max(0, position - 80)
    if previous_ends:
        start = max(start, max(previous_ends))
    delimiter = max(
        normalized.rfind(mark, start, position)
        for mark in (",", ";", ".")
    )
    if delimiter >= 0:
        start = delimiter + 1

    # Orchestrator prompts can include long absolute CSV paths between the
    # property and threshold language, e.g. "SAScore of molecules in /... to
    # below 1.5". Keep enough right-side context to preserve that threshold.
    end = min(len(normalized), position + 300)
    if next_starts:
        end = min(end, min(next_starts))
    window = normalized[start:end]
    direction_words = sorted(
        set(DECREASE_WORDS + INCREASE_WORDS + IMPROVE_WORDS),
        key=len,
        reverse=True,
    )
    direction_pattern = "|".join(re.escape(word) for word in direction_words)
    # In prompts like "improve QED and reduce SAScore", the QED window ends
    # before "SAScore" but may still include the connector plus the next
    # property's verb ("and reduce"). Trim that handoff so each property keeps
    # its own local direction.
    return re.sub(
        rf"\b(?:and|while|but|then|,)\s+(?:{direction_pattern})\s*$",
        "",
        window,
        flags=re.IGNORECASE,
    ).strip()


def alias_spans(text: str, oracle_name: str) -> list[tuple[int, int]]:
    aliases = ORACLE_ALIASES.get(oracle_name, (oracle_name,))
    spans = []
    for alias in aliases:
        phrase = normalize_text(alias)
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        match = re.search(pattern, text)
        if match:
            spans.append((match.start(), match.end()))
    return spans


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    return phrase_position(text, phrase) is not None


def phrase_position(text: str, phrase: str) -> int | None:
    pattern = r"(?<![a-z0-9])" + re.escape(normalize_text(phrase)) + r"(?![a-z0-9])"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.start()


def first_position(text: str, words: tuple[str, ...]) -> int | None:
    positions = []
    for word in words:
        position = text.find(word)
        if position >= 0:
            positions.append(position)
    if not positions:
        return None
    return min(positions)
