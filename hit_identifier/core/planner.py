import json
import re
from typing import Dict, List, Tuple

import pandas as pd


def _safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _candidate_score(row) -> float:
    vina = _safe_float(row.get("Vina Score"))
    qed = _safe_float(row.get("QED"))

    # SBDD seed ranking favors binding first, then drug-likeness. Vina is
    # negated so lower docking scores rank higher.
    score = 0.0
    if vina is not None:
        score += -vina
    else:
        score -= 1e6
    if qed is not None:
        score += qed * 10.0
    return score


def _round_or_none(value, ndigits: int = 3):
    value = _safe_float(value)
    if value is None:
        return None
    return float(round(value, ndigits))


def _action_family(action_id: str) -> str:
    action_id = str(action_id)
    if action_id.startswith("SBDD_GENERATE"):
        return "SBDD_GENERATE"
    if action_id.startswith("LBDD_GENERATE"):
        return "LBDD_GENERATE"
    if action_id.startswith("VIRTUAL_SCREEN"):
        return "VIRTUAL_SCREEN"
    return action_id


def _count_success_rows(data: pd.DataFrame, criteria: Dict) -> int:
    if data is None:
        return 0

    metric_filters = [
        ("Vina Score", "vina_lt", "lt"),
        ("QED", "qed_gt", "gt"),
        ("SA Score", "sa_lt", "lt"),
    ]
    active_filters = [
        item for item in metric_filters
        if criteria.get(item[1]) is not None
    ]
    if not active_filters:
        if "Vina Score" not in data.columns:
            return 0
        return int(data["Vina Score"].notna().sum())

    required_columns = {column for column, _, _ in active_filters}
    if not required_columns.issubset(set(data.columns)):
        return 0

    valid_mask = None
    for column, key, direction in active_filters:
        threshold = criteria[key]
        current_mask = data[column] < threshold if direction == "lt" else data[column] > threshold
        valid_mask = current_mask if valid_mask is None else (valid_mask & current_mask)
    return int(valid_mask.sum())


def build_action_history_summary(memory, criteria: Dict) -> Dict:
    summary = {}
    for item in memory.history:
        action_type = _action_family(item.get("action_id", ""))
        if not action_type:
            continue
        entry = summary.setdefault(action_type, {
            "num_runs": 0,
            "generated_molecules": 0,
            "success_count": 0,
            "best_vina": None,
            "mean_vina": None,
            "median_vina": None,
            "best_qed": None,
            "mean_qed": None,
            "median_qed": None,
            "best_sa": None,
            "mean_sa": None,
            "median_sa": None,
        })
        entry["num_runs"] += 1

        output_id = item.get("action_output", "")
        block = memory.stream.get(output_id, {})
        data = block.get("data")
        if data is None or len(data) == 0:
            continue

        entry["generated_molecules"] += int(len(data))
        entry["success_count"] += _count_success_rows(data, criteria)

        if "Vina Score" in data.columns:
            vina_values = pd.to_numeric(data["Vina Score"], errors="coerce").dropna()
            if len(vina_values) > 0:
                best_vina = _round_or_none(vina_values.min())
                if entry["best_vina"] is None or best_vina < entry["best_vina"]:
                    entry["best_vina"] = best_vina
                entry.setdefault("_vina_values", []).extend(vina_values.tolist())

        if "QED" in data.columns:
            qed_values = pd.to_numeric(data["QED"], errors="coerce").dropna()
            if len(qed_values) > 0:
                best_qed = _round_or_none(qed_values.max())
                if entry["best_qed"] is None or best_qed > entry["best_qed"]:
                    entry["best_qed"] = best_qed
                entry.setdefault("_qed_values", []).extend(qed_values.tolist())

        if "SA Score" in data.columns:
            sa_values = pd.to_numeric(data["SA Score"], errors="coerce").dropna()
            if len(sa_values) > 0:
                best_sa = _round_or_none(sa_values.min())
                if entry["best_sa"] is None or best_sa < entry["best_sa"]:
                    entry["best_sa"] = best_sa
                entry.setdefault("_sa_values", []).extend(sa_values.tolist())

    for entry in summary.values():
        if entry.get("_vina_values"):
            entry["mean_vina"] = _round_or_none(pd.Series(entry["_vina_values"]).mean())
            entry["median_vina"] = _round_or_none(pd.Series(entry["_vina_values"]).median())
        if entry.get("_qed_values"):
            entry["mean_qed"] = _round_or_none(pd.Series(entry["_qed_values"]).mean())
            entry["median_qed"] = _round_or_none(pd.Series(entry["_qed_values"]).median())
        if entry.get("_sa_values"):
            entry["mean_sa"] = _round_or_none(pd.Series(entry["_sa_values"]).mean())
            entry["median_sa"] = _round_or_none(pd.Series(entry["_sa_values"]).median())
        entry.pop("_vina_values", None)
        entry.pop("_qed_values", None)
        entry.pop("_sa_values", None)
    return summary


def _last_action_is_vs(memory) -> bool:
    if memory is None or not memory.history:
        return False
    return str(memory.history[-1].get("action_id", "")).startswith("VIRTUAL_SCREEN")


def _is_lbdd_eligible(
    candidate_option: Dict,
    memory,
    max_seed_atoms: int,
) -> bool:
    candidate_id = candidate_option["candidate_id"]
    if candidate_id in memory.get_used_seed_candidate_ids("LBDD_GENERATE"):
        return False

    source_block = memory.stream.get(candidate_option["molecule_set_id"], {})
    if not str(source_block.get("action", "")).startswith("SBDD_GENERATE"):
        return False
    if str(source_block.get("action", "")).startswith("LBDD_GENERATE"):
        return False

    mol_size = candidate_option.get("mol_size")
    if mol_size is None:
        return False
    if mol_size > max_seed_atoms:
        return False

    if candidate_option["vina"] is None:
        return False
    return True


def _reference_lbdd_option(reference_metrics: Dict) -> Dict:
    return {
        "seed_type": "reference",
        "action_type": "LBDD_GENERATE",
        "action_id": "LBDD_GENERATE000",
        "action_input": ["POCKET", "REFMOL"],
        "policy_mode": "shape_guided",
        "action_hint": "LBDD generates new molecules with shape similarity to the reference ligand and aims for good binding affinity.",
        "molecule_set_id": "",
        "candidate_id": "REFMOL",
        "smiles": "",
        "mol_size": None,
        "vina": reference_metrics["vina"],
        "qed": reference_metrics["qed"],
        "sa": reference_metrics["sa"],
        "score": 0.0,
    }


def build_generation_options(
    memory,
    criteria: Dict,
    reference_metrics: Dict,
    max_candidate_options: int = 8,
    max_lbdd_seed_atoms: int = 40,
) -> List[Dict]:
    options = []

    all_candidate_options = []
    for molecule_set_id, block in memory.stream.items():
        if block.get("type") != "MOL":
            continue
        if not str(block.get("action", "")).startswith("SBDD_GENERATE"):
            continue
        data = block.get("data")
        if data is None or len(data) == 0 or "Candidate ID" not in data.columns:
            continue
        required_columns = {"SMILES", "Vina Score", "QED", "SA Score"}
        if not required_columns.issubset(set(data.columns)):
            continue

        for _, row in data.iterrows():
            candidate_id = row["Candidate ID"]
            score = _candidate_score(row)
            mol_size = row.get("Heavy Atom Count", None)
            all_candidate_options.append({
                "seed_type": "candidate",
                "action_type": "LBDD_GENERATE",
                "action_id": "LBDD_GENERATE000",
                "action_input": [candidate_id, "POCKET"],
                "policy_mode": "shape_guided",
                "molecule_set_id": molecule_set_id,
                "candidate_id": candidate_id,
                "smiles": row.get("SMILES", ""),
                "mol_size": int(mol_size) if _safe_float(mol_size) is not None else None,
                "vina": _safe_float(row.get("Vina Score")),
                "qed": _safe_float(row.get("QED")),
                "sa": _safe_float(row.get("SA Score")),
                "score": score,
            })

    lbdd_seed_candidates = sorted(
        all_candidate_options,
        key=lambda item: (
            item["vina"] is None,
            item["vina"] if item["vina"] is not None else float("inf"),
            item["molecule_set_id"],
            item["candidate_id"],
        ),
    )

    lbdd_options = []
    for candidate_option in lbdd_seed_candidates:
        eligible = _is_lbdd_eligible(
            candidate_option,
            memory,
            max_seed_atoms=max_lbdd_seed_atoms,
        )
        if not eligible:
            continue
        lbdd_option = dict(candidate_option)
        lbdd_option["action_hint"] = "LBDD generates new molecules with shape similarity to this seed and aims for good binding affinity."
        lbdd_options.append(lbdd_option)
        if len(lbdd_options) >= max_candidate_options:
            break
    options.extend(lbdd_options)

    options.append({
        "seed_type": "reference",
        "action_type": "SBDD_GENERATE",
        "action_id": "SBDD_GENERATE000",
        "action_input": ["POCKET"],
        "policy_mode": "explore",
        "molecule_set_id": "",
        "candidate_id": "REFMOL",
        "smiles": "",
        "mol_size": None,
        "vina": reference_metrics["vina"],
        "qed": reference_metrics["qed"],
        "sa": reference_metrics["sa"],
        "score": 0.0,
    })
    if not memory.history and not reference_metrics.get("available", True):
        return options
    if reference_metrics.get("available", True):
        options.append(_reference_lbdd_option(reference_metrics))
    return options


def _compact_option(option: Dict) -> Dict:
    compact = {
        "action_type": option.get("action_type", ""),
        "policy_mode": option.get("policy_mode", ""),
        "action_hint": option.get("action_hint", ""),
    }
    if option["seed_type"] == "candidate":
        compact.update({
            "seed_id": option["candidate_id"],
            "mol_size": option["mol_size"],
            "vina": option["vina"],
            "qed": option["qed"],
            "sa": option["sa"],
            "smiles": option["smiles"],
        })
    else:
        compact.update({
            "seed_id": option.get("candidate_id", ""),
            "vina": option["vina"],
            "qed": option["qed"],
            "sa": option["sa"],
        })
    return compact


def build_action_options(
    options: List[Dict],
    target_resource: Dict,
    memory=None,
    allowed_action_types: List[str] = None,
) -> List[Dict]:
    action_options = []
    allowed_action_types = allowed_action_types or []

    def _is_allowed(action_type: str) -> bool:
        return not allowed_action_types or action_type in allowed_action_types

    sbdd_option = next((option for option in options if option.get("action_type") == "SBDD_GENERATE"), None)
    lbdd_options = [option for option in options if option.get("action_type") == "LBDD_GENERATE"]

    if sbdd_option is not None and _is_allowed("SBDD_GENERATE"):
        action_options.append({
            "action_type": "SBDD_GENERATE",
            "policy_mode": "generate",
            "num_seed_options": 0,
        })

    if lbdd_options and _is_allowed("LBDD_GENERATE"):
        best_seed = min(
            lbdd_options,
            key=lambda item: (
                item.get("vina") is None,
                item.get("vina") if item.get("vina") is not None else float("inf"),
                item.get("candidate_id", ""),
            ),
        )
        action_options.append({
            "action_type": "LBDD_GENERATE",
            "policy_mode": "shape_guided",
            "num_seed_options": len(lbdd_options),
            "best_seed_summary": {
                "seed_id": best_seed.get("candidate_id", ""),
                "vina": best_seed.get("vina"),
                "qed": best_seed.get("qed"),
                "sa": best_seed.get("sa"),
                "mol_size": best_seed.get("mol_size"),
            },
        })

    vs_config = target_resource.get("virtual_screen", {}) or {}
    if (
        vs_config.get("enabled")
        and target_resource.get("drug_sheet")
        and not _last_action_is_vs(memory)
        and _is_allowed("VIRTUAL_SCREEN")
    ):
        action_options.append({
            "action_type": "VIRTUAL_SCREEN",
            "policy_mode": "screen",
            "drug_sheet": target_resource.get("drug_sheet"),
        })
    return action_options


def build_action_prompt(
    task: Dict,
    memory,
    action_options: List[Dict],
    total_success_count: int,
    remaining_actions: int,
    initial_prompt: str,
) -> str:
    action_state = {
        "success_criteria": task["success_criteria"],
        "current_success_count": total_success_count,
        "remaining_actions": remaining_actions,
        "requested_action_types": task.get("allowed_action_types", []),
        "metric_directions": {
            "vina": "lower is better",
            "qed": "higher is better",
            "sa": "lower is better",
        },
        "action_history_summary": build_action_history_summary(memory, task["success_criteria"]),
        "legal_actions": action_options,
    }
    action_meanings = {
        "SBDD_GENERATE": "generate new molecules from the target pocket.",
        "LBDD_GENERATE": "use one seed molecule shape to generate new shape-similar molecules with good binding affinity.",
        "VIRTUAL_SCREEN": "screen known drugs against the target to find potential hits.",
    }
    legal_action_types = []
    for option in action_options:
        action_type = option.get("action_type", "")
        if action_type and action_type not in legal_action_types:
            legal_action_types.append(action_type)
    legal_action_meaning_lines = [
        f"- {action_type}: {action_meanings[action_type]}"
        for action_type in legal_action_types
        if action_type in action_meanings
    ]
    return (
        "Choose the next action from legal_actions.\n\n"
        "Legal action meanings:\n"
        f"{chr(10).join(legal_action_meaning_lines)}\n\n"
        "Rules:\n"
        "- Choose exactly one action_type from legal_actions.\n"
        "- Do not choose an action that is not listed in legal_actions.\n"
        "- Return exactly one JSON object and nothing else.\n\n"
        f"User task:\n{initial_prompt.strip()}\n\n"
        f"State:\n{json.dumps(action_state, indent=2)}\n\n"
        'JSON schema: {"action_type": "...", "reason": "..."}'
    )


def build_seed_prompt(
    task: Dict,
    seed_options: List[Dict],
    selected_action_type: str,
    initial_prompt: str,
) -> str:
    seed_state = {
        "selected_action_type": selected_action_type,
        "metric_directions": {
            "vina": "lower is better",
            "qed": "higher is better",
            "sa": "lower is better",
        },
        "eligible_seeds": [_compact_option(option) for option in seed_options],
    }
    return (
        "Choose one seed_id for the selected LBDD_GENERATE action.\n\n"
        "Rules:\n"
        "- Choose exactly one seed_id from eligible_seeds.\n"
        "- Do not choose a seed that is not listed in eligible_seeds.\n"
        "- Return exactly one JSON object and nothing else.\n\n"
        f"User task:\n{initial_prompt.strip()}\n\n"
        f"State:\n{json.dumps(seed_state, indent=2)}\n\n"
        'JSON schema: {"seed_id": "...", "reason": "..."}'
    )


def _iter_json_objects(text: str):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_policy_response(response: str, options: List[Dict] = None) -> Dict:
    objects = list(_iter_json_objects(response))
    if not objects:
        return {}
    if options is None:
        return objects[0]

    valid_keys = {
        (option.get("action_type", ""), option.get("candidate_id", ""))
        for option in options
    }
    for obj in objects:
        action_type = str(obj.get("action_type", "")).strip()
        seed_id = str(obj.get("seed_id", obj.get("candidate_id", ""))).strip()
        if (action_type, seed_id) in valid_keys:
            return obj
    return objects[0]


def choose_option_from_response(response: str, options: List[Dict]) -> Tuple[Dict, Dict]:
    parsed = parse_policy_response(response, options)
    option_by_key = {
        (option.get("action_type", ""), option.get("candidate_id", "")): option
        for option in options
    }
    requested_action_type = str(parsed.get("action_type", "")).strip()
    requested_seed_id = str(parsed.get("seed_id", parsed.get("candidate_id", ""))).strip()
    requested_mode = str(parsed.get("policy_mode", parsed.get("mode", ""))).strip().lower()
    requested_key = (requested_action_type, requested_seed_id)
    if requested_key in option_by_key:
        selected_option = option_by_key[requested_key]
        allowed_mode = selected_option.get("policy_mode", "")
        mode_mismatch = bool(requested_mode and requested_mode != allowed_mode)
        return selected_option, {
            "requested_action_type": requested_action_type,
            "requested_seed_id": requested_seed_id,
            "requested_mode": requested_mode,
            "allowed_mode": allowed_mode,
            "mode_mismatch": mode_mismatch,
            "fallback_used": False,
            "reason": parsed.get("reason", ""),
        }
    valid_choices = [
        {
            "action_type": option.get("action_type", ""),
            "seed_id": option.get("candidate_id", ""),
            "policy_mode": option.get("policy_mode", ""),
        }
        for option in options
    ]
    raise ValueError(
        "Model response did not select a valid action_type/seed_id pair. "
        f"Requested action_type={requested_action_type!r}, seed_id={requested_seed_id!r}, "
        f"policy_mode={requested_mode!r}. Valid choices: {json.dumps(valid_choices, indent=2)}"
    )


def choose_action_from_response(response: str, action_options: List[Dict]) -> Tuple[Dict, Dict]:
    parsed = parse_policy_response(response)
    requested_action_type = str(parsed.get("action_type", "")).strip()
    for option in action_options:
        if option.get("action_type") == requested_action_type:
            return option, {
                "requested_action_type": requested_action_type,
                "fallback_used": False,
                "reason": parsed.get("reason", ""),
            }
    valid_actions = [option.get("action_type", "") for option in action_options]
    raise ValueError(
        "Model response did not select a valid action_type. "
        f"Requested action_type={requested_action_type!r}. Valid actions: {json.dumps(valid_actions, indent=2)}"
    )


def choose_seed_from_response(response: str, seed_options: List[Dict]) -> Tuple[Dict, Dict]:
    parsed = parse_policy_response(response)
    requested_seed_id = str(parsed.get("seed_id", parsed.get("candidate_id", ""))).strip()
    for option in seed_options:
        if option.get("candidate_id") == requested_seed_id:
            return option, {
                "requested_seed_id": requested_seed_id,
                "fallback_used": False,
                "reason": parsed.get("reason", ""),
            }
    valid_seeds = [option.get("candidate_id", "") for option in seed_options]
    raise ValueError(
        "Model response did not select a valid LBDD seed_id. "
        f"Requested seed_id={requested_seed_id!r}. Valid seeds: {json.dumps(valid_seeds, indent=2)}"
    )
