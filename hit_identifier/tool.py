import json
import logging
import os
import shutil
import traceback
from datetime import datetime
from typing import Dict

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from .agent.agent import OpenAIAgent
from .chem.docking.sbdd_vina_docking import compute_vina_score
from .chem.evaluate import get_qed, get_sa
from .config.prompts import (
    ACTION_ALIASES,
    ALLOWED_ACTION_TYPES,
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
    PARSER_SYSTEM_PROMPT,
    build_chat_prompt,
    build_error_conclusion_prompt,
    build_generation_summary_prompt,
    build_intent_classifier_prompt,
    build_user_prompt_parser_prompt,
)
from .config.resources import load_runtime_config, build_resource_from_paths
from .core.actions import sample_lbdd_from_molset, sample_sbdd, virtual_screen_boltz
from .core.environment import run_action
from .core.errors import (
    ReferencePocketEmptyError,
    _check_reference_alignment,
    _detect_unsupported_reference,
    _diagnose_error,
)
from .core.memory import Memory
from .core.planner import (
    build_action_options,
    build_action_prompt,
    build_generation_options,
    build_seed_prompt,
    choose_action_from_response,
    choose_seed_from_response,
)
from .core.utils import get_response


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
runtime_logger = logging.getLogger("hit_identifier")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _extract_json_object(text: str) -> Dict:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"LLM prompt parser did not return a JSON object: {text}")


def _coerce_optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _parse_user_prompt_with_llm(prompt: str, agent) -> Dict:
    raw_text = prompt.strip()
    if not raw_text:
        raise ValueError("tool.generate() requires a non-empty user prompt.")

    parser_prompt = build_user_prompt_parser_prompt(raw_text)
    response, _ = get_response(
        parser_prompt,
        agent,
        sys_msg=PARSER_SYSTEM_PROMPT,
    )

    parsed = _extract_json_object(response)
    allowed_action_types = parsed.get("allowed_action_types", [])
    if allowed_action_types is None:
        allowed_action_types = []
    if not isinstance(allowed_action_types, list):
        raise ValueError(f"allowed_action_types must be a list, got: {allowed_action_types}")
    normalized_action_types = []
    invalid_actions = []
    for item in allowed_action_types:
        normalized_key = str(item).strip().upper().replace("-", "_").replace(" ", "_")
        normalized = ACTION_ALIASES.get(normalized_key) or ACTION_ALIASES.get(str(item).strip().upper())
        if normalized not in ALLOWED_ACTION_TYPES:
            invalid_actions.append(item)
            continue
        if normalized not in normalized_action_types:
            normalized_action_types.append(normalized)
    if invalid_actions:
        raise ValueError(f"Invalid allowed_action_types from LLM parser: {invalid_actions}")

    return {
        "pdb_path": str(parsed.get("pdb_path") or "").strip(),
        "sdf_path": str(parsed.get("sdf_path") or "").strip(),
        "drug_sheet_path": str(parsed.get("drug_sheet_path") or "").strip(),
        "target": str(parsed.get("target") or "").strip(),
        "min_pool_size": int(parsed.get("min_pool_size", 10) or 10) * 2,
        "max_iter": int(parsed.get("max_iter", 10) or 10),
        "target_vina": _coerce_optional_float(parsed.get("target_vina")),
        "target_qed": _coerce_optional_float(parsed.get("target_qed")),
        "target_sa": _coerce_optional_float(parsed.get("target_sa")),
        "allowed_action_types": normalized_action_types,
        "user_task": str(parsed.get("user_task") or raw_text).strip(),
    }



def _load_reference_mol(sdf_path: str):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles and "." not in smiles:
            return mol, smiles

    with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
        mol_block = f.read().strip()

    fallback_blocks = [mol_block]
    if mol_block and not mol_block.rstrip().endswith("$$$$"):
        fallback_blocks.append(mol_block + "\n$$$$\n")

    for block in fallback_blocks:
        mol = Chem.MolFromMolBlock(block, removeHs=False, strictParsing=False)
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles and "." not in smiles:
            return mol, smiles
    raise ValueError(f"Could not load a valid reference molecule from {sdf_path}")


def _build_task_from_reference(
    target_name: str,
    target_resource: Dict,
    min_pool_size: int,
    target_vina: float,
    target_qed: float,
    target_sa: float,
) -> Dict:
    reference_smiles = ""
    reference_vina = None
    reference_qed = None
    reference_sa = None
    if target_resource.get("sdf"):
        reference_mol, reference_smiles = _load_reference_mol(target_resource["sdf"])
        reference_vina, _ = compute_vina_score(reference_mol, target_resource=target_resource)
        reference_qed = get_qed(reference_smiles)
        reference_sa = get_sa(reference_smiles)
        runtime_logger.info(
            "Reference metrics for %s | Vina=%.2f QED=%.2f SA=%.2f",
            target_name,
            reference_vina,
            reference_qed,
            reference_sa,
        )
        alignment = _check_reference_alignment(target_resource, reference_mol)
        alignment["target"] = target_name
        alignment["reference_vina"] = reference_vina
        if alignment["pocket_residues"] == 0 or reference_vina == 0.0:
            runtime_logger.info(
                "Reference ligand for %s is not in the pocket | pocket_residues=%s reference_vina=%.2f 2d=%s",
                target_name, alignment["pocket_residues"], reference_vina, alignment.get("ligand_is_2d"),
            )
            raise ReferencePocketEmptyError(evidence=alignment)
    else:
        runtime_logger.info("No reference SDF configured for %s; using fixed-pocket SBDD input.", target_name)

    vina_threshold = float(target_vina) if target_vina is not None else None
    # Vina success-threshold priority:
    #   1) user-specified target_vina (highest priority);
    #   2) if the user did not specify one but a reference ligand is available,
    #      fall back to the reference ligand's Vina Score (keep molecules that
    #      bind at least as well as the reference);
    #   3) otherwise leave it unset and use best_binding selection below.
    vina_from_reference = False
    if vina_threshold is None and reference_vina is not None:
        vina_threshold = float(reference_vina)
        vina_from_reference = True
    qed_threshold = float(target_qed) if target_qed is not None else None
    sa_threshold = float(target_sa) if target_sa is not None else None
    has_metric_threshold = any(
        threshold is not None
        for threshold in [vina_threshold, qed_threshold, sa_threshold]
    )

    requirements = [
        f"Find at least {min_pool_size} promising molecules for this target.",
    ]
    if vina_threshold is not None:
        if vina_from_reference:
            requirements.append(
                f"Filter for Vina Score lower than the reference ligand's Vina Score ({vina_threshold:.2f})."
            )
        else:
            requirements.append(f"Filter for Vina Score lower than {vina_threshold:.2f}.")
    if qed_threshold is not None:
        requirements.append(f"Filter for QED higher than {qed_threshold:.2f}.")
    if sa_threshold is not None:
        requirements.append(f"Filter for SA Score lower than {sa_threshold:.2f}.")
    if not has_metric_threshold:
        requirements.append(
            "No metric threshold was specified and no reference ligand is available; "
            "select the best molecules by binding affinity, using lower Vina Score as better."
        )

    return {
        "target": target_name,
        "requirements": requirements,
        "metric_guidance": [
            "Lower Vina Score is better because stronger predicted binding corresponds to more negative values.",
            "Higher QED is better because it indicates stronger drug-like properties.",
            "Lower SA Score is better because it indicates easier synthesis.",
        ],
        "reference_metrics": {
            "vina": reference_vina,
            "qed": reference_qed,
            "sa": reference_sa,
            "available": bool(target_resource.get("sdf")),
        },
        "pocket": os.path.basename(target_resource["pdb"]),
        "drugs": [reference_smiles] if reference_smiles else [],
        "reference_sdf": target_resource.get("sdf") or "",
        "resource": 10,
        "metrics": {
            "size": "Size",
            "vina": "Vina Score",
            "qed": "QED",
            "sa": "SA Score",
        },
        "success_criteria": {
            "min_pool_size": int(min_pool_size),
            "vina_lt": vina_threshold,
            "vina_from_reference": vina_from_reference,
            "qed_gt": qed_threshold,
            "sa_lt": sa_threshold,
            "selection_mode": "threshold_filter" if has_metric_threshold else "best_binding",
            "ranking_metric": "Vina Score",
        },
    }


def _count_molecules_meeting_success_criteria(data, criteria: Dict) -> int:
    valid_mask = _build_success_mask(data, criteria)
    return int(valid_mask.sum())


def _build_success_mask(data, criteria: Dict):
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
        required_columns = ["Vina Score"]
    else:
        required_columns = [column for column, _, _ in active_filters]

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise KeyError(f"Cannot evaluate success criteria; missing columns: {missing}")

    if not active_filters:
        return data["Vina Score"].notna()

    valid_mask = None
    for column, key, direction in active_filters:
        threshold = criteria[key]
        current_mask = data[column] < threshold if direction == "lt" else data[column] > threshold
        valid_mask = current_mask if valid_mask is None else (valid_mask & current_mask)
    return valid_mask


def _count_total_molecules_meeting_success_criteria(memory, criteria: Dict) -> int:
    total = 0
    for block in memory.stream.values():
        if block.get("type") != "MOL":
            continue
        data = block.get("data")
        if data is None or len(data) == 0:
            continue
        required_columns = {"Vina Score"}
        if required_columns.issubset(set(data.columns)):
            total += _count_molecules_meeting_success_criteria(data, criteria)
    return total


def _export_successful_molecules(memory, criteria: Dict, output_dir: str):
    successful_rows = []
    successful_frames = []
    successful_root = os.path.join(output_dir, "successful_results")
    successful_sdf_dir = os.path.join(successful_root, "successful_molecules_sdf")
    os.makedirs(successful_root, exist_ok=True)
    os.makedirs(successful_sdf_dir, exist_ok=True)

    def _json_safe(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value != value:
            return None
        return value

    for mem_id, block in memory.stream.items():
        if block.get("type") != "MOL":
            continue
        data = block.get("data")
        if data is None or len(data) == 0:
            continue
        required_columns = {"Candidate ID", "SDF Path", "Vina Score", "QED", "SA Score"}
        if not required_columns.issubset(set(data.columns)):
            continue

        valid_mask = _build_success_mask(data, criteria)
        successful = data.loc[valid_mask].copy()
        if len(successful) == 0:
            continue
        successful["Molecule Set ID"] = mem_id
        successful["Source Action"] = block.get("action", "")
        successful_frames.append(successful)

    if successful_frames:
        if len(successful_frames) == 1:
            successful_all = successful_frames[0]
        else:
            import pandas as pd
            successful_all = pd.concat(successful_frames, ignore_index=True)
        if "Vina Score" in successful_all.columns:
            successful_all = successful_all.sort_values("Vina Score", ascending=True)

        for _, row in successful_all.iterrows():
            source_sdf = row["SDF Path"]
            candidate_id = row["Candidate ID"]
            exported_sdf_path = None
            if isinstance(source_sdf, str) and source_sdf and os.path.exists(source_sdf):
                target_sdf = os.path.join(successful_sdf_dir, f"{candidate_id}.sdf")
                shutil.copy2(source_sdf, target_sdf)
                exported_sdf_path = target_sdf
            row_dict = {key: _json_safe(value) for key, value in row.to_dict().items()}
            row_dict["Exported SDF Path"] = exported_sdf_path
            successful_rows.append(row_dict)

    json_path = os.path.join(successful_root, "successful_molecules.json")
    with open(json_path, "w") as f:
        json.dump(successful_rows, f, indent=2)


def _build_agent(
    model: str,
    backend: str,
    openai_api_key: str = "",
):
    backend = backend.lower()
    if backend in ("auto", "openai"):
        key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAIAgent(
            key=key,
            model=model,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _safe_load_json(path: str, default):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _summarize_run_state(run_state: Dict) -> Dict:
    if not isinstance(run_state, dict):
        return {}

    summary = {
        "success": run_state.get("success", False),
        "error_message": run_state.get("error_message", ""),
        "actions": [],
    }
    task = run_state.get("task", {})
    if task:
        summary["task"] = {
            "target": task.get("target", ""),
            "success_criteria": task.get("success_criteria", {}),
            "requirements": task.get("requirements", []),
        }

    for key in sorted([k for k in run_state.keys() if str(k).isdigit()], key=lambda x: int(x)):
        item = run_state.get(key, {})
        if not isinstance(item, dict):
            continue
        action = item.get("action", ["", []])
        success_check = item.get("success_check", {})
        summary["actions"].append({
            "round": int(key) + 1,
            "action_id": action[0] if isinstance(action, list) and action else "",
            "action_output": item.get("action_output", ""),
            "selected_option": item.get("selected_option", {}),
            "success_check": success_check,
            "action_reason": item.get("action_decision", {}).get("reason", ""),
        })
    return summary


def _summarize_molecule_library(run_dir: str) -> Dict:
    library_dir = os.path.join(run_dir, "molecule_library") if run_dir else ""
    if not os.path.isdir(library_dir):
        return {"path": library_dir, "count": 0}
    entries = [
        name for name in os.listdir(library_dir)
        if os.path.isdir(os.path.join(library_dir, name))
    ]
    return {
        "path": library_dir,
        "count": len(entries),
        "examples": sorted(entries)[:10],
    }


def _summarize_successful_results(run_dir: str) -> Dict:
    successful_dir = os.path.join(run_dir, "successful_results") if run_dir else ""
    successful_json = os.path.join(successful_dir, "successful_molecules.json")
    rows = _safe_load_json(successful_json, [])
    result = {
        "path": successful_dir,
        "json_path": successful_json,
        "count": len(rows) if isinstance(rows, list) else 0,
    }
    if isinstance(rows, list) and rows:
        import statistics
        for col, key in [("Vina Score", "vina"), ("QED", "qed"), ("SA Score", "sa")]:
            vals = [r[col] for r in rows if isinstance(r.get(col), (int, float)) and r[col] == r[col]]
            if vals:
                result.setdefault("stats", {})[key] = {
                    "min": round(min(vals), 3),
                    "max": round(max(vals), 3),
                    "median": round(statistics.median(vals), 3),
                }
    return result


class HitIdentifierAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        log_dir: str = "hit_identifier/log",
        llm_backend: str = "openai",
        runtime_config_path: str = "",
        openai_api_key: str = "",
    ):
        self.model = model
        self.log_dir = log_dir
        self.llm_backend = llm_backend
        self.runtime_config_path = runtime_config_path
        self.openai_api_key = openai_api_key
        self.last_result = None
        self.chat_history = []
        self.llm_agent = None

    def _get_llm_agent(
        self,
        model: str = None,
        backend: str = None,
        openai_api_key: str = None,
        force_new: bool = False,
    ):
        model = model or self.model
        backend = backend or self.llm_backend
        openai_api_key = openai_api_key or self.openai_api_key

        uses_default_config = (
            model == self.model
            and backend == self.llm_backend
            and openai_api_key == self.openai_api_key
        )
        if force_new or not uses_default_config:
            return _build_agent(
                model=model,
                backend=backend,
                openai_api_key=openai_api_key,
            )
        if self.llm_agent is None:
            self.llm_agent = _build_agent(
                model=self.model,
                backend=self.llm_backend,
                openai_api_key=self.openai_api_key,
            )
        return self.llm_agent

    def _collect_chat_context(self, run_dir: str = None) -> Dict:
        result = dict(self.last_result or {})
        if run_dir:
            result["run_dir"] = run_dir
            result.setdefault("run_state_dir", os.path.join(run_dir, "run_state"))
            result.setdefault("successful_results_dir", os.path.join(run_dir, "successful_results"))

        target_name = result.get("target", "")
        run_dir = result.get("run_dir", "")
        run_state_path = os.path.join(run_dir, "run_state", f"{target_name}.json") if run_dir and target_name else ""
        run_state = _safe_load_json(run_state_path, {})

        context = {
            "agent_name": "Hit Identifier",
            "method_requirements": {
                "SBDD": {
                    "task_type": "molecule_generation",
                    "required_input_files": ["pdb_path in prompt"],
                    "not_required": ["sdf_path", "drug_sheet_path"],
                    "notes": "Uses the target pocket directly to generate molecules.",
                },
                "LBDD": {
                    "task_type": "molecule_generation",
                    "required_input_files": ["pdb_path in prompt", "ligand seed"],
                    "seed_sources": [
                        "sdf_path in prompt if provided",
                        "previous SBDD-generated/evaluated molecules only",
                    ],
                    "not_required": ["drug_sheet_path"],
                    "notes": "Can start from a provided reference ligand or from SBDD outputs. Virtual-screening hits are not used as LBDD seeds.",
                },
                "VIRTUAL_SCREEN": {
                    "task_type": "virtual_screening_not_generation",
                    "required_input_files": ["pdb_path in prompt", "drug_sheet_path in prompt"],
                    "not_required": ["sdf_path"],
                    "notes": "Screens known drugs; it does not generate new molecules.",
                },
            },
            "last_result": result,
            "last_run": _summarize_run_state(run_state),
            "molecule_library": _summarize_molecule_library(run_dir),
            "successful_results": _summarize_successful_results(run_dir),
            "chat_history": self.chat_history[-6:],
        }
        return context

    def _classify_intent(self, prompt: str, agent) -> str:
        classifier_prompt = build_intent_classifier_prompt(prompt)
        response, _ = get_response(
            classifier_prompt,
            agent,
            sys_msg=INTENT_CLASSIFIER_SYSTEM_PROMPT,
        )
        parsed = _extract_json_object(response)
        intent = str(parsed.get("intent", "chat")).strip().lower()
        return intent if intent in ("generate", "chat") else "chat"

    def run(self, prompt: str, **kwargs):
        agent = self._get_llm_agent(force_new=True)
        intent = self._classify_intent(prompt, agent)
        if intent == "generate":
            return self.generate(prompt, agent=agent, **kwargs)
        return self.chat(prompt, agent=agent)

    def chat(self, message: str, run_dir: str = None, agent=None):
        context = self._collect_chat_context(run_dir=run_dir)
        if agent is None:
            agent = self._get_llm_agent()
        chat_prompt = build_chat_prompt(context, message)
        answer, _ = get_response(chat_prompt, agent)

        self.chat_history.append({"user": message, "assistant": answer})
        return answer, {"responses": agent.responses}

    def _summarize_generation_with_llm(self, agent) -> str:
        context = self._collect_chat_context()
        summary_prompt = build_generation_summary_prompt(context)
        answer, _ = get_response(summary_prompt, agent)
        return answer

    def _conclude_error_with_llm(self, diagnosis: Dict, agent) -> str:
        conclusion_prompt = build_error_conclusion_prompt(diagnosis)
        answer, _ = get_response(conclusion_prompt, agent)
        return answer

    def _build_action_dicts(self):
        return [
            {
                "action_id": "SBDD_GENERATE",
                "desc": "Generate new molecules from the target pocket.",
                "func": sample_sbdd,
                "cost": "1",
            },
            {
                "action_id": "LBDD_GENERATE",
                "desc": "Use one seed molecule shape to generate new shape-similar molecules with good binding affinity.",
                "func": sample_lbdd_from_molset,
                "cost": "1",
            },
            {
                "action_id": "VIRTUAL_SCREEN",
                "desc": "Screen known drugs against the target to find potential hits.",
                "func": virtual_screen_boltz,
                "cost": "1",
            },
        ]

    def _initialize_memory(self, task, target_resource):
        memory = Memory()
        memory.add_pocket(
            filename=task["pocket"],
            pdb=target_resource["pdb"],
            sdf=target_resource.get("sdf") or "",
            lbdd=target_resource["lbdd"],
            sbdd=target_resource["sbdd"],
            desc=(
                f"This is a target resource bundle for {task['target']}. It contains "
                "the pdb pocket, an optional reference sdf ligand, and method-specific "
                "configs for both LBDD and SBDD generation."
            ),
        )
        for action in self._build_action_dicts():
            memory.add_action(**action)
        return memory

    def _prepare_task(
        self,
        prompt: str,
        agent,
        max_iter: int,
        runtime_config_path: str,
    ):
        runtime_config = load_runtime_config(runtime_config_path)
        prompt_config = _parse_user_prompt_with_llm(prompt, agent)
        if max_iter is None:
            max_iter = prompt_config["max_iter"]

        pdb_path = prompt_config["pdb_path"]
        if not pdb_path:
            raise ValueError(
                "A .pdb file path must be specified in the prompt, "
                "e.g. 'Find hits for /path/to/EGFR.pdb with Vina < -10'."
            )
        sdf_path = prompt_config["sdf_path"]
        drug_sheet_path = prompt_config["drug_sheet_path"]
        bad_reference = _detect_unsupported_reference(prompt, sdf_path, pdb_path, drug_sheet_path)
        if bad_reference:
            raise ValueError(f"Could not load a valid reference molecule from {bad_reference}")
        target_name = prompt_config["target"] or os.path.splitext(os.path.basename(pdb_path))[0]

        target_resource = build_resource_from_paths(
            target_name=target_name,
            pdb_path=pdb_path,
            sdf_path=sdf_path,
            drug_sheet_path=drug_sheet_path,
            runtime_config=runtime_config,
        )
        if not target_resource["pdb"] or not os.path.exists(target_resource["pdb"]):
            raise FileNotFoundError(
                f"[Errno 2] No such file or directory: '{target_resource['pdb']}'"
            )
        if target_resource.get("sdf") and not os.path.exists(target_resource["sdf"]):
            raise FileNotFoundError(
                f"[Errno 2] No such file or directory: '{target_resource['sdf']}'"
            )

        # Fail fast if no action can run at the start (mirrors planner.build_action_options
        # for round 1): SBDD always works from the pocket; LBDD needs a seed (a reference
        # ligand, since there are no prior molecules yet); VS needs an enabled drug sheet.
        allowed = prompt_config["allowed_action_types"]
        def _action_allowed(action_type):
            return (not allowed) or (action_type in allowed)
        vs_config = target_resource.get("virtual_screen", {}) or {}
        sbdd_ok = _action_allowed("SBDD_GENERATE")
        lbdd_ok = _action_allowed("LBDD_GENERATE") and bool(target_resource.get("sdf"))
        vs_ok = (
            _action_allowed("VIRTUAL_SCREEN")
            and bool(target_resource.get("drug_sheet"))
            and bool(vs_config.get("enabled"))
        )
        if not (sbdd_ok or lbdd_ok or vs_ok):
            error = ValueError(
                "No legal actions are available after applying the prompt action list. "
                f"Requested action types: {allowed or 'not specified'}."
            )
            error.hi_target = target_name
            error.hi_allowed = allowed
            raise error

        task = _build_task_from_reference(
            target_name,
            target_resource,
            min_pool_size=prompt_config["min_pool_size"],
            target_vina=prompt_config["target_vina"],
            target_qed=prompt_config["target_qed"],
            target_sa=prompt_config["target_sa"],
        )
        task["resource"] = max_iter
        task["allowed_action_types"] = prompt_config["allowed_action_types"]
        return target_resource, task, prompt_config["user_task"], max_iter

    def _save_run_state(self, logger, task, run_state_dir):
        os.makedirs(run_state_dir, exist_ok=True)
        with open(os.path.join(run_state_dir, f"{task['target']}.json"), "w") as file:
            json.dump(logger, file, indent=4)

    def generate(self, prompt: str, max_iter: int = None, agent=None, **overrides):
        config = {
            "model": self.model,
            "log_dir": self.log_dir,
            "llm_backend": self.llm_backend,
            "runtime_config_path": self.runtime_config_path,
            "openai_api_key": self.openai_api_key,
        }
        config.update(overrides)

        if agent is None:
            agent = self._get_llm_agent(
                model=config["model"],
                backend=config["llm_backend"],
                openai_api_key=config["openai_api_key"],
            )
        try:
            target_resource, task, initial_prompt, max_iter = self._prepare_task(
                prompt=prompt,
                agent=agent,
                max_iter=max_iter,
                runtime_config_path=config["runtime_config_path"],
            )
        except Exception as exc:
            diagnosis = _diagnose_error(exc, stage="prepare")
            runtime_logger.info("[Done] Task preparation failed (%s); returning error conclusion.",
                                diagnosis["error_type"])
            return (
                self._conclude_error_with_llm(diagnosis, agent),
                {"responses": agent.responses},
            )

        os.makedirs(config["log_dir"], exist_ok=True)
        run_id = str(datetime.now().strftime("%y-%m-%d-%H-%M-%S")) + f"_{task['target']}"
        path_to_log = os.path.join(config["log_dir"], run_id)
        os.makedirs(path_to_log, exist_ok=True)

        memory = self._initialize_memory(task, target_resource)
        logger = {"model": config["model"]}
        resource = task["resource"]
        caught_exception = None
        runtime_logger.info(
            "User requirement | target=%s max_iter=%s allowed_actions=%s criteria=%s task=%s",
            task["target"],
            max_iter,
            task.get("allowed_action_types", []) or "any",
            json.dumps(task["success_criteria"], sort_keys=True),
            initial_prompt,
        )
        criteria = task["success_criteria"]
        criteria_parts = [f"min_pool={criteria['min_pool_size']}"]
        if criteria.get("vina_lt") is not None:
            criteria_parts.append(f"Vina<{criteria['vina_lt']}")
        if criteria.get("qed_gt") is not None:
            criteria_parts.append(f"QED>{criteria['qed_gt']}")
        if criteria.get("sa_lt") is not None:
            criteria_parts.append(f"SA<{criteria['sa_lt']}")
        methods_str = ", ".join(task.get("allowed_action_types") or []) or "any"
        runtime_logger.info("target=%s  pdb=%s", task['target'], target_resource['pdb'])
        runtime_logger.info("methods=%s  criteria: %s", methods_str, ' | '.join(criteria_parts))

        try:
            for n_iter in range(max_iter):
                logger[n_iter] = {}
                total_success_count = _count_total_molecules_meeting_success_criteria(
                    memory,
                    task["success_criteria"],
                )
                seed_options = build_generation_options(
                    memory=memory,
                    criteria=task["success_criteria"],
                    reference_metrics=task["reference_metrics"],
                )
                action_options = build_action_options(
                    options=seed_options,
                    target_resource=target_resource,
                    memory=memory,
                    allowed_action_types=task.get("allowed_action_types", []),
                )
                if not action_options:
                    requested_actions = task.get("allowed_action_types", [])
                    raise ValueError(
                        "No legal actions are available after applying the prompt action list. "
                        f"Requested action types: {requested_actions or 'not specified'}."
                    )
                input_prompt = build_action_prompt(
                    task=task,
                    memory=memory,
                    action_options=action_options,
                    total_success_count=total_success_count,
                    remaining_actions=resource,
                    initial_prompt=initial_prompt,
                )
                logger[n_iter]["input_prompt"] = input_prompt
                logger[n_iter]["seed_options"] = seed_options
                logger[n_iter]["action_options"] = action_options

                _iter_resp_start = len(agent.responses)
                runtime_logger.debug("action prompt:\n%s", input_prompt)
                response, _ = get_response(input_prompt, agent)
                logger[n_iter]["response"] = response
                runtime_logger.debug("model response:\n%s", response)

                try:
                    selected_action, action_decision = choose_action_from_response(response, action_options)
                except ValueError as exc:
                    runtime_logger.warning("action selection failed (%s); retrying", exc)
                    allowed_action_types = [
                        option.get("action_type", "")
                        for option in action_options
                        if option.get("action_type", "")
                    ]
                    retry_prompt = (
                        f"{input_prompt}\n\n"
                        f"Your previous response was invalid: {exc}\n"
                        f"Allowed action_type values are exactly: {allowed_action_types}. "
                        "Choose again using one of these values only. "
                        "Return exactly one JSON object."
                    )
                    logger[n_iter]["action_retry_prompt"] = retry_prompt
                    runtime_logger.debug("action retry prompt:\n%s", retry_prompt)
                    retry_response, _ = get_response(retry_prompt, agent)
                    logger[n_iter]["action_retry_response"] = retry_response
                    runtime_logger.debug("model response:\n%s", retry_response)
                    selected_action, action_decision = choose_action_from_response(retry_response, action_options)
                    action_decision["retry_used"] = True
                logger[n_iter]["action_decision"] = action_decision

                if selected_action["action_type"] == "LBDD_GENERATE":
                    lbdd_seed_options = [
                        option for option in seed_options
                        if option.get("action_type") == "LBDD_GENERATE"
                    ]
                    seed_prompt = build_seed_prompt(
                        task=task,
                        seed_options=lbdd_seed_options,
                        selected_action_type=selected_action["action_type"],
                        initial_prompt=initial_prompt,
                    )
                    logger[n_iter]["seed_prompt"] = seed_prompt
                    runtime_logger.debug("seed prompt:\n%s", seed_prompt)
                    seed_response, _ = get_response(seed_prompt, agent)
                    logger[n_iter]["seed_response"] = seed_response
                    runtime_logger.debug("model response:\n%s", seed_response)
                    try:
                        selected_option, seed_decision = choose_seed_from_response(seed_response, lbdd_seed_options)
                    except ValueError as exc:
                        runtime_logger.warning("seed selection failed (%s); retrying", exc)
                        retry_prompt = (
                            f"{seed_prompt}\n\n"
                            f"Your previous response was invalid: {exc}\n"
                            "Choose again using only a seed_id from eligible_seeds. "
                            "Return exactly one JSON object."
                        )
                        logger[n_iter]["seed_retry_prompt"] = retry_prompt
                        runtime_logger.debug("seed retry prompt:\n%s", retry_prompt)
                        retry_response, _ = get_response(retry_prompt, agent)
                        logger[n_iter]["seed_retry_response"] = retry_response
                        runtime_logger.debug("model response:\n%s", retry_response)
                        selected_option, seed_decision = choose_seed_from_response(retry_response, lbdd_seed_options)
                        seed_decision["retry_used"] = True
                    logger[n_iter]["seed_decision"] = seed_decision
                elif selected_action["action_type"] == "SBDD_GENERATE":
                    selected_option = next(
                        option for option in seed_options
                        if option.get("action_type") == "SBDD_GENERATE"
                    )
                elif selected_action["action_type"] == "VIRTUAL_SCREEN":
                    selected_option = {
                        "seed_type": "drug_sheet",
                        "action_type": "VIRTUAL_SCREEN",
                        "action_id": "VIRTUAL_SCREEN000",
                        "action_input": ["POCKET"],
                        "policy_mode": "screen",
                        "molecule_set_id": "",
                        "candidate_id": "DRUG_SHEET",
                        "action_hint": selected_action.get("action_hint", ""),
                    }
                else:
                    raise ValueError(f"Unsupported selected action type: {selected_action['action_type']}")

                action_id = selected_option["action_id"]
                action_input = selected_option["action_input"]
                logger[n_iter]["policy_decision"] = action_decision
                logger[n_iter]["selected_option"] = selected_option
                logger[n_iter]["action"] = (action_id, action_input)

                action_output, cost, metadata = run_action(
                    action_id,
                    action_input,
                    memory=memory,
                    agent=agent,
                    metrics=task["metrics"],
                    target_pdb=target_resource["pdb"],
                    target_resource=target_resource,
                    drugs=task["drugs"],
                    log_dir=path_to_log,
                    round_index=n_iter + 1,
                )
                iter_responses = agent.responses[_iter_resp_start:]
                metadata["responses"] = iter_responses
                logger[n_iter]["llm_responses"] = [r.model_dump() for r in iter_responses]
                if metadata.get("screened_drug_ids"):
                    action_input = ["POCKET"] + metadata["screened_drug_ids"]
                logger[n_iter]["action_output"] = action_output
                logger[n_iter]["action"] = (action_id, action_input)
                resource -= cost
                memory.add_history(
                    action_id=action_id,
                    action_input=action_input,
                    action_output=action_output,
                    metadata=metadata,
                )
                runtime_logger.info(
                    "Round %d/%d: action=%s input=%s output=%s",
                    n_iter + 1, max_iter, action_id, action_input,
                    action_output,
                )

                if action_output in memory.stream and memory.stream[action_output].get("type") == "MOL":
                    metrics_summary = memory.stream[action_output]["metrics"]
                    vina_metrics = metrics_summary.get("vina", {})
                    current_data = memory.stream[action_output]["data"]
                    current_meeting_count = _count_molecules_meeting_success_criteria(
                        current_data,
                        task["success_criteria"],
                    )
                    total_meeting_count = _count_total_molecules_meeting_success_criteria(
                        memory,
                        task["success_criteria"],
                    )
                    logger[n_iter]["success_check"] = {
                        "current_set_meeting_count": current_meeting_count,
                        "total_meeting_count": total_meeting_count,
                        "min_pool_size": task["success_criteria"]["min_pool_size"],
                        "criteria": task["success_criteria"],
                        "metrics": metrics_summary,
                    }
                    runtime_logger.info(
                        "Round %d/%d: %d molecules | best Vina this round=%.2f | cumulative success %d/%d",
                        n_iter + 1, max_iter, len(current_data),
                        vina_metrics.get('min', float('nan')),
                        total_meeting_count, task['success_criteria']['min_pool_size'],
                    )
                    if total_meeting_count >= task["success_criteria"]["min_pool_size"]:
                        logger["success"] = True
                        break
                elif action_output == "EMPTY SET":
                    runtime_logger.info("Round %d/%d: No molecules generated", n_iter + 1, max_iter)

                if resource <= 0:
                    logger["success"] = False
                    break

            if "success" not in logger:
                logger["success"] = False
        except Exception as exc:
            logger["success"] = False
            logger["error_message"] = traceback.format_exc() + str(exc)
            caught_exception = exc

        diagnosis = None
        if caught_exception is not None:
            diagnosis = _diagnose_error(
                caught_exception, task=task, target_resource=target_resource, stage="generate"
            )

        final_success_count = _count_total_molecules_meeting_success_criteria(memory, task["success_criteria"])
        if logger.get("success"):
            runtime_logger.info("[Done] Success — %d molecules found (goal: %d)", final_success_count, task['success_criteria']['min_pool_size'])
        elif diagnosis is not None:
            runtime_logger.info("[Done] Run failed (%s) — %d molecules found (goal: %d)", diagnosis["error_type"], final_success_count, task['success_criteria']['min_pool_size'])
        else:
            runtime_logger.info("[Done] Criteria not met — %d molecules found (goal: %d)", final_success_count, task['success_criteria']['min_pool_size'])
        if diagnosis is not None:
            # Error caught after the run dir was created: drop the empty dir and return
            # the conclusion, so error runs leave no Saving/Run-dir artifacts (matches CP1).
            shutil.rmtree(path_to_log, ignore_errors=True)
            return (
                self._conclude_error_with_llm(diagnosis, agent),
                {"responses": agent.responses},
            )

        runtime_logger.info("[Done] Saving all %d successful molecules to: %s", final_success_count, os.path.join(path_to_log, 'successful_results'))
        runtime_logger.info("[Done] Run dir: %s", path_to_log)

        logger["task"] = task
        run_state_dir = os.path.join(path_to_log, "run_state")
        self._save_run_state(logger, task, run_state_dir)
        _export_successful_molecules(
            memory=memory,
            criteria=task["success_criteria"],
            output_dir=path_to_log,
        )
        result = {
            "success": logger.get("success", False),
            "target": task["target"],
            "run_dir": path_to_log,
            "run_state_dir": run_state_dir,
            "successful_results_dir": os.path.join(path_to_log, "successful_results"),
        }
        self.last_result = result
        return self._summarize_generation_with_llm(agent), {"responses": agent.responses}


if __name__ == "__main__":
    agent = HitIdentifierAgent()

    # chat_examples = [
    #     "Can you do virtual screening? What else can you do?",
    #     "Can you run SBDD and LBDD for dataset/pdbs/CDK5.pdb?",
    #     "What input files do you need before generating molecules?",
    # ]
    # for message in chat_examples:
    #     print(f"\nUser: {message}")
    #     print(f"Response: {agent.run(message)}")


    # error_list = [
    #     # (1) no_legal_actions: LBDD requested but no reference ligand provided
    #     "Find at least 20 hits for dataset/pdbs/ADRB1.pdb with Vina lower than -8 within 5 rounds. Action list: LBDD.",
    #     # (2a) reference_not_in_pocket — 3D ligand but far from the pocket (misaligned)
    #     # (2b) reference_not_in_pocket — 2D ligand (no 3D conformer), not in the pocket
    #     # (3) reference_wrong_format: reference is .smi, not SDF
    #     # (4) pdb_not_found: pdb path does not exist
    #     "Find at least 20 hits for dataset/pdbs/DOES_NOT_EXIST.pdb with Vina lower than -8 within 5 rounds. Action list: SBDD.",
    #     # (5) reference_sdf_corrupt: .sdf extension but invalid content
    #     # (6) sdf_not_found: reference .sdf path does not exist
    # ]

    # for prompt in error_list:
    #     response, metadata = agent.run(prompt)
    #     print(f"\nResponse:\n{response}")


    response, metadata = agent.run(
        "Find at least 200 potential hits for dataset/pdbs/PSEN1.pdb within 3 rounds. Reference ligand is dataset/sdfs/EGFR.pdb__AQ4.sdf. Use SBDD and LBDD only."
    )
    print(f"\nResponse:\n{response}")

