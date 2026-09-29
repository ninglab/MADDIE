import os
import logging
import json
import shutil

import pandas as pd

from ..chem.evaluate import evaluate
from ..core.utils import get_cost
from ..config.resources import PROJECT_ROOT

runtime_logger = logging.getLogger("hit_identifier")


def _generation_work_root(kwargs, dirname: str) -> str:
    target_resource = kwargs.get("target_resource", {}) or {}
    tmp_root = target_resource.get("tmp_root") or kwargs["log_dir"]
    run_name = os.path.basename(os.path.normpath(kwargs["log_dir"]))
    return os.path.join(tmp_root, "hit_identifier_work", run_name, dirname)


def _cleanup_generation_case_dir(metadata):
    case_dir = metadata.get("case_dir", "") if isinstance(metadata, dict) else ""
    if not case_dir or not os.path.isdir(case_dir):
        return
    shutil.rmtree(case_dir)
    parent_dir = os.path.dirname(case_dir)
    while parent_dir and os.path.isdir(parent_dir) and not os.listdir(parent_dir):
        os.rmdir(parent_dir)
        if os.path.basename(parent_dir) == "hit_identifier_work":
            break
        parent_dir = os.path.dirname(parent_dir)


def _normalize_sbdd_action_input(action_input, memory):
    if not action_input:
        raise ValueError("SBDD_GENERATE requires at least one input item")

    first = action_input[0]
    if first in memory.stream:
        return action_input

    # Fallback: the model may output only a candidate id such as ['MOL002_ROW000'].
    if isinstance(first, str) and "_ROW" in first:
        matches = []
        for mem_id, block in memory.stream.items():
            if block.get("type") != "MOL":
                continue
            data = block.get("data")
            if data is None or "Candidate ID" not in data.columns:
                continue
            if (data["Candidate ID"] == first).any():
                matches.append(mem_id)
        if len(matches) == 1:
            return [matches[0], first]
        if len(matches) > 1:
            raise ValueError(f"Candidate id '{first}' appears in multiple molecule sets: {matches}")

    raise KeyError(first)


def _normalize_molecule_action_input(action_input, memory):
    if not action_input:
        raise ValueError("Molecule-seeded generation requires at least one input item")

    first = action_input[0]
    if first in memory.stream:
        return action_input

    if isinstance(first, str) and "_ROW" in first:
        matches = []
        for mem_id, block in memory.stream.items():
            if block.get("type") != "MOL":
                continue
            data = block.get("data")
            if data is None or "Candidate ID" not in data.columns:
                continue
            if (data["Candidate ID"] == first).any():
                matches.append(mem_id)
        if len(matches) == 1:
            return [matches[0], first]
        if len(matches) > 1:
            raise ValueError(f"Candidate id '{first}' appears in multiple molecule sets: {matches}")

    raise KeyError(first)


def run_action(action_id, action_input, memory, agent, metrics, **kwargs):
    #run existing action
    action_block = memory.stream[action_id]
    if action_block["type"] == "SBDD_GENERATE":
        action_input = _normalize_sbdd_action_input(action_input, memory)
        input_block = memory.stream[action_input[0]]
        target_resource = kwargs["target_resource"]
        if input_block["type"] != "POCKET":
            raise ValueError("SBDD_GENERATE always runs from the fixed/reference pocket and does not accept molecule seeds")
        sbdd_kwargs = {
            "target_pdb": target_resource["pdb"],
            "target_resource": target_resource,
            "config_path": target_resource["sbdd"]["config"],
            "python_bin": target_resource["sbdd"]["python_bin"],
            "runner_script": target_resource["sbdd"]["runner_script"],
            "hit_identifier_root": PROJECT_ROOT,
            "tmp_root": target_resource["tmp_root"],
            "output_root": _generation_work_root(kwargs, "sbdd_runs"),
            "n_samples": int(target_resource["sbdd"].get("sample_size", 20)),
            "round_index": kwargs.get("round_index"),
        }
        sbdd_kwargs["input_sdf"] = input_block.get("sdf") or ""
        outputs, metadata = action_block["func"](
            **sbdd_kwargs,
        )
    elif action_block["type"] == "LBDD_GENERATE":
        action_input = _normalize_molecule_action_input(action_input, memory)
        input_block = memory.stream[action_input[0]]
        target_resource = kwargs["target_resource"]
        if input_block["type"] == "MOL":
            if len(action_input) < 2:
                raise ValueError("LBDD_GENERATE requires a selected candidate id")
            input_df = input_block["data"]
            selected_candidate_id = action_input[1]
        elif input_block["type"] == "POCKET":
            input_df = pd.DataFrame([{
                "Candidate ID": "REFMOL",
                "SMILES": "",
                "SDF Path": input_block["sdf"],
            }])
            selected_candidate_id = "REFMOL"
        else:
            raise ValueError("LBDD_GENERATE requires a molecule candidate or reference pocket input")

        outputs, metadata = action_block["func"](
            input_df=input_df,
            target_pdb=target_resource["pdb"],
            target_resource=target_resource,
            config_path=target_resource["lbdd"]["config"],
            python_bin=target_resource["lbdd"]["python_bin"],
            runner_script=target_resource["lbdd"]["runner_script"],
            hit_identifier_root=PROJECT_ROOT,
            output_root=_generation_work_root(kwargs, "lbdd_runs"),
            n_samples=int(target_resource["lbdd"].get("sample_size", 20)),
            tmp_root=target_resource["tmp_root"],
            selected_candidate_id=selected_candidate_id,
            disallowed_candidate_ids=memory.get_used_seed_candidate_ids("LBDD_GENERATE"),
            round_index=kwargs.get("round_index"),
        )
    elif action_block["type"] == "VIRTUAL_SCREEN":
        outputs, metadata = action_block["func"](
            target_resource=kwargs["target_resource"],
            hit_identifier_root=PROJECT_ROOT,
            output_root=_generation_work_root(kwargs, "virtual_screen_runs"),
            round_index=kwargs.get("round_index"),
            excluded_drug_ids=memory.get_screened_drug_ids(),
        )
        vs_id = memory.add_vs(
            data=outputs,
            action=action_id,
            generation_metadata=metadata,
            return_id=True,
        )
        metadata = dict(metadata)
        metadata["vs_id"] = vs_id
        if len(outputs) > 0:
            if "SDF Path" not in outputs.columns:
                raise KeyError("Virtual screening top hits must include an 'SDF Path' column")
            if "SMILES" not in outputs.columns:
                raise KeyError("Virtual screening top hits must include a 'SMILES' column")
            data, outputs_metrics = evaluate(outputs, metrics, **kwargs)
            mol_id = memory.add_mol(
                data=data,
                action=action_id,
                metrics=outputs_metrics,
                generation_metadata=metadata,
                storage_root=os.path.join(kwargs["log_dir"], "molecule_library"),
                return_id=True,
            )
            memory.stream[vs_id]["data"] = memory.stream[mol_id]["data"].copy()
            results_json = metadata.get("results_json", "")
            if results_json:
                result_data = memory.stream[mol_id]["data"].drop(
                    columns=["affinity_json", "predicted_structure_path", "exported_structure_path", "Pose Source"],
                    errors="ignore",
                )
                with open(results_json, "w") as f:
                    json.dump(
                        {"results": result_data.to_dict(orient="records")},
                        f,
                        indent=2,
                    )
            top_hit_sdf_dir = os.path.join(metadata.get("case_dir", ""), "top_hit_sdf")
            if os.path.isdir(top_hit_sdf_dir):
                shutil.rmtree(top_hit_sdf_dir)
            _cleanup_generation_case_dir(metadata)
        else:
            mol_id = "EMPTY SET"
            _cleanup_generation_case_dir(metadata)
        return mol_id, int(get_cost(action_block["cost"])), metadata
    else:
        raise NotImplementedError
    
    if len(outputs) > 0:
        #run evaluation
        data, outputs_metrics = evaluate(outputs, metrics, **kwargs)
        #add mol to memory
        mol_id = memory.add_mol(
            data=data,
            action=action_id,
            metrics=outputs_metrics,
            generation_metadata=metadata,
            storage_root=os.path.join(kwargs["log_dir"], "molecule_library"),
            return_id=True,
        )
        results_json = metadata.get("results_json", "")
        if results_json:
            result_data = memory.stream[mol_id]["data"].copy()
            if action_block["type"] == "LBDD_GENERATE":
                seed_molecule = ""
                for seed_run in metadata.get("seed_runs", []) or []:
                    seed_molecule = seed_run.get("seed_candidate_id", "")
                    if seed_molecule:
                        break
                if not seed_molecule:
                    seed_molecule = metadata.get("seed_candidate_id", "")
                result_data["seed_molecule"] = seed_molecule
            with open(results_json, "w") as f:
                json.dump(
                    {"results": result_data.to_dict(orient="records")},
                    f,
                    indent=2,
                )
        _cleanup_generation_case_dir(metadata)
    else:
        mol_id = "EMPTY SET"
        _cleanup_generation_case_dir(metadata)
    #calculate cost
    cost = get_cost(action_block["cost"])
    return mol_id, int(cost), metadata
