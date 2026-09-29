import glob
import json
import logging
import os
import shutil
from typing import Dict, Tuple

import pandas as pd
from rdkit import Chem

from .process_utils import run_with_live_output

runtime_logger = logging.getLogger("hit_identifier")


def _prepare_case_dir(run_dir: str, pdb_path: str, sdf_path: str = "", round_index: int = None) -> Tuple[str, str, str]:
    if round_index is None:
        case_name = f"sbdd_round_{len(glob.glob(os.path.join(run_dir, 'sbdd_round_*'))) + 1:03d}"
    else:
        case_name = f"sbdd_round_{round_index:03d}"
    case_dir = os.path.join(run_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    pdb_name = os.path.basename(pdb_path)
    sdf_name = os.path.basename(sdf_path) if sdf_path else ""
    local_pdb = os.path.join(case_dir, pdb_name)
    shutil.copy2(pdb_path, local_pdb)
    if sdf_path:
        local_sdf = os.path.join(case_dir, sdf_name)
        shutil.copy2(sdf_path, local_sdf)
    return case_dir, pdb_name, sdf_name


def _collect_rows_from_sdf(result_dir: str):
    rows = []
    for sdf_path in sorted(glob.glob(os.path.join(result_dir, "**", "*.sdf"), recursive=True)):
        supplier = Chem.SDMolSupplier(sdf_path)
        for mol in supplier:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            if smi and "." not in smi:
                rows.append({"SMILES": smi, "SDF Path": sdf_path})
    return rows


def _remove_if_exists(path: str):
    if path and os.path.exists(path):
        os.remove(path)


def _write_generation_results_json(path: str, payload: Dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_sbdd_generation(
    pdb_path: str,
    sdf_path: str,
    config_path: str,
    python_bin: str,
    runner_script: str,
    hit_identifier_root: str,
    output_root: str,
    num_samples: int = 10,
    round_index: int = None,
) -> Tuple[pd.DataFrame, Dict]:
    if not runner_script:
        raise ValueError("runner_script is required for SBDD generation")
    if not hit_identifier_root:
        raise ValueError("hit_identifier_root is required for SBDD generation")
    case_dir, pdb_name, sdf_name = _prepare_case_dir(output_root, pdb_path, sdf_path, round_index=round_index)
    result_dir = os.path.join(case_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    cmd = [
        python_bin,
        runner_script,
        config_path,
        "--num_samples",
        str(num_samples),
        "--result_path",
        result_dir,
        "--pdb_path",
        os.path.join(case_dir, pdb_name),
    ]
    if sdf_name:
        cmd.extend(["--sdf_path", os.path.join(case_dir, sdf_name)])
    runtime_logger.debug("SBDD start")
    result = run_with_live_output(
        cmd,
        cwd=hit_identifier_root,
        env={"HIT_IDENTIFIER_ROOT": hit_identifier_root},
    )
    runtime_logger.debug("SBDD done")
    if result["returncode"] != 0:
        raise RuntimeError(
            f"SBDD generation failed with code {result['returncode']}\n"
            f"STDOUT:\n{result['stdout_tail']}\nSTDERR:\n{result['stderr_tail']}"
        )

    rows = _collect_rows_from_sdf(result_dir)
    _remove_if_exists(os.path.join(case_dir, pdb_name))
    if sdf_name:
        _remove_if_exists(os.path.join(case_dir, sdf_name))
    results_json = os.path.join(case_dir, "generation_results.json")
    metadata = {
        "method": "sbdd",
        "command": cmd,
        "config_path": config_path,
        "runner_script": runner_script,
        "hit_identifier_root": hit_identifier_root,
        "case_dir": case_dir,
        "result_dir": result_dir,
        "input_pdb_path": pdb_path,
        "input_sdf_path": sdf_path,
        "case_pdb_path": os.path.join(case_dir, pdb_name),
        "case_sdf_path": os.path.join(case_dir, sdf_name) if sdf_name else "",
        "results_json": results_json,
    }
    _write_generation_results_json(
        results_json,
        {
            "method": "sbdd",
            "num_samples": num_samples,
            "input_pdb_path": pdb_path,
            "input_sdf_path": sdf_path,
            "results": pd.DataFrame(rows).drop(columns=["SDF Path"], errors="ignore").to_dict(orient="records"),
        },
    )
    return pd.DataFrame(rows, columns=["SMILES", "SDF Path"]), metadata
