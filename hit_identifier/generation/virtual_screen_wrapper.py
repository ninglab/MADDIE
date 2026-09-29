import os
import json
import glob
import logging
import shutil
import subprocess
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from rdkit import Chem
from .process_utils import run_with_live_output

runtime_logger = logging.getLogger("hit_identifier")


def _boltz_library_paths(python_bin: str):
    env_root = os.path.dirname(os.path.dirname(os.path.abspath(python_bin)))
    patterns = [
        os.path.join(env_root, "lib", "python*", "site-packages", "nvidia", "cu*", "lib"),
        os.path.join(env_root, "lib", "python*", "site-packages", "nvidia", "cuda_nvrtc", "lib"),
    ]
    paths = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isdir(path):
                paths.append(path)
    env_lib = os.path.join(env_root, "lib")
    if os.path.isdir(env_lib):
        paths.append(env_lib)
    return list(dict.fromkeys(paths))


def _virtual_screen_env(python_bin: str, hit_identifier_root: str):
    extra_paths = _boltz_library_paths(python_bin)
    existing_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = extra_paths + ([existing_ld_library_path] if existing_ld_library_path else [])
    env = {"HIT_IDENTIFIER_ROOT": hit_identifier_root}
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def _obabel_convert(input_path: str, output_path: str):
    obabel = shutil.which("obabel")
    if not obabel:
        raise RuntimeError("OpenBabel executable 'obabel' was not found in the tool.py environment PATH")

    result = subprocess.run(
        [obabel, input_path, "-O", output_path],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenBabel failed to convert {input_path} to {output_path}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"OpenBabel did not produce a valid file: {output_path}")
    return output_path


def _validate_sdf_has_molecule(sdf_path: str):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return sdf_path
    raise ValueError(f"Converted SDF does not contain a valid RDKit molecule: {sdf_path}")


def _extract_ligand_resnames_from_cif(cif_path: str):
    ligand_names = []
    in_nonpoly_loop = False
    comp_id_index = None
    headers = []

    with open(cif_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "loop_":
                in_nonpoly_loop = False
                comp_id_index = None
                headers = []
                continue
            if line.startswith("_pdbx_entity_nonpoly."):
                in_nonpoly_loop = True
                headers.append(line)
                if line == "_pdbx_entity_nonpoly.comp_id":
                    comp_id_index = len(headers) - 1
                continue
            if not in_nonpoly_loop:
                continue
            if line.startswith("_"):
                headers.append(line)
                if line == "_pdbx_entity_nonpoly.comp_id":
                    comp_id_index = len(headers) - 1
                continue
            if line.startswith("#"):
                in_nonpoly_loop = False
                continue
            if comp_id_index is None:
                continue
            parts = line.split()
            if len(parts) <= comp_id_index:
                continue
            comp_id = parts[comp_id_index].strip("'\"")
            if comp_id and comp_id not in {".", "?"}:
                ligand_names.append(comp_id)

    aliases = []
    for name in ligand_names + ["LIG1", "LIG"]:
        if name not in aliases:
            aliases.append(name)
        short_name = name[:3]
        if short_name and short_name not in aliases:
            aliases.append(short_name)
    return aliases


def _is_ligand_resname(resname: str, ligand_resnames):
    return any(
        resname == ligand_resname
        or resname.startswith(ligand_resname)
        or ligand_resname.startswith(resname)
        for ligand_resname in ligand_resnames
    )


def _split_predicted_pdb(full_pdb_path: str, ligand_pdb_path: str, protein_pdb_path: str, ligand_resnames=None):
    ligand_resnames = ligand_resnames or ["LIG1", "LIG"]
    ligand_count = 0
    protein_count = 0
    os.makedirs(os.path.dirname(os.path.abspath(ligand_pdb_path)), exist_ok=True)
    with open(full_pdb_path, "r") as source, open(ligand_pdb_path, "w") as ligand, open(protein_pdb_path, "w") as protein:
        for line in source:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip()
            if _is_ligand_resname(resname, ligand_resnames):
                ligand.write(("HETATM" + line[6:]) if line.startswith("ATOM") else line)
                ligand_count += 1
            elif line.startswith("ATOM"):
                protein.write(line)
                protein_count += 1
        ligand.write("END\n")
        protein.write("END\n")
    if ligand_count == 0:
        raise ValueError(
            f"No ligand records found in converted PDB: {full_pdb_path}; "
            f"looked for residue names {ligand_resnames}"
        )
    if protein_count == 0:
        raise ValueError(f"No protein ATOM records found in converted PDB: {full_pdb_path}")
    return ligand_pdb_path, protein_pdb_path


def _ca_coords_by_residue(pdb_path: str):
    coords = []
    seen = set()
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip().upper() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            key = (line[21:22].strip() or "A", line[22:26].strip(), line[26:27].strip())
            if key in seen:
                continue
            seen.add(key)
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    return np.asarray(coords, dtype=float)


def _kabsch_transform(mobile: np.ndarray, reference: np.ndarray):
    if len(mobile) != len(reference) or len(mobile) < 3:
        raise ValueError(
            f"Cannot align predicted protein to template: matched CA counts are {len(mobile)} and {len(reference)}"
        )
    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    reference_centered = reference - reference_centroid
    covariance = mobile_centered.T @ reference_centered
    v, _, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(v @ wt))
    rotation = v @ correction @ wt
    translation = reference_centroid - mobile_centroid @ rotation
    aligned = mobile @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))
    return rotation, translation, rmsd


def _transform_pdb_coordinates(input_path: str, output_path: str, rotation: np.ndarray, translation: np.ndarray):
    with open(input_path, "r") as source, open(output_path, "w") as target:
        for line in source:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                try:
                    coord = np.asarray(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=float,
                    )
                except ValueError:
                    target.write(line)
                    continue
                x, y, z = coord @ rotation + translation
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            target.write(line)


def _align_complex_to_template(full_pdb_path: str, template_pdb_path: str, output_path: str):
    reference = _ca_coords_by_residue(template_pdb_path)
    mobile = _ca_coords_by_residue(full_pdb_path)
    n_match = min(len(reference), len(mobile))
    rotation, translation, rmsd = _kabsch_transform(mobile[:n_match], reference[:n_match])
    _transform_pdb_coordinates(full_pdb_path, output_path, rotation, translation)
    return {"template_alignment_ca_count": n_match, "template_alignment_ca_rmsd": rmsd}


def _convert_predicted_cif_to_ligand_sdf_and_protein_pdb(
    cif_path: str,
    ligand_sdf_path: str,
    protein_pdb_path: str,
    temp_root: str,
    template_pdb_path: str = "",
):
    if not cif_path or not os.path.exists(cif_path):
        raise FileNotFoundError(f"Missing Boltz predicted structure: {cif_path}")
    os.makedirs(temp_root, exist_ok=True)
    full_pdb_path = os.path.join(temp_root, "predicted_complex.pdb")
    ligand_pdb_path = os.path.join(temp_root, "predicted_ligand.pdb")
    ligand_resnames = _extract_ligand_resnames_from_cif(cif_path)
    _obabel_convert(cif_path, full_pdb_path)
    if template_pdb_path and os.path.exists(template_pdb_path):
        aligned_pdb_path = os.path.join(temp_root, "predicted_complex_aligned.pdb")
        _align_complex_to_template(full_pdb_path, template_pdb_path, aligned_pdb_path)
        full_pdb_path = aligned_pdb_path
    _split_predicted_pdb(full_pdb_path, ligand_pdb_path, protein_pdb_path, ligand_resnames=ligand_resnames)
    _obabel_convert(ligand_pdb_path, ligand_sdf_path)
    _validate_sdf_has_molecule(ligand_sdf_path)
    return ligand_sdf_path, protein_pdb_path


def _add_vs_ligand_sdfs(data: pd.DataFrame, case_dir: str, template_pdb_path: str = ""):
    data = data.copy()
    sdf_dir = os.path.join(case_dir, "top_hit_sdf")
    os.makedirs(sdf_dir, exist_ok=True)

    sdf_paths = []
    protein_paths = []
    kept_rows = []
    conversion_failures = []
    for row_index, (_, row) in enumerate(data.iterrows(), start=1):
        structure_path = row.get("exported_structure_path") or row.get("predicted_structure_path")
        drug_id = row["drug_id"]
        protein_pdb_path = os.path.join(sdf_dir, f"{row_index:03d}_{drug_id}_protein.pdb")
        sdf_path = os.path.join(sdf_dir, f"{row_index:03d}_{drug_id}.sdf")
        temp_root = os.path.join(sdf_dir, f"{row_index:03d}_{drug_id}_tmp")
        try:
            if not isinstance(structure_path, str) or not structure_path or not os.path.exists(structure_path):
                raise FileNotFoundError(f"Missing Boltz predicted structure for VS hit {row.get('drug_id', '')}")
            _convert_predicted_cif_to_ligand_sdf_and_protein_pdb(
                structure_path,
                ligand_sdf_path=sdf_path,
                protein_pdb_path=protein_pdb_path,
                temp_root=temp_root,
                template_pdb_path=template_pdb_path,
            )
        except Exception as exc:
            conversion_failures.append({
                "drug_id": drug_id,
                "structure_path": structure_path if isinstance(structure_path, str) else "",
                "sdf_path": sdf_path,
                "protein_pdb_path": protein_pdb_path,
                "reason": str(exc),
            })
            for path in [sdf_path, protein_pdb_path]:
                if os.path.exists(path):
                    os.remove(path)
            continue
        finally:
            if os.path.isdir(temp_root):
                shutil.rmtree(temp_root)
        kept_rows.append(row.to_dict())
        sdf_paths.append(sdf_path)
        protein_paths.append(protein_pdb_path)

    if conversion_failures:
        failures_path = os.path.join(case_dir, "conversion_failures.json")
        with open(failures_path, "w") as f:
            json.dump({"failures": conversion_failures}, f, indent=2)

    if kept_rows:
        data = pd.DataFrame(kept_rows)
    else:
        data = data.iloc[0:0].copy()
    data["SDF Path"] = sdf_paths
    data["Protein Path"] = protein_paths
    return data


def _cleanup_boltz_intermediates(case_dir: str):
    for dirname in ["boltz_inputs", "boltz_outputs", "predicted_structures", "boltz_runtime_patch"]:
        path = os.path.join(case_dir, dirname)
        if os.path.isdir(path):
            shutil.rmtree(path)


def run_virtual_screen_boltz(
    pdb_path: str,
    drug_sheet: str,
    python_bin: str,
    runner_script: str,
    hit_identifier_root: str,
    output_root: str,
    cache_root: str = "",
    sample_size: int = 0,
    top_k: int = 50,
    round_index: int = None,
    excluded_drug_ids=None,
    use_msa_server: bool = False,
    use_potentials: bool = True,
    accelerator: str = "gpu",
    devices: int = 1,
    no_kernels: bool = False,
    template_force: bool = False,
    template_threshold: float = 1.0,
) -> Tuple[pd.DataFrame, Dict]:
    if not runner_script:
        raise ValueError("runner_script is required for virtual screening")
    if not hit_identifier_root:
        raise ValueError("hit_identifier_root is required for virtual screening")
    if not drug_sheet:
        raise ValueError("drug_sheet is required for virtual screening")

    if round_index is None:
        existing = [
            name for name in os.listdir(output_root)
            if name.startswith("vs_round_")
        ] if os.path.isdir(output_root) else []
        case_name = f"vs_round_{len(existing) + 1:03d}"
    else:
        case_name = f"vs_round_{round_index:03d}"

    case_dir = os.path.join(output_root, case_name)
    os.makedirs(case_dir, exist_ok=True)
    results_csv = os.path.join(case_dir, "virtual_screen_results.csv")
    results_json = os.path.join(case_dir, "virtual_screen_results.json")
    excluded_drug_ids = sorted(set(str(value) for value in (excluded_drug_ids or []) if value))

    cmd = [
        python_bin,
        runner_script,
        "--pdb_path",
        pdb_path,
        "--drug_sheet",
        drug_sheet,
        "--result_path",
        case_dir,
        "--top_k",
        str(top_k),
        "--sample_size",
        str(sample_size),
        "--accelerator",
        accelerator,
        "--devices",
        str(devices),
    ]
    for drug_id in excluded_drug_ids:
        cmd.extend(["--exclude_drug_id", drug_id])
    if cache_root:
        cmd.extend(["--cache_root", cache_root])
    if use_msa_server:
        cmd.append("--use_msa_server")
    if use_potentials:
        cmd.append("--use_potentials")
    if no_kernels:
        cmd.append("--no_kernels")
    if template_force:
        cmd.append("--template_force")
        cmd.extend(["--template_threshold", str(template_threshold)])

    runtime_logger.debug("VS start")
    result = run_with_live_output(
        cmd,
        cwd=hit_identifier_root,
        env=_virtual_screen_env(python_bin, hit_identifier_root),
        tail_limit=8000,
    )
    runtime_logger.debug("VS done")
    if result["returncode"] != 0:
        raise RuntimeError(
            f"Virtual screening failed with code {result['returncode']}\n"
            f"STDOUT:\n{result['stdout_tail']}\nSTDERR:\n{result['stderr_tail']}"
        )

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Virtual screening did not produce expected results CSV: {results_csv}")
    data = pd.read_csv(results_csv)
    data = _add_vs_ligand_sdfs(data, case_dir, template_pdb_path=pdb_path)
    data = data.drop(
        columns=["affinity_json", "predicted_structure_path", "exported_structure_path", "Pose Source"],
        errors="ignore",
    )
    data.to_csv(results_csv, index=False)
    screened_drug_ids = []
    if os.path.exists(results_json):
        with open(results_json, "r") as f:
            _runner_payload = json.load(f)
        screened_drug_ids = _runner_payload.get("screened_drug_ids", [])
        result_payload = {"results": data.to_dict(orient="records")}
        with open(results_json, "w") as f:
            json.dump(result_payload, f, indent=2)
    if os.path.exists(results_csv):
        os.remove(results_csv)
    _cleanup_boltz_intermediates(case_dir)
    metadata = {
        "method": "virtual_screen_boltz2",
        "command": cmd,
        "case_dir": case_dir,
        "input_pdb_path": pdb_path,
        "drug_sheet": drug_sheet,
        "sample_size": sample_size,
        "excluded_drug_ids": excluded_drug_ids,
        "screened_drug_ids": screened_drug_ids,
        "results_json": results_json,
        "stdout_tail": result["stdout_tail"],
        "stderr_tail": result["stderr_tail"],
    }
    return data, metadata
