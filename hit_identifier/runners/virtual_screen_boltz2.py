import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, Descriptors
import torch


AllChem.Descriptors = Descriptors


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}


def write_clean_template_pdb(input_path: str, output_path: str) -> str:
    """Write a Boltz-friendly protein template PDB from a pocket PDB."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    residue_map = {}
    residues_by_chain = {}
    cleaned_atoms = []
    atom_serial = 1
    with open(input_path, "r") as source:
        for line in source:
            record = line[:6].strip()
            if record != "ATOM":
                continue
            if len(line) < 54:
                continue
            res_name = line[17:20].strip().upper()
            if res_name not in AA3_TO_AA1:
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            chain_id = line[21:22].strip() or "A"
            residue_key = (chain_id, line[22:26].strip(), line[26:27].strip(), res_name)
            chain_residues = residue_map.setdefault(chain_id, {})
            if residue_key not in chain_residues:
                chain_residues[residue_key] = len(chain_residues) + 1
                residues_by_chain.setdefault(chain_id, []).append(res_name)
            new_resseq = chain_residues[residue_key]
            cleaned = (
                f"{line[:6]}{atom_serial:5d} "
                f"{line[12:16]} "
                f"{res_name:>3} {chain_id:1s}"
                f"{new_resseq:4d} "
                f"   {line[30:54]}"
                f"{line[54:].rstrip()}\n"
            )
            cleaned_atoms.append(cleaned)
            atom_serial += 1
    with open(output_path, "w") as target:
        target.write("HEADER CLEANED BOLTZ TEMPLATE\n")
        for chain_id, residues in residues_by_chain.items():
            total = len(residues)
            for idx in range(0, total, 13):
                chunk = residues[idx:idx + 13]
                serial = idx // 13 + 1
                target.write(
                    f"SEQRES {serial:3d} {chain_id:1s} {total:4d}  "
                    f"{' '.join(f'{res:>3s}' for res in chunk)}\n"
                )
        for cleaned in cleaned_atoms:
            target.write(cleaned)
        target.write("END\n")
    if atom_serial == 1:
        raise ValueError(f"No protein ATOM records could be cleaned from template PDB: {input_path}")
    return output_path


def sanitize_id(value: str, fallback: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or fallback


def extract_protein_sequences_from_pdb(pdb_path: str) -> List[Dict[str, str]]:
    chains = {}
    seen = set()
    with open(pdb_path, "r") as handle:
        for line in handle:
            if line[:6].strip() == "ENDMDL":
                break
            if line[:6].strip() != "ATOM":
                continue
            res_name = line[17:20].strip().upper()
            aa = AA3_TO_AA1.get(res_name)
            if aa is None:
                continue
            chain_id = line[21:22].strip() or "A"
            res_id = line[22:26].strip()
            insert_id = line[26:27].strip()
            key = (chain_id, res_id, insert_id)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain_id, []).append(aa)

    sequences = [
        {"id": sanitize_id(chain_id, f"CHAIN_{idx + 1}"), "sequence": "".join(seq)}
        for idx, (chain_id, seq) in enumerate(chains.items())
        if seq
    ]
    if not sequences:
        raise ValueError(f"No protein sequence could be extracted from PDB: {pdb_path}")
    return sequences


def pick_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    fieldnames = list(fieldnames or [])
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match
    return None


def normalize_ligand_smiles(smiles: str) -> Optional[str]:
    fragments = [fragment.strip() for fragment in str(smiles or "").split(".") if fragment.strip()]
    if not fragments:
        return None

    candidates = []
    for fragment in fragments:
        mol = Chem.MolFromSmiles(fragment, sanitize=False)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                continue
        heavy_atoms = mol.GetNumHeavyAtoms()
        carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        if heavy_atoms == 0 or carbon_atoms == 0:
            continue
        candidates.append((heavy_atoms, carbon_atoms, Chem.MolToSmiles(mol)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][2]


def read_drug_sheet(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        smiles_col = pick_column(reader.fieldnames, ["smiles", "SMILES"])
        if not smiles_col:
            raise KeyError("Drug sheet must contain a smiles column")

        rows = []
        for idx, row in enumerate(reader, start=1):
            raw_smiles = str(row.get(smiles_col, "")).strip()
            smiles = normalize_ligand_smiles(raw_smiles)
            if not smiles:
                continue
            rows.append({
                "drug_id": f"DRUG_{idx:05d}",
                "smiles": smiles,
            })
    if not rows:
        raise ValueError(f"No valid drug rows with SMILES found in {path}")
    return rows


def sample_drugs_without_overlap(
    drugs: List[Dict[str, str]],
    sample_size: int,
    excluded_drug_ids: set,
) -> List[Dict[str, str]]:
    eligible = [drug for drug in drugs if drug["drug_id"] not in excluded_drug_ids]
    if not eligible:
        raise ValueError("No unscreened drugs remain for virtual screening")
    if sample_size <= 0 or sample_size >= len(eligible):
        return eligible
    return random.sample(eligible, sample_size)


def write_boltz_yaml(
    path: str,
    protein_chains: List[Dict[str, str]],
    smiles: str,
    template_pdb_path: str = "",
    include_template: bool = False,
    template_force: bool = False,
    template_threshold: float = 0.0,
) -> None:
    sequences = []
    for chain in protein_chains:
        sequences.append({
            "protein": {
                "id": chain["id"],
                "sequence": chain["sequence"],
                "msa": "empty",
            }
        })
    sequences.append({"ligand": {"id": "L", "smiles": smiles}})
    payload = {
        "version": 1,
        "sequences": sequences,
        "properties": [{"affinity": {"binder": "L"}}],
    }
    if include_template and template_pdb_path and os.path.exists(template_pdb_path):
        template = {
            "pdb": template_pdb_path,
            "chain_id": [chain["id"] for chain in protein_chains],
        }
        if template_force:
            template["force"] = True
            template["threshold"] = float(template_threshold)
        payload["templates"] = [
            template
        ]
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def write_boltz_sitecustomize(result_path: str) -> str:
    patch_dir = os.path.join(result_path, "boltz_runtime_patch")
    os.makedirs(patch_dir, exist_ok=True)
    patch_path = os.path.join(patch_dir, "sitecustomize.py")
    with open(patch_path, "w") as handle:
        handle.write(
            "try:\n"
            "    from rdkit.Chem import AllChem, Descriptors\n"
            "    AllChem.Descriptors = Descriptors\n"
            "except Exception:\n"
            "    pass\n"
        )
    return patch_dir


def boltz_subprocess_env(result_path: str) -> Dict[str, str]:
    patch_dir = write_boltz_sitecustomize(result_path)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = patch_dir + (os.pathsep + existing if existing else "")
    return env


def template_can_be_parsed(
    pdb_path: str,
    protein_chains: List[Dict[str, str]],
    smiles: str,
    cache_root: str,
    result_path: str,
    template_force: bool = False,
    template_threshold: float = 0.0,
) -> bool:
    if not pdb_path or not os.path.exists(pdb_path):
        return False
    try:
        from pathlib import Path

        from boltz.data.mol import load_canonicals
        from boltz.data.parse.yaml import parse_yaml

        cache_path = Path(cache_root).expanduser() if cache_root else Path.home() / ".boltz"
        mol_dir = cache_path / "mols"
        if not mol_dir.exists():
            return False
        probe_path = os.path.join(result_path, "template_probe.yaml")
        write_boltz_yaml(
            probe_path,
            protein_chains,
            smiles,
            template_pdb_path=pdb_path,
            include_template=True,
            template_force=template_force,
            template_threshold=template_threshold,
        )
        ccd = load_canonicals(mol_dir)
        parse_yaml(Path(probe_path), ccd, mol_dir, boltz2=True)
        os.remove(probe_path)
        return True
    except Exception as exc:
        print(f"Boltz template skipped because it could not be parsed: {exc}")
        probe_path = os.path.join(result_path, "template_probe.yaml")
        if os.path.exists(probe_path):
            os.remove(probe_path)
        return False


def recursive_find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, key)
            if found is not None:
                return found
    return None


def parse_affinity_outputs(output_root: str, drug_id: str) -> Dict[str, object]:
    drug_candidates = []
    fallback_candidates = []
    for root, _, files in os.walk(output_root):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            path = os.path.join(root, filename)
            lowered = path.lower()
            if drug_id.lower() in lowered:
                drug_candidates.append(path)
            elif "affinity" in lowered:
                fallback_candidates.append(path)

    best = {}
    for path in sorted(drug_candidates) + sorted(fallback_candidates):
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        pred = recursive_find_key(payload, "affinity_pred_value")
        prob = recursive_find_key(payload, "affinity_probability_binary")
        if pred is not None or prob is not None:
            best = {
                "affinity_pred_value": pred,
                "affinity_probability_binary": prob,
                "affinity_json": path,
            }
            break
    if not best:
        best = {
            "affinity_pred_value": None,
            "affinity_probability_binary": None,
            "affinity_json": "",
        }
    return best


def find_predicted_structure(output_root: str, drug_id: str) -> str:
    candidates = []
    for root, _, files in os.walk(output_root):
        lowered_root = root.lower()
        if drug_id.lower() not in lowered_root:
            continue
        for filename in files:
            lowered_name = filename.lower()
            if lowered_name.endswith((".cif", ".pdb")) and drug_id.lower() in lowered_name:
                candidates.append(os.path.join(root, filename))
    return sorted(candidates)[0] if candidates else ""


def boltz_executable() -> str:
    env_bin = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(env_bin, "boltz")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("boltz")
    if found:
        return found
    return "boltz"


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", required=True)
    parser.add_argument("--drug_sheet", required=True)
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--cache_root", default="")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--exclude_drug_id", action="append", default=[])
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument("--use_potentials", action="store_true")
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--template_force", action="store_true")
    parser.add_argument("--template_threshold", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = os.path.join(args.result_path, "boltz_inputs")
    output_dir = os.path.join(args.result_path, "boltz_outputs")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    template_pdb_path = write_clean_template_pdb(
        args.pdb_path,
        os.path.join(args.result_path, "cleaned_template.pdb"),
    )

    protein_chains = extract_protein_sequences_from_pdb(args.pdb_path)
    all_drugs = read_drug_sheet(args.drug_sheet)
    excluded_drug_ids = set(str(value) for value in (args.exclude_drug_id or []) if value)
    drugs = sample_drugs_without_overlap(all_drugs, args.sample_size, excluded_drug_ids)
    include_template = bool(drugs) and template_can_be_parsed(
        template_pdb_path,
        protein_chains,
        drugs[0]["smiles"],
        args.cache_root,
        args.result_path,
        template_force=args.template_force,
        template_threshold=args.template_threshold,
    )
    for drug in drugs:
        yaml_path = os.path.join(input_dir, f"{drug['drug_id']}.yaml")
        write_boltz_yaml(
            yaml_path,
            protein_chains,
            drug["smiles"],
            template_pdb_path=template_pdb_path,
            include_template=include_template,
            template_force=args.template_force,
            template_threshold=args.template_threshold,
        )

    cmd = [
        boltz_executable(),
        "predict",
        input_dir,
        "--out_dir",
        output_dir,
        "--model",
        "boltz2",
        "--accelerator",
        args.accelerator,
        "--devices",
        str(args.devices),
        "--override",
    ]
    if args.cache_root:
        cmd.extend(["--cache", args.cache_root])
    if args.use_msa_server:
        cmd.append("--use_msa_server")
    if args.use_potentials:
        cmd.append("--use_potentials")
    if args.no_kernels:
        cmd.append("--no_kernels")

    subprocess.run(cmd, check=True, env=boltz_subprocess_env(args.result_path))

    results = []
    for drug in drugs:
        parsed = parse_affinity_outputs(output_dir, drug["drug_id"])
        row = dict(drug)
        row.update(parsed)
        row["predicted_structure_path"] = find_predicted_structure(output_dir, drug["drug_id"])
        results.append(row)

    def sort_key(row):
        prob = row.get("affinity_probability_binary")
        pred = row.get("affinity_pred_value")
        prob_key = float(prob) if prob is not None else -1.0
        pred_key = float(pred) if pred is not None else float("inf")
        return (-prob_key, pred_key, row["drug_id"])

    results = sorted(results, key=sort_key)
    top_results = results[: args.top_k] if args.top_k > 0 else results
    structure_dir = os.path.join(args.result_path, "predicted_structures")
    os.makedirs(structure_dir, exist_ok=True)
    for row_index, row in enumerate(top_results, start=1):
        structure_path = row.get("predicted_structure_path", "")
        if structure_path and os.path.exists(structure_path):
            extension = os.path.splitext(structure_path)[1]
            exported_path = os.path.join(structure_dir, f"{row_index:03d}_{row['drug_id']}{extension}")
            shutil.copy2(structure_path, exported_path)
            row["exported_structure_path"] = exported_path
        else:
            raise FileNotFoundError(f"Boltz predicted structure not found for {row['drug_id']}")
        row["SMILES"] = row["smiles"]
        row.pop("smiles", None)

    csv_path = os.path.join(args.result_path, "virtual_screen_results.csv")
    json_path = os.path.join(args.result_path, "virtual_screen_results.json")
    fieldnames = [
        "drug_id",
        "SMILES",
        "affinity_probability_binary",
        "affinity_pred_value",
        "predicted_structure_path",
        "exported_structure_path",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in top_results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    with open(json_path, "w") as handle:
        json.dump({
            "pdb_path": args.pdb_path,
            "drug_sheet": args.drug_sheet,
            "protein_chains": protein_chains,
            "boltz_command": cmd,
            "num_total_drugs": len(all_drugs),
            "num_excluded_drugs": len(excluded_drug_ids),
            "num_screened": len(results),
            "sample_size": args.sample_size,
            "top_k": args.top_k,
            "screened_drug_ids": [drug["drug_id"] for drug in drugs],
            "template_requested": bool(args.pdb_path),
            "template_used": include_template,
            "template_force": args.template_force,
            "template_threshold": args.template_threshold if args.template_force else "",
            "results": top_results,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
