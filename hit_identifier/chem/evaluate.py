import logging
import os

from tqdm import tqdm

import numpy as np
import pandas as pd
from typing import Dict

runtime_logger = logging.getLogger("hit_identifier")

from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem
from rdkit.Chem import QED

def get_qed(smi):
    mol = Chem.MolFromSmiles(smi)
    return QED.qed(mol)

def get_sa(smi):
    mol = Chem.MolFromSmiles(smi)
    return sascorer.calculateScore(mol)

def get_heavy_atom_count(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms()

def _load_first_valid_sdf_mol(sdf_path: str):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError(f"Could not load a valid molecule from SDF: {sdf_path}")


def get_vina_from_sdf(sdf_path: str, target_pdb: str, target_resource: Dict = None):
    """Return (vina_score, docked_sdf_path). docked_sdf_path is non-empty only in
    docking_score mode, where QuickVina2-CUDA produces a re-docked pose."""
    from .docking.sbdd_vina_docking import compute_vina_score
    mol = _load_first_valid_sdf_mol(sdf_path)
    scoring_target_resource = None
    if target_resource is not None:
        scoring_target_resource = dict(target_resource)
        scoring_target_resource["pdb"] = target_pdb
        scoring_target_resource["sdf"] = sdf_path
    score, docked_sdf_path = compute_vina_score(
        mol=mol,
        protein_path=target_pdb,
        target_resource=scoring_target_resource,
        ligand_sdf_path=sdf_path,
    )
    return float(score), docked_sdf_path or ""

def evaluate(data: pd.DataFrame, metrics: Dict[str, str], **kwargs):
    return_metrics = {}
    smiles = data["SMILES"].tolist()
    if "Heavy Atom Count" not in data.columns:
        data["Heavy Atom Count"] = [get_heavy_atom_count(smi) for smi in smiles]
    for key, key_data in metrics.items():
        if key == "size":
            return_metrics[key] = len(data)
        elif key == "qed":
            if key_data not in data.columns:
                values = [get_qed(smi) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {
                "min": np.min(data[key_data]),
                "max": np.max(data[key_data]),
                "median": np.median(data[key_data]),
            }
        elif key == "sa":
            if key_data not in data.columns:
                values = [get_sa(smi) for smi in smiles]
                data[key_data] = values
            return_metrics[key] = {
                "min": np.min(data[key_data]),
                "max": np.max(data[key_data]),
                "median": np.median(data[key_data]),
            }
        elif key == "vina":
            path_to_pdb = kwargs["target_pdb"]
            if key_data not in data.columns:
                if "SDF Path" not in data.columns:
                    raise KeyError(
                        "Vina evaluation now requires an 'SDF Path' column so docking uses the real generated structure."
                    )
                runtime_logger.debug("Vina evaluation start")
                vina_scores = []
                docked_sdf_paths = []
                for _, row in tqdm(data.iterrows(), total=len(data), desc="Vina docking"):
                    sdf_path = row["SDF Path"]
                    row_pdb = row["Protein Path"] if "Protein Path" in row and isinstance(row["Protein Path"], str) and row["Protein Path"] else path_to_pdb
                    score, docked_sdf_path = get_vina_from_sdf(
                        sdf_path, row_pdb, target_resource=kwargs.get("target_resource")
                    )
                    vina_scores.append(score)
                    docked_sdf_paths.append(docked_sdf_path)
                data[key_data] = vina_scores
                # docking_score mode returns a re-docked pose; overwrite SDF Path with
                # it so everything derived from SDF Path (molecule_library/*.sdf,
                # successful_results, LBDD seed source) reflects the scored pose.
                # vina_score (score_only) mode returns "", so SDF Path is left as-is.
                updated_paths = list(data["SDF Path"])
                for i, docked_sdf_path in enumerate(docked_sdf_paths):
                    if docked_sdf_path and os.path.exists(docked_sdf_path):
                        updated_paths[i] = docked_sdf_path
                data["SDF Path"] = updated_paths
                runtime_logger.debug("Vina evaluation done")
            failed = int(np.isinf(data[key_data]).sum())
            if failed > 0:
                failure_rows = data[np.isinf(data[key_data])].copy()
                preview_lines = []
                for _, row in failure_rows.head(5).iterrows():
                    candidate_id = row.get("Candidate ID", "")
                    sdf_path = row.get("SDF Path", "")
                    error = row.get("Vina Error", "")
                    prefix = f"{candidate_id}: " if candidate_id else ""
                    detail = error or "Vina Score is inf"
                    preview_lines.append(f"{prefix}{sdf_path}: {detail}")
                preview = "\n".join(preview_lines)
                remaining = failed - min(failed, 5)
                if remaining > 0:
                    preview += f"\n... {remaining} more Vina evaluation failures"
                runtime_logger.warning(
                    f"Dropping {failed} molecule(s) with inf Vina score (Uni-Dock failed); "
                    f"{len(data) - failed} remaining.\n{preview}"
                )
                data = data[~np.isinf(data[key_data])].reset_index(drop=True)
                smiles = data["SMILES"].tolist()  # re-sync after dropping inf rows
                if data.empty:
                    raise RuntimeError(
                        f"Vina evaluation produced invalid inf score for all molecule(s):\n{preview}"
                    )
            return_metrics[key] = {
                "min": np.min(data[key_data]),
                "max": np.max(data[key_data]),
                "median": np.median(data[key_data]),
                "failed": failed,
            }
        else:
            raise NotImplementedError
    return data, return_metrics
