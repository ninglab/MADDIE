import logging
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from ..generation import run_lbdd_generation, run_sbdd_generation, run_virtual_screen_boltz

runtime_logger = logging.getLogger("hit_identifier")


def _load_valid_seed_mol(sdf_path: str):
    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        for mol in supplier:
            if mol is not None and mol.GetNumConformers():
                return mol
    except Exception as exc:
        raise ValueError(f"Could not sanitize seed SDF {sdf_path}: {exc}") from exc
    raise ValueError(f"Could not load a valid seed molecule from {sdf_path}")


def _rank_for_lbdd_seed(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    if "Vina Score" in df.columns:
        df["_vina_rank"] = df["Vina Score"].rank(method="average", ascending=True)
    else:
        df["_vina_rank"] = np.arange(1, len(df) + 1)
    if "QED" in df.columns:
        df["_qed_rank"] = df["QED"].rank(method="average", ascending=False)
    else:
        df["_qed_rank"] = np.arange(1, len(df) + 1)
    if "SA Score" in df.columns:
        df["_sa_rank"] = df["SA Score"].rank(method="average", ascending=True)
    else:
        df["_sa_rank"] = np.arange(1, len(df) + 1)
    return df.sort_values(
        ["_vina_rank", "_qed_rank", "_sa_rank", "SMILES"],
        ascending=[True, True, True, True],
    )


def sample_lbdd_from_molset(
    input_df: pd.DataFrame,
    target_pdb: str,
    target_resource: dict,
    config_path: str,
    python_bin: str,
    runner_script: str,
    hit_identifier_root: str,
    output_root: str,
    n_samples: int,
    tmp_root: str,
    selected_candidate_id: str = "",
    disallowed_candidate_ids=None,
    round_index: int = None,
) -> pd.DataFrame:
    if len(input_df) == 0:
        return pd.DataFrame(columns=["SMILES"]), {"selected_smiles": []}

    ranked = _rank_for_lbdd_seed(input_df)
    disallowed_candidate_ids = set(disallowed_candidate_ids or [])
    if disallowed_candidate_ids:
        ranked = ranked[~ranked["Candidate ID"].isin(disallowed_candidate_ids)]
    if len(ranked) == 0:
        raise ValueError("No unused candidates remain for LBDD seed selection")
    if selected_candidate_id:
        if selected_candidate_id in disallowed_candidate_ids:
            raise ValueError(f"Candidate ID '{selected_candidate_id}' was already used as an LBDD seed")
        selected_rows = ranked[ranked["Candidate ID"] == selected_candidate_id]
        if len(selected_rows) == 0:
            raise KeyError(f"Candidate ID '{selected_candidate_id}' not found in molecule set")
        fallback_rows = ranked[ranked["Candidate ID"] != selected_candidate_id]
        candidate_rows = pd.concat([selected_rows, fallback_rows], ignore_index=True)
    else:
        candidate_rows = ranked
    all_outputs = []
    metadata = {
        "selected_candidate_ids": [],
        "selected_smiles": [],
        "skipped_seed_candidates": [],
        "seed_runs": [],
    }

    os.makedirs(output_root, exist_ok=True)
    for seed_idx, (_, row) in enumerate(candidate_rows.iterrows()):
        seed_sdf = os.path.join(output_root, "lbdd_seed_0.sdf")
        candidate_id = row["Candidate ID"] if "Candidate ID" in row else ""
        source_sdf = row["SDF Path"] if "SDF Path" in row and isinstance(row["SDF Path"], str) else ""
        try:
            if not source_sdf:
                raise KeyError(
                    "LBDD seed selection now requires 'SDF Path' so shape-guided generation uses the real generated structure."
                )
            if not os.path.exists(source_sdf):
                raise FileNotFoundError(f"Selected seed SDF does not exist: {source_sdf}")
            seed_target_pdb = target_pdb
            source_protein = row["Protein Path"] if "Protein Path" in row and isinstance(row["Protein Path"], str) else ""
            if source_protein:
                if not os.path.exists(source_protein):
                    raise FileNotFoundError(f"Selected seed protein does not exist: {source_protein}")
                seed_target_pdb = source_protein
            seed_mol = _load_valid_seed_mol(source_sdf)
            from ..chem.docking.sbdd_vina_docking import dock_mol_to_sdf

            if source_protein:
                docked_seed_sdf = source_sdf
                docked_seed_vina = row["Vina Score"] if "Vina Score" in row else float("nan")
            else:
                runtime_logger.debug("Vina docking start")
                docked_seed_sdf, docked_seed_vina = dock_mol_to_sdf(
                    mol=seed_mol,
                    target_resource=target_resource,
                    output_sdf_path=seed_sdf,
                    ligand_sdf_path=source_sdf,
                    tmp_dir=tmp_root,
                )
                runtime_logger.debug("Vina docking done")
            _load_valid_seed_mol(docked_seed_sdf)
            outputs, seed_metadata = run_lbdd_generation(
                pdb_path=seed_target_pdb,
                sdf_path=docked_seed_sdf,
                config_path=config_path,
                python_bin=python_bin,
                runner_script=runner_script,
                hit_identifier_root=hit_identifier_root,
                output_root=output_root,
                num_samples=n_samples,
                round_index=round_index,
            )
        except Exception as exc:
            metadata["skipped_seed_candidates"].append({
                "candidate_id": candidate_id,
                "source_sdf": source_sdf,
                "reason": str(exc),
            })
            if os.path.exists(seed_sdf):
                os.remove(seed_sdf)
            continue
        if os.path.exists(seed_sdf):
            os.remove(seed_sdf)
        all_outputs.append(outputs)
        metadata["selected_candidate_ids"].append(candidate_id)
        metadata["selected_smiles"].append(row["SMILES"] if "SMILES" in row else "")
        seed_metadata["seed_sdf"] = seed_metadata.get("case_sdf_path", docked_seed_sdf)
        seed_metadata["seed_source_sdf_path"] = source_sdf
        seed_metadata["seed_protein_path"] = seed_target_pdb
        seed_metadata["seed_docked_vina"] = docked_seed_vina
        seed_metadata["seed_candidate_id"] = candidate_id
        for key in [
            "method",
            "case_dir",
            "result_dir",
            "results_json",
            "input_pdb_path",
            "input_sdf_path",
        ]:
            if key in seed_metadata:
                metadata[key] = seed_metadata[key]
        metadata["seed_candidate_id"] = candidate_id
        metadata["seed_runs"].append(seed_metadata)
        break

    if not all_outputs:
        raise ValueError(
            "No valid LBDD seed could be prepared. Skipped candidates: "
            f"{metadata['skipped_seed_candidates']}"
        )
    combined = pd.concat(all_outputs, ignore_index=True).drop_duplicates(subset=["SMILES"])
    return combined.head(n_samples), metadata


def sample_sbdd(
    target_pdb: str,
    target_resource: dict,
    config_path: str,
    python_bin: str,
    runner_script: str,
    hit_identifier_root: str,
    tmp_root: str,
    output_root: str,
    n_samples: int,
    input_sdf: str = "",
    round_index: int = None,
) -> pd.DataFrame:
    os.makedirs(output_root, exist_ok=True)
    if not input_sdf:
        input_sdf = target_resource.get("sdf") or ""
    seed_metadata = {
        "seed_candidate_id": "REFMOL",
        "seed_source_sdf_path": input_sdf,
        "seed_sdf": input_sdf,
    }

    outputs, metadata = run_sbdd_generation(
        pdb_path=target_pdb,
        sdf_path=input_sdf,
        config_path=config_path,
        python_bin=python_bin,
        runner_script=runner_script,
        hit_identifier_root=hit_identifier_root,
        output_root=output_root,
        num_samples=n_samples,
        round_index=round_index,
    )
    metadata.update(seed_metadata)
    return outputs.head(n_samples), metadata


def virtual_screen_boltz(
    target_resource: dict,
    hit_identifier_root: str,
    output_root: str,
    round_index: int = None,
    excluded_drug_ids=None,
):
    config = target_resource.get("virtual_screen", {}) or {}
    if not config.get("enabled"):
        raise ValueError("VIRTUAL_SCREEN was selected, but target_resource.virtual_screen.enabled is not true")
    return run_virtual_screen_boltz(
        pdb_path=target_resource["pdb"],
        drug_sheet=target_resource.get("drug_sheet") or config.get("drug_sheet", ""),
        python_bin=config["python_bin"],
        runner_script=config["runner_script"],
        hit_identifier_root=hit_identifier_root,
        output_root=output_root,
        cache_root=config.get("cache_root", ""),
        sample_size=int(config.get("sample_size", 0)),
        top_k=int(config.get("top_k", 50)),
        round_index=round_index,
        excluded_drug_ids=excluded_drug_ids or [],
        use_msa_server=bool(config.get("use_msa_server", False)),
        use_potentials=bool(config.get("use_potentials", True)),
        accelerator=config.get("accelerator", "gpu"),
        devices=int(config.get("devices", 1)),
        no_kernels=bool(config.get("no_kernels", False)),
        template_force=bool(config.get("template_force", False)),
        template_threshold=float(config.get("template_threshold", 1.0)),
    )
