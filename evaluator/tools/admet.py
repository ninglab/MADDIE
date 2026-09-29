"""ADMET property prediction using ADMET-AI."""

from typing import List, Optional, Union
import pandas as pd

# Mapping from MADDIE task metric names → ADMET-AI DataFrame column names.
# Verified against admet_ai v0.0.9 output columns.
METRIC_TO_COLUMN: dict = {
    "Carcinogens": "Carcinogens_Lagunin",
    "hERG": "hERG",
    "Mutagenicity": "AMES",
    "DILI": "DILI",
    "ClinTox": "ClinTox",
}

# Columns grouped by ADMET category (for convenience)
ABSORPTION_COLS = [
    "Caco2_Wang", "HIA_Hou", "Pgp_Broccatelli",
    "Bioavailability_Ma", "Solubility_AqSolDB",
]
DISTRIBUTION_COLS = ["BBB_Martins", "PPBR_AZ", "VDss_Lombardo"]
METABOLISM_COLS = [
    "CYP2D6_Veith", "CYP3A4_Veith", "CYP2C9_Veith",
    "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels",
]
EXCRETION_COLS = ["Half_Life_Obach", "Clearance_Hepatocyte_AZ"]
TOXICITY_COLS = list(METRIC_TO_COLUMN.values())


_MODEL = None


def _load_model():
    """Load ADMETModel once and cache it globally."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import logging
        import os
        import sys
        # Suppress ADMET-AI / chemprop model loading output
        logging.disable(logging.WARNING)
        devnull = open(os.devnull, "w")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            from admet_ai import ADMETModel
            _MODEL = ADMETModel()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            devnull.close()
            logging.disable(logging.NOTSET)
        return _MODEL
    except ImportError as e:
        raise ImportError(
            "ADMET-AI is not installed. Run: pip install admet_ai"
        ) from e


def predict_admet(
    smiles: Union[str, List[str]],
    columns: Optional[List[str]] = None,
) -> Union[dict, pd.DataFrame]:
    """
    Predict ADMET properties for one or more molecules.

    Args:
        smiles: Single SMILES string or list of SMILES.
        columns: Specific column names to return. None returns all columns.

    Returns:
        dict for a single molecule, DataFrame for a list.
    """
    single = isinstance(smiles, str)
    smiles_list = [smiles] if single else smiles

    model = _load_model()
    preds: pd.DataFrame = model.predict(smiles=smiles_list)

    if columns:
        available = [c for c in columns if c in preds.columns]
        preds = preds[available]

    return preds.iloc[0].to_dict() if single else preds


def predict_toxicity_batch(smiles_list: List[str]) -> List[dict]:
    """Batch version of predict_toxicity. Calls model.predict() once for all molecules."""
    from rdkit import Chem
    empty = {k: None for k in METRIC_TO_COLUMN}
    results = [dict(empty) for _ in smiles_list]

    valid_indices = [i for i, smi in enumerate(smiles_list) if Chem.MolFromSmiles(smi) is not None]
    valid_smiles = [smiles_list[i] for i in valid_indices]
    if not valid_smiles:
        return results

    try:
        batch_df = predict_admet(valid_smiles)
    except Exception:
        return results

    for batch_idx, orig_idx in enumerate(valid_indices):
        out = {}
        for metric, col in METRIC_TO_COLUMN.items():
            out[metric] = float(batch_df.iloc[batch_idx][col]) if col in batch_df.columns else None
        results[orig_idx] = out

    return results


def predict_toxicity(smiles: str) -> dict:
    """
    Predict the five MADDIE toxicity endpoints:
    Carcinogens, hERG, Mutagenicity, DILI, ClinTox.

    Values are probabilities [0, 1]; lower is better (less toxic).
    Returns None for any property not found in the model output.
    """
    try:
        result = predict_admet(smiles)
    except Exception:
        return {k: None for k in METRIC_TO_COLUMN}

    out = {}
    for metric, col in METRIC_TO_COLUMN.items():
        out[metric] = float(result[col]) if col in result else None
    return out


def predict_absorption(smiles: str) -> dict:
    try:
        return predict_admet(smiles, ABSORPTION_COLS)
    except Exception:
        return {}


def predict_distribution(smiles: str) -> dict:
    try:
        return predict_admet(smiles, DISTRIBUTION_COLS)
    except Exception:
        return {}


def predict_metabolism(smiles: str) -> dict:
    try:
        return predict_admet(smiles, METABOLISM_COLS)
    except Exception:
        return {}


def predict_excretion(smiles: str) -> dict:
    try:
        return predict_admet(smiles, EXCRETION_COLS)
    except Exception:
        return {}


def predict_full_admet(smiles: str) -> dict:
    """Return the complete ADMET-AI property dictionary for a single molecule."""
    try:
        return predict_admet(smiles)
    except Exception as e:
        return {"error": str(e)}
