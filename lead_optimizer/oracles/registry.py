"""
Helpers for loading evaluator-backed MolOpt oracle callables and their metadata.
"""

import importlib
from pathlib import Path
import sys
from typing import Callable, Optional


class evaluator_path_context:
    """Temporarily expose the workspace root that contains evaluator/."""

    def __init__(self, evaluator_root: Optional[Path | str]):
        self.evaluator_root = Path(evaluator_root).resolve() if evaluator_root else None
        self.added = False

    def __enter__(self):
        if self.evaluator_root is None:
            return
        root = str(self.evaluator_root)
        if root not in sys.path:
            sys.path.insert(0, root)
            self.added = True

    def __exit__(self, exc_type, exc, traceback):
        if self.added:
            try:
                sys.path.remove(str(self.evaluator_root))
            except ValueError:
                pass


def _as_score(value, default=0.0) -> float:
    if value is None:
        return default
    return float(value)


TOXICITY_COLUMNS = {
    "herg": "hERG",
    "dili": "DILI",
    "clintox": "ClinTox",
    "mutagenicity": "AMES",
    "carcinogens": "Carcinogens_Lagunin",
}


TOXICITY_METRICS = {
    "herg": "hERG",
    "dili": "DILI",
    "clintox": "ClinTox",
    "mutagenicity": "Mutagenicity",
    "carcinogens": "Carcinogens",
}


def _required_score(value, label: str, smiles: str) -> float:
    if value is None:
        raise ValueError(f"{label} prediction returned None for SMILES: {smiles}")
    return float(value)


def _predict_admet(predict_admet, smiles: str, columns: list[str]) -> dict:
    try:
        return _predict_admet_float32(predict_admet, smiles, columns)
    except RuntimeError as exc:
        if "mat1 and mat2" not in str(exc) or "same dtype" not in str(exc):
            raise
        _coerce_admet_model_float32(predict_admet)
        return _predict_admet_float32(predict_admet, smiles, columns)


def _predict_admet_float32(predict_admet, smiles: str, columns: list[str]) -> dict:
    try:
        import torch
    except ImportError:
        return predict_admet(smiles, columns=columns)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        return predict_admet(smiles, columns=columns)
    finally:
        torch.set_default_dtype(previous_dtype)


def _coerce_admet_model_float32(predict_admet) -> None:
    model = predict_admet.__globals__.get("_MODEL")
    if model is None:
        return
    for models in getattr(model, "model_lists", []):
        for submodel in models:
            if hasattr(submodel, "float"):
                submodel.float()


def _build_registry_from_evaluator(evaluator_root: Optional[Path | str] = None):
    with evaluator_path_context(evaluator_root):
        from evaluator.tools import (  # noqa: E402
            calculate_hba,
            calculate_hbd,
            calculate_logp,
            calculate_molecular_weight,
            calculate_qed,
            calculate_rot_bonds,
            calculate_sascore,
            calculate_tpsa,
            predict_admet,
        )

    def toxicity_risk(smiles, endpoint):
        column = TOXICITY_COLUMNS[endpoint]
        result = _predict_admet(predict_admet, smiles, columns=[column])
        return _required_score(result.get(column), column, smiles)

    return {
        "qed": lambda smiles: _as_score(calculate_qed(smiles)),
        "sascore": lambda smiles: _as_score(calculate_sascore(smiles), default=10.0),
        "tpsa": lambda smiles: _as_score(calculate_tpsa(smiles)),
        "mol_wt": lambda smiles: _as_score(calculate_molecular_weight(smiles)),
        "logp": lambda smiles: _as_score(calculate_logp(smiles)),
        "hbd": lambda smiles: _as_score(calculate_hbd(smiles)),
        "hba": lambda smiles: _as_score(calculate_hba(smiles)),
        "rot_bonds": lambda smiles: _as_score(calculate_rot_bonds(smiles)),
        "herg": lambda smiles: toxicity_risk(smiles, "herg"),
        "dili": lambda smiles: toxicity_risk(smiles, "dili"),
        "clintox": lambda smiles: toxicity_risk(smiles, "clintox"),
        "mutagenicity": lambda smiles: toxicity_risk(smiles, "mutagenicity"),
        "carcinogens": lambda smiles: toxicity_risk(smiles, "carcinogens"),
    }


def _build_registry_from_liddia_oracles(molopt_repo: Optional[Path | str] = None):
    if molopt_repo:
        from lead_optimizer.tools.molopt import molopt_repo_context
        context = molopt_repo_context(Path(molopt_repo))
    else:
        from contextlib import nullcontext
        context = nullcontext()

    with context:
        liddia_oracles = importlib.import_module("liddia_oracles")

    def toxicity_risk(smiles, endpoint):
        tox = liddia_oracles._toxicity(smiles)
        metric = TOXICITY_METRICS[endpoint]
        return _required_score(tox.get(metric), metric, smiles)

    return {
        "qed": liddia_oracles.qed,
        "sascore": liddia_oracles._raw_sascore,
        "tpsa": liddia_oracles.tpsa,
        "mol_wt": liddia_oracles.mol_wt,
        "logp": lambda smiles: _as_score(liddia_oracles.calculate_logp(smiles)),
        "hbd": liddia_oracles.hbd,
        "hba": liddia_oracles.hba,
        "rot_bonds": liddia_oracles.rot_bonds,
        "herg": lambda smiles: toxicity_risk(smiles, "herg"),
        "dili": lambda smiles: toxicity_risk(smiles, "dili"),
        "clintox": lambda smiles: toxicity_risk(smiles, "clintox"),
        "mutagenicity": lambda smiles: toxicity_risk(smiles, "mutagenicity"),
        "carcinogens": lambda smiles: toxicity_risk(smiles, "carcinogens"),
    }


def build_oracle_registry(
    molopt_repo: Optional[Path | str] = None,
    evaluator_root: Optional[Path | str] = None,
    include_defaults: bool = True,
    custom_oracles: Optional[dict[str, Callable]] = None,
) -> dict[str, Callable]:
    """
    Build an oracle registry from evaluator-based oracles.
    
    Args:
        molopt_repo: Path to mol-opt repo, used as a fallback source for
            liddia_oracles.py.
        evaluator_root: Path containing evaluator/. In NINGLAB this is the
            repository root, because evaluator/ is a sibling of lead_optimizer/.
        include_defaults: Include standard oracles (qed, sascore, tpsa, mol_wt, logp).
        custom_oracles: Dict of custom oracle name → callable to add.
    
    Returns:
        Dict mapping oracle name → callable.
    
    Example:
        ```python
        from lead_optimizer.oracles.registry import build_oracle_registry
        
        registry = build_oracle_registry(
            molopt_repo="/path/to/mol-opt",
            custom_oracles={
                "compact_qed": lambda smi: qed(smi) - tpsa(smi)/200,
            }
        )
        ```
    """
    registry = {}
    
    if include_defaults:
        try:
            registry.update(_build_registry_from_evaluator(evaluator_root))
        except ImportError as evaluator_error:
            if evaluator_root is not None and molopt_repo is None:
                raise ImportError(
                    "Could not import evaluator-backed oracles. Make sure "
                    f"evaluator_root contains evaluator/: {evaluator_error}"
                )
            try:
                registry.update(_build_registry_from_liddia_oracles(molopt_repo))
            except ImportError as oracle_error:
                if molopt_repo is not None:
                    raise ImportError(
                        "Could not import evaluator-backed oracles from evaluator/ "
                        "or liddia_oracles.py. Check evaluator_root/molopt_repo: "
                        f"{evaluator_error}; {oracle_error}"
                    )
    
    if custom_oracles:
        registry.update(custom_oracles)
    
    if not registry:
        raise ValueError("No oracles were registered. Pass custom_oracles or enable include_defaults.")
    
    return registry


def oracle_info() -> dict[str, dict]:
    """
    Describe raw evaluator meaning separately from the MolOpt score contract.

    Example:
        ```python
        info = oracle_info()
        print(info["herg"]["raw_direction"])  # "decrease"
        print(info["herg"]["molopt_transform"])  # "request-dependent decrease desirability"
        ```
    """
    return {
        "qed": {
            "description": "Drug-likeness (QED, 0-1)",
            "raw_direction": "increase",
            "molopt_transform": "identity",
            "molopt_direction": "increase",
            "standalone_policy": "allowed",
        },
        "sascore": {
            "description": "Synthetic accessibility (1-10, lower is easier)",
            "raw_direction": "decrease",
            "molopt_transform": "10 - raw_sascore",
            "molopt_direction": "increase",
            "standalone_policy": "allowed_with_caveats",
        },
        "tpsa": {
            "description": "Topological polar surface area",
            "raw_direction": "target_range",
            "molopt_transform": "request-dependent bounded desirability",
            "molopt_direction": "increase",
            "standalone_policy": "avoid_standalone",
        },
        "mol_wt": {
            "description": "Molecular weight",
            "raw_direction": "target_range",
            "molopt_transform": "request-dependent bounded desirability",
            "molopt_direction": "increase",
            "standalone_policy": "avoid_standalone",
        },
        "logp": {
            "description": "LogP (lipophilicity)",
            "raw_direction": "target_range",
            "molopt_transform": "request-dependent bounded desirability",
            "molopt_direction": "increase",
            "standalone_policy": "avoid_standalone",
        },
        "hbd": {
            "description": "Hydrogen-bond donor count",
            "raw_direction": "target_range",
            "molopt_transform": "request-dependent bounded desirability",
            "molopt_direction": "increase",
            "standalone_policy": "avoid_standalone",
        },
        "hba": {
            "description": "Hydrogen-bond acceptor count",
            "raw_direction": "target_range",
            "molopt_transform": "request-dependent bounded desirability",
            "molopt_direction": "increase",
            "standalone_policy": "avoid_standalone",
        },
        "rot_bonds": {
            "description": "Rotatable-bond count",
            "raw_direction": "target_range",
            "molopt_transform": "request-dependent bounded desirability",
            "molopt_direction": "increase",
            "standalone_policy": "avoid_standalone",
        },
        "herg": {
            "description": "hERG liability (cardiac toxicity)",
            "raw_direction": "decrease",
            "molopt_transform": "request-dependent decrease desirability",
            "molopt_direction": "increase",
            "standalone_policy": "allowed_with_caveats",
        },
        "dili": {
            "description": "Drug-induced liver injury risk",
            "raw_direction": "decrease",
            "molopt_transform": "request-dependent decrease desirability",
            "molopt_direction": "increase",
            "standalone_policy": "allowed_with_caveats",
        },
        "clintox": {
            "description": "Clinical toxicity risk",
            "raw_direction": "decrease",
            "molopt_transform": "request-dependent decrease desirability",
            "molopt_direction": "increase",
            "standalone_policy": "allowed_with_caveats",
        },
        "mutagenicity": {
            "description": "Mutagenicity risk",
            "raw_direction": "decrease",
            "molopt_transform": "request-dependent decrease desirability",
            "molopt_direction": "increase",
            "standalone_policy": "allowed_with_caveats",
        },
        "carcinogens": {
            "description": "Carcinogenicity risk",
            "raw_direction": "decrease",
            "molopt_transform": "request-dependent decrease desirability",
            "molopt_direction": "increase",
            "standalone_policy": "allowed_with_caveats",
        },
    }
