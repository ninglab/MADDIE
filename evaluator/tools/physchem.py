"""Physicochemical property calculations using RDKit."""

from typing import Optional
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA


def _mol(smiles: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smiles)


def calculate_qed(smiles: str) -> Optional[float]:
    """Quantitative Estimate of Drug-likeness (0-1, higher is better)."""
    mol = _mol(smiles)
    return QED.qed(mol) if mol else None


def calculate_sascore(smiles: str) -> Optional[float]:
    """Synthetic Accessibility Score (1-10, lower is easier to synthesize)."""
    mol = _mol(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return sascorer.calculateScore(mol)
    except ImportError:
        pass
    try:
        from sascorer import calculateScore
        return calculateScore(mol)
    except ImportError:
        return None


def calculate_molecular_weight(smiles: str) -> Optional[float]:
    mol = _mol(smiles)
    return Descriptors.ExactMolWt(mol) if mol else None


def calculate_logp(smiles: str) -> Optional[float]:
    mol = _mol(smiles)
    return Descriptors.MolLogP(mol) if mol else None


def calculate_tpsa(smiles: str) -> Optional[float]:
    mol = _mol(smiles)
    return CalcTPSA(mol) if mol else None


def calculate_hbd(smiles: str) -> Optional[int]:
    mol = _mol(smiles)
    return rdMolDescriptors.CalcNumHBD(mol) if mol else None


def calculate_hba(smiles: str) -> Optional[int]:
    mol = _mol(smiles)
    return rdMolDescriptors.CalcNumHBA(mol) if mol else None


def calculate_rot_bonds(smiles: str) -> Optional[int]:
    mol = _mol(smiles)
    return rdMolDescriptors.CalcNumRotatableBonds(mol) if mol else None


def calculate_all_properties(smiles: str) -> dict:
    """Calculate all physicochemical properties in one call."""
    mol = _mol(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    result = {
        "QED": QED.qed(mol),
        "MolecularWeight": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
    }

    sa = calculate_sascore(smiles)
    if sa is not None:
        result["SAScore"] = sa

    return result
