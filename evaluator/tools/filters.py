"""Molecular filters: PAINS/Brenk (RDKit) and Lilly MedChem Rules (medchem)."""

from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def _mol(smiles: str) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles(smiles)


def _make_catalog(*catalogs) -> FilterCatalog:
    params = FilterCatalogParams()
    for cat in catalogs:
        params.AddCatalog(cat)
    return FilterCatalog(params)


_PAINS_CATALOG = _make_catalog(
    FilterCatalogParams.FilterCatalogs.PAINS_A,
    FilterCatalogParams.FilterCatalogs.PAINS_B,
    FilterCatalogParams.FilterCatalogs.PAINS_C,
)
_BRENK_CATALOG = _make_catalog(FilterCatalogParams.FilterCatalogs.BRENK)


def apply_pains_filter(smiles: str) -> bool:
    """Return True if molecule passes PAINS filter (no pan-assay interference)."""
    mol = _mol(smiles)
    if mol is None:
        return False
    return not _PAINS_CATALOG.HasMatch(mol)


def apply_brenk_filter(smiles: str) -> bool:
    """Return True if molecule passes Brenk structural alert filter."""
    mol = _mol(smiles)
    if mol is None:
        return False
    return not _BRENK_CATALOG.HasMatch(mol)


def apply_rdkit_filter(smiles: str) -> bool:
    """
    Apply combined RDKit filter: PAINS + Brenk + relaxed Lipinski Ro5.
    Returns True if molecule passes all checks.
    """
    mol = _mol(smiles)
    if mol is None:
        return False

    if _PAINS_CATALOG.HasMatch(mol):
        return False
    if _BRENK_CATALOG.HasMatch(mol):
        return False

    # Relaxed Ro5 (allow slight violations common in modern drug design)
    if Descriptors.ExactMolWt(mol) > 600:
        return False
    if Descriptors.MolLogP(mol) > 6:
        return False
    if rdMolDescriptors.CalcNumHBD(mol) > 5:
        return False
    if rdMolDescriptors.CalcNumHBA(mol) > 10:
        return False

    return True


def apply_lilly_filter(smiles: str) -> bool:
    """
    Apply Lilly MedChem Rules via the medchem library.
    Returns True if molecule passes.
    """
    mol = _mol(smiles)
    if mol is None:
        return False

    from medchem.functional import lilly_demerit_filter
    results = lilly_demerit_filter([smiles])
    return bool(results[0])


def get_filter_details(smiles: str) -> dict:
    """Run all filters and return detailed per-filter results."""
    mol = _mol(smiles)
    if mol is None:
        return {"valid": False, "error": "Invalid SMILES"}

    pains = apply_pains_filter(smiles)
    brenk = apply_brenk_filter(smiles)

    # Collect which PAINS/Brenk alerts fired
    pains_matches = [e.GetDescription() for e in _PAINS_CATALOG.GetMatches(mol)]
    brenk_matches = [e.GetDescription() for e in _BRENK_CATALOG.GetMatches(mol)]

    rdkit_pass = apply_rdkit_filter(smiles)
    lilly_pass = apply_lilly_filter(smiles)

    return {
        "valid": True,
        "PAINS": pains,
        "PAINS_alerts": pains_matches,
        "Brenk": brenk,
        "Brenk_alerts": brenk_matches,
        "RDKIT": rdkit_pass,
        "LILLY": lilly_pass,
    }
