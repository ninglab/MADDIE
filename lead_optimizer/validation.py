def _load_rdkit():
    try:
        from rdkit import Chem, rdBase
    except ImportError:
        return None
    return Chem, rdBase


def normalize_smiles(smiles: str) -> str:
    """Remove one accidental outer bracket pair when the inner SMILES is valid."""
    smiles = smiles.strip()
    rdkit = _load_rdkit()
    if rdkit is None:
        return smiles

    Chem, rdBase = rdkit
    with rdBase.BlockLogs():
        if Chem.MolFromSmiles(smiles) is not None:
            return smiles

        if smiles.startswith("[") and smiles.endswith("]"):
            inner = smiles[1:-1].strip()
            if Chem.MolFromSmiles(inner) is not None:
                return inner

    return smiles


def canonicalize_smiles(smiles: str) -> str | None:
    """Return a canonical deduplication key, validating with RDKit when present."""
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()
    if not smiles:
        return None

    rdkit = _load_rdkit()
    if rdkit is None:
        return smiles

    Chem, rdBase = rdkit
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)
