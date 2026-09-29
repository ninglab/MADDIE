from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

_FP_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def calculate_tanimoto_diversity(smiles_list: List[str]) -> dict:
    """
    Calculate Tanimoto diversity for a set of molecules.
    Uses Morgan fingerprints (radius=2, nBits=2048).
    Diversity = mean pairwise Tanimoto distance (1 - similarity).
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(_FP_GEN.GetFingerprint(mol))

    n = len(fps)
    if n < 2:
        return {"error": "Need at least 2 valid molecules to compute diversity."}

    distances = []
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        distances.extend(1 - s for s in sims)

    arr = np.array(distances)
    return {
        "n_molecules": n,
        "mean_tanimoto_distance": round(float(arr.mean()), 4),
        "min_tanimoto_distance": round(float(arr.min()), 4),
        "max_tanimoto_distance": round(float(arr.max()), 4),
    }
