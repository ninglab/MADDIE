"""
Uni-Dock wrapper for the MADDIE evaluator.
Drop-in replacement for run_quickvina2cuda().
"""
import os
import sys

# Set UNIDOCK_PATH if the unidock package is not already importable.
_unidock_path = os.environ.get("UNIDOCK_PATH")
if _unidock_path:
    sys.path.insert(0, _unidock_path)

from typing import Optional
from unidock import dock


def run_unidock(
    smiles: str,
    pdb_path: str,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Optional[float]:
    return dock(smiles, pdb_path, seed=seed)
