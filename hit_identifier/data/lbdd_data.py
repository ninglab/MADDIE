import os
import sys


_HIT_IDENTIFIER_ROOT = os.environ.get(
    "HIT_IDENTIFIER_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DIFFSMOL_ROOT = os.path.join(_HIT_IDENTIFIER_ROOT, "vendor", "diffsmol")
if DIFFSMOL_ROOT not in sys.path:
    sys.path.insert(0, DIFFSMOL_ROOT)

from datasets import get_dataset  


def build_dataset(config, transform, pdb_path: str, sdf_path: str):
    files = [(pdb_path, sdf_path)]
    return get_dataset(
        config=config.data,
        transform=transform,
        files=files,
    )
