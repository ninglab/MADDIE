import ctypes
import os
from pathlib import Path


def preload_conda_cpp_runtime() -> None:
    """Prefer the active conda C++ runtime over older site-wide libraries."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    lib_dir = Path(conda_prefix) / "lib"
    candidates = [
        lib_dir / "libstdc++.so.6",
        lib_dir / "libgcc_s.so.1",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # If preloading fails we fall back to the default loader behavior.
            pass
