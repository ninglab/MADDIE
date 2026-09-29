"""Boltz-2 based protein-ligand structure prediction and binding affinity."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def predict_structure_and_affinity(
    smiles: str,
    protein_pdb_path: Optional[str] = None,
    protein_sequence: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_msa_server: bool = True,
) -> dict:
    """
    Use Boltz-2 to predict protein-ligand binding structure and affinity.

    Requires boltz installed: pip install boltz
    GPU is strongly recommended.

    Args:
        smiles: Ligand SMILES string.
        protein_pdb_path: Path to protein PDB (preferred – uses experimental structure).
        protein_sequence: Amino acid sequence (used when no PDB available).
        cache_dir: Directory for caching Boltz-2 outputs.
        use_msa_server: Whether to use the online MSA server for alignment.

    Returns:
        dict with keys:
            'affinity'       – predicted binding affinity (lower = stronger, kcal/mol scale)
            'confidence'     – model confidence score [0, 1]
            'structure_path' – path to predicted complex PDB/CIF
            'error'          – error message if prediction failed
    """
    try:
        import boltz  # noqa: F401
    except ImportError:
        return {"error": "Boltz not installed. See: https://github.com/jwohlwend/boltz"}

    if protein_pdb_path is None and protein_sequence is None:
        return {"error": "Provide either protein_pdb_path or protein_sequence."}

    job_id = hashlib.md5(f"{smiles}{protein_pdb_path}{protein_sequence}".encode()).hexdigest()[:10]
    work_dir = Path(cache_dir) / job_id if cache_dir else Path(tempfile.mkdtemp()) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    cached = _load_cached_result(work_dir)
    if cached:
        return cached

    input_yaml = _write_input_yaml(
        smiles=smiles,
        protein_pdb_path=protein_pdb_path,
        protein_sequence=protein_sequence,
        work_dir=work_dir,
        job_id=job_id,
    )

    cmd = ["boltz", "predict", input_yaml, "--out_dir", str(work_dir)]
    if use_msa_server:
        cmd.append("--use_msa_server")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        return {"error": "boltz CLI not found. Install with: pip install boltz"}
    except subprocess.TimeoutExpired:
        return {"error": "Boltz-2 prediction timed out (>30 min)"}

    if proc.returncode != 0:
        return {"error": f"Boltz-2 failed:\n{proc.stderr[-2000:]}"}

    result = _parse_boltz_output(work_dir)
    _save_cached_result(work_dir, result)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_input_yaml(
    smiles: str,
    protein_pdb_path: Optional[str],
    protein_sequence: Optional[str],
    work_dir: Path,
    job_id: str,
) -> str:
    try:
        import yaml
    except ImportError:
        # Minimal YAML writer – Boltz input is simple enough
        yaml = None

    sequences = []

    if protein_pdb_path:
        sequences.append({"protein": {"id": "A", "structure": str(protein_pdb_path)}})
    else:
        sequences.append({"protein": {"id": "A", "sequence": protein_sequence}})

    sequences.append({"ligand": {"id": "B", "smiles": smiles}})

    data = {"sequences": sequences, "affinity": True}
    yaml_path = work_dir / f"input_{job_id}.yaml"

    if yaml:
        yaml_path.write_text(yaml.dump(data))
    else:
        # Hand-written YAML (only needed if PyYAML is absent)
        lines = ["sequences:"]
        for seq in sequences:
            for k, v in seq.items():
                lines.append(f"  - {k}:")
                for fk, fv in v.items():
                    lines.append(f"      {fk}: {fv!r}")
        lines.append("affinity: true")
        yaml_path.write_text("\n".join(lines))

    return str(yaml_path)


def _parse_boltz_output(work_dir: Path) -> dict:
    result: dict = {}

    # Affinity JSON
    for af in work_dir.rglob("*affinity*.json"):
        try:
            data = json.loads(af.read_text())
            if "affinity" in data:
                result["affinity"] = float(data["affinity"])
            if "confidence" in data:
                result["confidence"] = float(data["confidence"])
            break
        except Exception:
            continue

    # Confidence / summary JSON
    for cf in work_dir.rglob("*confidence*.json"):
        try:
            data = json.loads(cf.read_text())
            if "confidence_score" in data and "confidence" not in result:
                result["confidence"] = float(data["confidence_score"])
            break
        except Exception:
            continue

    # Structure file
    for pat in ("*.pdb", "*.cif"):
        hits = sorted(work_dir.rglob(pat))
        if hits:
            result["structure_path"] = str(hits[0])
            break

    if not result:
        result["error"] = "Could not parse Boltz-2 output"

    return result


def _load_cached_result(work_dir: Path) -> Optional[dict]:
    cache_file = work_dir / "result.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    return None


def _save_cached_result(work_dir: Path, result: dict) -> None:
    try:
        (work_dir / "result.json").write_text(json.dumps(result, indent=2))
    except Exception:
        pass
