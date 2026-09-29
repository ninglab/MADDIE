"""Molecular docking using AutoDock Vina / QuickVina-W."""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Ligand preparation (SMILES → PDBQT)
# ---------------------------------------------------------------------------

def smiles_to_pdbqt(smiles: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convert SMILES to PDBQT string (or file).
    Tries meeko first (best quality), falls back to pure RDKit.
    Returns the PDBQT string, or None on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

    # Try meeko (handles rotatable bonds and charges properly)
    pdbqt_str = _smiles_to_pdbqt_meeko(mol)

    # Fallback: pure RDKit PDBQT writer (no external tools needed)
    if pdbqt_str is None:
        pdbqt_str = _mol_to_pdbqt_rdkit(mol)

    if pdbqt_str and output_path:
        Path(output_path).write_text(pdbqt_str)
    return pdbqt_str


def _smiles_to_pdbqt_meeko(mol) -> Optional[str]:
    """Use meeko to prepare ligand PDBQT (supports both meeko 0.5 and 0.7 APIs)."""
    try:
        from meeko import MoleculePreparation
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        # meeko 0.7+ API
        try:
            from meeko import PDBQTWriterLegacy
            pdbqt_str, is_ok, error = PDBQTWriterLegacy.write_string(mol_setups[0])
            if is_ok:
                return pdbqt_str
        except (ImportError, AttributeError):
            traceback.print_exc()

        # meeko ≤ 0.5 API
        try:
            return mol_setups[0].write_pdbqt_string()
        except AttributeError:
            traceback.print_exc()
    except Exception:
        traceback.print_exc()
    return None


def _mol_to_pdbqt_rdkit(mol) -> Optional[str]:
    """
    Pure RDKit fallback: write a minimal PDBQT for a 3D molecule.
    Assigns Gasteiger charges and maps elements to AutoDock atom types.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        traceback.print_exc()

    conf = mol.GetConformer()
    lines = ["ROOT\n"]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue  # skip explicit H in PDBQT heavy-atom section
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        symbol = atom.GetSymbol().upper()
        adtype = _LIGAND_ELEMENT_TO_ADTYPE.get(symbol, symbol[:1] or "C")
        try:
            charge = float(atom.GetDoubleProp("_GasteigerCharge"))
            if charge != charge:  # NaN check
                charge = 0.0
        except KeyError:
            traceback.print_exc()
            charge = 0.0
        name = f"{symbol}{idx + 1}"[:4].ljust(4)
        lines.append(
            f"ATOM  {idx + 1:5d} {name} LIG A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
            f"  1.00  0.00    {charge:+.3f} {adtype}\n"
        )
    lines.append("ENDROOT\n")
    lines.append("TORSDOF 0\n")
    return "".join(lines)


_LIGAND_ELEMENT_TO_ADTYPE: dict = {
    "C": "C", "N": "NA", "O": "OA", "S": "SA",
    "H": "HD", "P": "P", "F": "F",
    "CL": "Cl", "BR": "Br", "I": "I",
}


# ---------------------------------------------------------------------------
# Receptor preparation (PDB → PDBQT)
# ---------------------------------------------------------------------------

def prepare_receptor_pdbqt(pdb_path: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """
    Prepare protein receptor PDBQT from PDB file.
    Caches result so repeat calls are instant.
    Returns path to .pdbqt file, or None on failure.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        return None

    out_dir = Path(cache_dir) if cache_dir else pdb_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_path = out_dir / f"{pdb_path.stem}_receptor.pdbqt"

    if pdbqt_path.exists():
        print("PDBQT already exists...")
        return str(pdbqt_path)

    # Try ADFRsuite / mgltools prepare_receptor (CLI)
    for cmd in ["prepare_receptor", "prepare_receptor4.py"]:
        try:
            result = subprocess.run(
                [cmd, "-r", str(pdb_path), "-o", str(pdbqt_path), "-A", "hydrogens"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and pdbqt_path.exists():
                return str(pdbqt_path)
        except FileNotFoundError:
            traceback.print_exc()
            continue

    # Fallback: pure Python PDB → PDBQT converter (no external dependencies)
    # if _pdb_to_pdbqt_python(pdb_path, pdbqt_path):
    #     return str(pdbqt_path)

    return None


# AutoDock atom type mapping (element → AD4 type)
_ELEMENT_TO_ADTYPE: dict = {
    "C": "C", "N": "NA", "O": "OA", "S": "SA",
    "H": "HD", "P": "P", "F": "F",
    "CL": "Cl", "BR": "Br", "I": "I",
    "ZN": "Zn", "FE": "Fe", "CA": "Ca",
    "MG": "Mg", "MN": "Mn", "CU": "Cu",
}


def _pdb_to_pdbqt_python(pdb_path: Path, pdbqt_path: Path) -> bool:
    """
    Minimal pure-Python PDB → PDBQT converter for receptor preparation.
    Keeps only ATOM records (protein heavy atoms), assigns AutoDock atom types,
    and sets partial charges to 0.000.
    """
    lines_out = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            # Element: prefer column 77-78 (standard PDB), else infer from atom name
            element = line[76:78].strip().upper() if len(line) > 76 else ""
            if not element:
                atom_name = line[12:16].strip().lstrip("0123456789")
                element = "".join(c for c in atom_name if c.isalpha())[:2].upper()

            adtype = _ELEMENT_TO_ADTYPE.get(element, element[:1] or "C")

            # Build PDBQT line: original PDB columns 1-66, then charge + adtype
            base = line.rstrip().ljust(66)
            lines_out.append(f"{base}    0.000  {adtype}\n")

    if not lines_out:
        return False

    pdbqt_path.write_text("".join(lines_out))
    return True


# ---------------------------------------------------------------------------
# Binding box extraction
# ---------------------------------------------------------------------------

def get_binding_box_from_pdb(pdb_path: str) -> Tuple[list, list]:
    """
    Estimate docking box center and size from the HETATM ligand in the PDB.
    Returns (center_xyz, box_size_xyz) as lists of floats.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            res = line[17:20].strip()
            if res in ("HOH", "WAT", "SOL"):
                continue
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                traceback.print_exc()
                continue

    if not coords:
        # Fall back to protein Cα centroid
        for line in open(pdb_path):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    traceback.print_exc()
                    continue

    if not coords:
        return [0.0, 0.0, 0.0], [22.5, 22.5, 22.5]

    arr = np.array(coords)
    center = arr.mean(axis=0).tolist()
    span = arr.max(axis=0) - arr.min(axis=0)
    # Add 8 Å padding; clamp to [15, 30]
    box_size = [float(np.clip(s + 8, 15, 30)) for s in span]
    return center, box_size


# ---------------------------------------------------------------------------
# Docking engines
# ---------------------------------------------------------------------------

def _run_vina_python(
    smiles: str,
    receptor_pdbqt: str,
    center: list,
    box_size: list,
    exhaustiveness: int,
    cache_dir: Optional[str],
) -> Optional[float]:
    """AutoDock Vina via Python API (vina package)."""
    from vina import Vina

    work_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:10]
    lig_path = work_dir / f"lig_{smiles_hash}.pdbqt"

    pdbqt = smiles_to_pdbqt(smiles, str(lig_path))
    if pdbqt is None:
        return None

    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
    v.set_ligand_from_file(str(lig_path))
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=1)
    energies = v.energies(n_poses=1)
    return float(energies[0][0])


def _run_quickvina_cmdline(
    smiles: str,
    receptor_pdbqt: str,
    center: list,
    box_size: list,
    exhaustiveness: int,
    cache_dir: Optional[str],
) -> Optional[float]:
    """QuickVina-W (or vina) via command line."""
    work_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
    work_dir.mkdir(parents=True, exist_ok=True)

    smiles_hash = hashlib.md5(smiles.encode()).hexdigest()[:10]
    lig_path = work_dir / f"lig_{smiles_hash}.pdbqt"
    out_path = work_dir / f"out_{smiles_hash}.pdbqt"
    log_path = work_dir / f"log_{smiles_hash}.txt"

    pdbqt = smiles_to_pdbqt(smiles, str(lig_path))
    if pdbqt is None:
        return None

    # Try quickvina-w, quickvina2, then fall back to vina
    for exe in ["qvina-w", "quickvina-w", "qvina2", "vina"]:
        cmd = [
            exe,
            "--receptor", receptor_pdbqt,
            "--ligand", str(lig_path),
            "--out", str(out_path),
            "--log", str(log_path),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            traceback.print_exc()
            continue

        if result.returncode != 0:
            continue

        # Parse best affinity from stdout
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "1":
                try:
                    return float(parts[1])
                except ValueError:
                    traceback.print_exc()

        # Also try log file
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "1":
                    try:
                        return float(parts[1])
                    except ValueError:
                        traceback.print_exc()

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_docking(
    smiles: str,
    pdb_path: str,
    cache_dir: Optional[str] = None,
    exhaustiveness: int = 8,
) -> Optional[float]:
    """
    High-level docking entry point.
    Handles receptor prep and binding-box extraction automatically.
    Tries Python Vina API first, then QuickVina / Vina CLI.

    Returns binding affinity in kcal/mol (more negative = stronger binding),
    or None if docking fails.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        return None

    cache_path = Path(cache_dir) if cache_dir else pdb_path.parent / "docking_cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt = prepare_receptor_pdbqt(str(pdb_path), str(cache_path))
    if receptor_pdbqt is None:
        return None

    center, box_size = get_binding_box_from_pdb(str(pdb_path))

    # Try Python API (faster, no subprocess overhead)
    try:
        score = _run_vina_python(smiles, receptor_pdbqt, center, box_size,
                                  exhaustiveness, str(cache_path))
        if score is not None:
            return score
    except Exception:
        traceback.print_exc()

    # Fallback: CLI
    return _run_quickvina_cmdline(smiles, receptor_pdbqt, center, box_size,
                                   exhaustiveness, str(cache_path))
