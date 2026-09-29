"""Error detection and diagnosis for the Hit Identifier generation flow.

These helpers are pure/self-contained (no LLM agent, no heavy chem deps). tool.py
raises/uses them to fail fast on bad inputs and to turn any failure into a
structured diagnosis that the LLM conclusion writer consumes.
"""
import math
import os
import re
import traceback
from typing import Dict


POCKET_RADIUS = 10.0  # mirrors diffsmol RADIUS in vendor/diffsmol/utils/data.py


class ReferencePocketEmptyError(Exception):
    """Raised before generation when the reference ligand does not sit inside the
    target pocket, so LBDD would build an empty pocket and crash downstream."""

    def __init__(self, evidence: Dict):
        self.evidence = evidence or {}
        super().__init__(
            "Reference ligand does not sit inside the target pocket "
            f"(pocket_residues={self.evidence.get('pocket_residues')}, "
            f"reference_vina={self.evidence.get('reference_vina')})."
        )


def _detect_unsupported_reference(prompt: str, sdf_path: str, pdb_path: str, drug_sheet_path: str) -> str:
    """The prompt parser only routes .sdf paths into sdf_path. If the user named a
    reference-ligand file in a non-SDF molecular format (.smi/.csv/.mol2/...), it is
    silently dropped. Return that path so the caller can fail fast with a clear reason."""
    if sdf_path:
        return ""
    assigned = {p for p in (pdb_path, drug_sheet_path) if p}
    for match in re.finditer(r"(/[^\s'\"]+\.(?:smi|csv|mol2|mol|txt))\b", prompt, re.IGNORECASE):
        path = match.group(1)
        if path not in assigned:
            return path
    return ""


def _ligand_conformer_coords(mol):
    if mol is None or mol.GetNumConformers() == 0:
        return []
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append((pos.x, pos.y, pos.z))
    return coords


def _protein_atoms_by_residue(pdb_path):
    residues = {}
    if not pdb_path or not os.path.exists(pdb_path):
        return residues
    with open(pdb_path, "r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            res_key = (line[21], line[22:27].strip())  # chain + resSeq(+icode)
            residues.setdefault(res_key, []).append((x, y, z))
    return residues


def _check_reference_alignment(target_resource: Dict, reference_mol) -> Dict:
    """Dependency-free mirror of diffsmol pocket extraction: count protein residues
    within POCKET_RADIUS of the reference ligand. pocket_residues == 0 means the
    ligand is not in the pocket and LBDD would build an empty protein point cloud."""
    lig = _ligand_conformer_coords(reference_mol)
    evidence = {
        "sdf": target_resource.get("sdf", ""),
        "pdb": target_resource.get("pdb", ""),
        "pocket_radius": POCKET_RADIUS,
        "ligand_atoms": len(lig),
    }
    if not lig:
        evidence.update({"pocket_residues": 0, "ligand_is_2d": True,
                         "min_ligand_protein_dist": None})
        return evidence

    zs = [p[2] for p in lig]
    evidence["ligand_is_2d"] = (max(zs) - min(zs)) < 1e-3

    residues = _protein_atoms_by_residue(target_resource.get("pdb", ""))
    r2 = POCKET_RADIUS * POCKET_RADIUS
    pocket = 0
    min_d2 = None
    for atoms in residues.values():
        in_pocket = False
        for (px, py, pz) in atoms:
            for (lx, ly, lz) in lig:
                d2 = (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2
                if min_d2 is None or d2 < min_d2:
                    min_d2 = d2
                if d2 <= r2:
                    in_pocket = True
        if in_pocket:
            pocket += 1
    evidence["pocket_residues"] = pocket
    evidence["min_ligand_protein_dist"] = round(math.sqrt(min_d2), 2) if min_d2 is not None else None
    return evidence


def _extract_path_from_message(msg: str) -> str:
    match = re.search(r"(/[^\s'\":]+\.(?:sdf|smi|csv|txt|mol2|mol|pdb))", msg)
    return match.group(1) if match else ""


def _diagnose_error(exc, *, task=None, target_resource=None, stage="generate") -> Dict:
    """Classify a hit_identifier failure into a structured diagnosis consumed by the
    LLM conclusion writer. Pure rule-based; no model call here."""
    msg = str(exc)
    # prep-stage raises have no task; they may carry context on the exception itself.
    target = (task or {}).get("target", "") or getattr(exc, "hi_target", "")
    allowed = (task or {}).get("allowed_action_types") or getattr(exc, "hi_allowed", []) or []
    has_ref = bool((target_resource or {}).get("sdf"))
    tb_tail = "\n".join(traceback.format_exc().strip().splitlines()[-8:])

    # (2) reference ligand not in pocket — proactive pre-check
    if isinstance(exc, ReferencePocketEmptyError):
        ev = dict(exc.evidence)
        ev.setdefault("target", target)
        if ev.get("ligand_is_2d"):
            root_cause = (
                "The reference ligand is 2D (no real 3D conformer, all atoms coplanar), so it cannot be "
                "placed in the target pocket and the LBDD pocket comes out empty."
            )
            fix = "Generate a 3D conformer for the reference ligand and align/dock it into the pocket before use."
        else:
            root_cause = (
                "The reference ligand's 3D coordinates lie outside the target pocket (misaligned with the "
                f"protein), so no protein residues fall within {ev.get('pocket_radius', POCKET_RADIUS)} A and the LBDD pocket is empty."
            )
            fix = "Re-align or re-dock the reference ligand into the pocket so it shares the protein coordinate frame."
        return {
            "error_type": "reference_not_in_pocket",
            "target": target or ev.get("target", ""),
            "stage": stage,
            "root_cause": root_cause,
            "evidence": ev,
            "suggestions": [fix],
        }

    # (2) reference ligand not in pocket — reactive (downstream LBDD crash)
    if "No valid LBDD seed could be prepared" in msg or "must match the size of tensor" in msg:
        return {
            "error_type": "reference_not_in_pocket",
            "target": target,
            "stage": stage,
            "root_cause": (
                "LBDD seed preparation failed: the reference ligand could not be placed in the pocket "
                "(empty protein point cloud), so no seed could be built."
            ),
            "evidence": {
                "sdf": (target_resource or {}).get("sdf", ""),
                "pdb": (target_resource or {}).get("pdb", ""),
                "detail": tb_tail,
            },
            "suggestions": [
                "Provide a reference ligand aligned inside the target pocket.",
                "Verify the reference ligand and protein share the same coordinate frame.",
            ],
        }

    # (3)/(5) reference file could not be loaded
    if "Could not load a valid reference molecule" in msg or "Bad input file" in msg:
        path = _extract_path_from_message(msg) or (target_resource or {}).get("sdf", "")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".sdf":
            return {
                "error_type": "reference_sdf_corrupt",
                "target": target,
                "stage": stage,
                "root_cause": (
                    "The reference file has a .sdf extension but its contents are not a valid molblock, "
                    "so RDKit could not parse a molecule from it."
                ),
                "evidence": {"path": path, "ext": ext},
                "suggestions": [
                    "Re-export or regenerate the reference SDF and confirm it is a complete, valid molblock.",
                ],
            }
        return {
            "error_type": "reference_wrong_format",
            "target": target,
            "stage": stage,
            "root_cause": (
                f"The reference file is '{ext or 'an unknown type'}', not a 3D SDF. The reference loader "
                "only accepts SDF/molblock with 3D coordinates, so no reference molecule could be read."
            ),
            "evidence": {"path": path, "ext": ext},
            "suggestions": [
                "Provide the reference ligand as a .sdf containing a 3D conformer.",
                "If only SMILES is available, embed a 3D conformer (e.g. with RDKit) and write it to SDF first.",
            ],
        }

    # (1) LBDD requested but no seed/reference available
    if "No legal actions are available" in msg:
        return {
            "error_type": "no_legal_actions",
            "target": target,
            "stage": stage,
            "root_cause": (
                "Only LBDD was requested, but LBDD needs a seed (a reference ligand or molecules from a "
                "previous round) and none was available at the start, so no action could be taken."
            ),
            "evidence": {"allowed_action_types": allowed, "has_reference_sdf": has_ref},
            "suggestions": [
                "Add SBDD to the action list so molecules can be generated directly from the pocket.",
                "Provide a 3D reference ligand inside the pocket to seed LBDD.",
                "Run one SBDD round first to produce seeds, then switch to LBDD.",
            ],
        }

    # (4)/(6) a required input file is missing on disk
    if isinstance(exc, FileNotFoundError):
        path = _extract_path_from_message(msg)
        if os.path.splitext(path)[1].lower() == ".pdb" or ".pdb" in msg:
            return {
                "error_type": "pdb_not_found",
                "target": target,
                "stage": stage,
                "root_cause": (
                    "The target .pdb file could not be found at the expected path "
                    "(likely an upstream env directory-name mismatch)."
                ),
                "evidence": {"path": path},
                "suggestions": [
                    "Verify the target pdb exists and that the env directory name matches the target and timestamp.",
                ],
            }
        return {
            "error_type": "sdf_not_found",
            "target": target,
            "stage": stage,
            "root_cause": "The reference ligand file does not exist at the given path.",
            "evidence": {"path": path},
            "suggestions": [
                "Check the reference ligand path and provide an existing .sdf file.",
            ],
        }

    # unknown / unhandled
    return {
        "error_type": "unknown",
        "target": target,
        "stage": stage,
        "root_cause": f"Unhandled {type(exc).__name__}: {msg}",
        "evidence": {"detail": tb_tail},
        "suggestions": ["Inspect the run logs for the full traceback."],
    }
