import contextlib
import hashlib
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
from typing import Dict, Optional

_unidock_path = os.environ.get("UNIDOCK_PATH")
if _unidock_path:
    sys.path.insert(0, _unidock_path)
from unidock import dock as _unidock

_DEFAULT_TMP_DIR = os.environ.get("MADDIE_TMP_DIR", tempfile.gettempdir())

import numpy as np
import rdkit.Chem as Chem
from meeko import MoleculePreparation
from openbabel import openbabel as _ob, pybel
_ob.obErrorLog.SetOutputLevel(-1)
from rdkit.Chem import AllChem
from vina import Vina
import AutoDockTools


if not hasattr(np, "int"):
    np.int = int


def get_random_id(length: int = 30) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def normalize_dataset_relative_path(root_dir: str, rel_path: str) -> str:
    if rel_path is None:
        return None
    rel_path = os.path.normpath(str(rel_path))
    if os.path.isabs(rel_path):
        return rel_path
    root_dir = os.path.normpath(str(root_dir))
    root_base = os.path.basename(root_dir)
    prefix = root_base + os.sep
    if rel_path == root_base:
        return "."
    if rel_path.startswith(prefix):
        return rel_path[len(prefix):]
    return rel_path


def _default_test_protein_name(ligand_filename: str) -> str:
    ligand_base = os.path.basename(ligand_filename)
    return ligand_base[:4] + "_protein.pdb"


def suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class BaseDockingTask:
    def __init__(self, protein_path, ligand_rdmol):
        self.protein_path = protein_path
        self.ligand_rdmol = ligand_rdmol

    def run(self, *args, **kwargs):
        raise NotImplementedError()


class PrepLig:
    def __init__(self, input_mol, mol_format):
        if mol_format == "smi":
            self.ob_mol = pybel.readstring("smi", input_mol)
        elif mol_format == "sdf":
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f"mol_format {mol_format} not supported")

    @suppress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return None
        return preparator.write_pdbqt_string()


class PrepProt:
    def __init__(self, pdb_file):
        self.prot = pdb_file
        self.prot_pqr = None

    def addH(self, prot_pqr):
        self.prot_pqr = prot_pqr
        pdb2pqr_cmd = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
        if pdb2pqr_cmd is None:
            raise RuntimeError("Could not find 'pdb2pqr30' or 'pdb2pqr' in PATH while preparing receptor.")
        result = subprocess.run(
            [pdb2pqr_cmd, "--ff=AMBER", self.prot, self.prot_pqr],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.exists(self.prot_pqr):
            raise RuntimeError(
                "Failed to generate receptor PQR file. "
                f"command={pdb2pqr_cmd} | input={self.prot} | output={self.prot_pqr} | "
                f"returncode={result.returncode} | stderr={result.stderr.strip()} | stdout={result.stdout.strip()}"
            )

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py")
        result = subprocess.run(
            [sys.executable, prepare_receptor, "-r", self.prot_pqr, "-o", prot_pdbqt],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not os.path.exists(prot_pdbqt):
            raise RuntimeError(
                "Failed to generate receptor PDBQT file. "
                f"python={sys.executable} | script={prepare_receptor} | input={self.prot_pqr} | output={prot_pdbqt} | "
                f"returncode={result.returncode} | stderr={result.stderr.strip()} | stdout={result.stdout.strip()}"
            )


class VinaDock:
    def __init__(self, lig_pdbqt, prot_pdbqt):
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
        self.pocket_center = None
        self.box_size = None

    def dock(self, score_func="vina", seed=0, mode="dock", exhaustiveness=16, save_pose=False, **kwargs):
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == "score_only":
            score = v.score()[0]
        elif mode == "minimize":
            score = v.optimize()[0]
        elif mode == "dock":
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError(mode)

        if not save_pose:
            return score

        if mode == "score_only":
            pose = None
        elif mode == "minimize":
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                dir=os.path.dirname(os.path.abspath(self.lig_pdbqt)),
            )
            tmp.close()
            v.write_pose(tmp.name, overwrite=True)
            with open(tmp.name, "r") as f:
                pose = f.read()
            os.remove(tmp.name)
        else:
            pose = v.poses(n_poses=1)
            v.write_poses(self.lig_pdbqt, n_poses=1, overwrite=True)
        return score, pose, self.lig_pdbqt


def _normalize_box_size(box_size):
    """Accept a scalar (cubic box) or a 3-element sequence; return [x, y, z] floats."""
    if isinstance(box_size, (int, float)):
        return [float(box_size), float(box_size), float(box_size)]
    return [float(box_size[0]), float(box_size[1]), float(box_size[2])]


def _extract_vina_result_from_pdbqt(pdbqt_file: str) -> Optional[float]:
    """Read the best 'REMARK VINA RESULT' affinity from a QuickVina output PDBQT."""
    with open(pdbqt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                return float(line.split()[3])
    return None


def run_quickvina_gpu(
    lig_pdbqt: str,
    prot_pdbqt: str,
    center,
    box_size,
    out_pdbqt: str,
    gpu_config: Dict,
) -> float:
    """
    Dock an already-prepared ligand PDBQT with QuickVina2-GPU (CUDA) and return
    the best affinity. The ligand PDBQT is prepared upstream from the molecule's
    real SDF pose (via PrepLig), so no SMILES->3D regeneration happens here; the
    docking search is localized by the caller-provided center/box.
    """
    path_to_tool = gpu_config.get(
        "path_to_tool", os.environ.get("VINA_CUDA_DIR", "/path/to/Vina-CUDA/QuickVina2-CUDA")
    )
    env = gpu_config.get(
        "path_to_tool_environment",
        os.environ.get("VINA_CUDA_ENV", "/path/to/envs/vina-cuda"),
    )
    thread = int(gpu_config.get("thread", 8192))
    binary = gpu_config.get("binary", "QuickVina2-GPU-2-1-CUDA")
    binary_path = os.path.join(path_to_tool, binary)

    command = (
        "conda init;"
        " . ~/.bashrc;"
        f" conda activate {env};"
        f' "{binary_path}"'
        f' --ligand "{lig_pdbqt}"'
        f' --receptor "{prot_pdbqt}"'
        f" --thread {thread}"
        f" --center_x {center[0]:.3f} --center_y {center[1]:.3f} --center_z {center[2]:.3f}"
        f" --size_x {box_size[0]:.1f} --size_y {box_size[1]:.1f} --size_z {box_size[2]:.1f}"
        f' --out "{out_pdbqt}"'
    )
    # bash (not /bin/sh) so `conda activate` / `. ~/.bashrc` resolve correctly.
    result = subprocess.run(
        command, shell=True, executable="/bin/bash", capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(out_pdbqt):
        raise RuntimeError(
            "QuickVina2-GPU docking failed. "
            f"binary={binary_path} | env={env} | returncode={result.returncode} | "
            f"stderr={result.stderr.strip()} | stdout={result.stdout.strip()}"
        )
    score = _extract_vina_result_from_pdbqt(out_pdbqt)
    if score is None:
        raise RuntimeError(f"QuickVina2-GPU produced no VINA RESULT in {out_pdbqt}")
    return float(score)


class VinaDockingTask(BaseDockingTask):
    @classmethod
    def from_generated_mol(
        cls,
        ligand_rdmol,
        ligand_filename,
        protein_filename=None,
        protein_root="../data/crossdocked",
        test=True,
        **kwargs,
    ):
        if protein_filename is None:
            if test:
                protein_fn = os.path.join(
                    os.path.dirname(ligand_filename),
                    _default_test_protein_name(ligand_filename),
                )
            else:
                protein_fn = os.path.join(
                    os.path.dirname(ligand_filename),
                    os.path.basename(ligand_filename)[:-4] + "_pocket10.pdb",
                )
        else:
            protein_fn = protein_filename
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(
        self,
        protein_path,
        ligand_rdmol,
        tmp_dir=_DEFAULT_TMP_DIR,
        center=None,
        box_size=None,
        size_factor=1.0,
        buffer=5.0,
        receptor_cache_dir=None,
        gpu_config=None,
    ):
        super().__init__(protein_path, ligand_rdmol)
        self.gpu_config = gpu_config
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        if receptor_cache_dir is None:
            receptor_cache_dir = os.path.join(self.tmp_dir, "receptor_cache")
        self.receptor_cache_dir = os.path.realpath(receptor_cache_dir)
        os.makedirs(self.receptor_cache_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.task_id + "_ligand.sdf")

        ligand_rdmol = Chem.Mol(ligand_rdmol)
        if ligand_rdmol.GetNumConformers() == 0:
            ligand_rdmol = Chem.AddHs(ligand_rdmol)
            AllChem.EmbedMolecule(ligand_rdmol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(ligand_rdmol)
        else:
            ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        self.center = (pos.max(0) + pos.min(0)) / 2 if center is None else center
        # Box size precedence: explicit box_size arg > docking.gpu.box_size (config,
        # cubic scalar or [x,y,z]) > ligand-pose bounding box * size_factor + buffer.
        # The box always stays centered on the ligand pose above.
        if box_size is None and gpu_config:
            box_size = gpu_config.get("box_size")
        if box_size is not None:
            self.size_x, self.size_y, self.size_z = _normalize_box_size(box_size)
        elif size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

    def _get_receptor_cache_paths(self):
        receptor_base = os.path.splitext(os.path.basename(self.receptor_path))[0]
        receptor_hash = hashlib.md5(self.receptor_path.encode("utf-8")).hexdigest()[:12]
        cache_prefix = os.path.join(self.receptor_cache_dir, f"{receptor_base}_{receptor_hash}")
        return cache_prefix + ".pqr", cache_prefix + ".pdbqt"

    def run(self, mode="dock", exhaustiveness=16, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + ".pdbqt"
        docked_sdf_path = self.ligand_path[:-4] + "_docked.sdf"
        protein_pqr, protein_pdbqt = self._get_receptor_cache_paths()

        lig = PrepLig(self.ligand_path, "sdf")
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            prot.addH(protein_pqr)
        else:
            prot.prot_pqr = protein_pqr

        if not os.path.exists(protein_pdbqt):
            prot.get_pdbqt(protein_pdbqt)

        if self.gpu_config:
            # GPU path: QuickVina2-CUDA re-docks the ligand from its prepared
            # (pose-preserving) PDBQT within the pose-centered box. There is no
            # score_only mode, so this always reports a docked ("docking score")
            # affinity plus the docked pose.
            gpu_out_pdbqt = self.ligand_path[:-4] + "_gpu_out.pdbqt"
            score = run_quickvina_gpu(
                lig_pdbqt=ligand_pdbqt,
                prot_pdbqt=protein_pdbqt,
                center=self.center,
                box_size=[self.size_x, self.size_y, self.size_z],
                out_pdbqt=gpu_out_pdbqt,
                gpu_config=self.gpu_config,
            )
            if os.path.exists(gpu_out_pdbqt):
                docked_mol = next(pybel.readfile("pdbqt", gpu_out_pdbqt))
                # Generated molecules are stored without explicit H; strip the
                # AutoDock (polar) hydrogens the docking output carries so the saved
                # docked molecule.sdf keeps the same heavy-atom-only convention.
                docked_mol.OBMol.DeleteHydrogens()
                docked_mol.write("sdf", docked_sdf_path, overwrite=True)
            else:
                docked_sdf_path = ""
            return [{"affinity": score, "pose": None, "docked_sdf_path": docked_sdf_path}]

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center = self.center
        dock.box_size = [self.size_x, self.size_y, self.size_z]
        score, pose, _ = dock.dock(
            score_func="vina",
            mode=mode,
            exhaustiveness=exhaustiveness,
            save_pose=True,
            **kwargs,
        )
        if pose is not None and os.path.exists(ligand_pdbqt):
            docked_mol = next(pybel.readfile("pdbqt", ligand_pdbqt))
            docked_mol.write("sdf", docked_sdf_path, overwrite=True)
        else:
            docked_sdf_path = ""
        return [{"affinity": score, "pose": pose, "docked_sdf_path": docked_sdf_path}]


def _resolve_gpu_config(target_resource: Optional[Dict]) -> Optional[Dict]:
    """
    Resolve the docking engine from target_resource["docking"].
      "vina_score"    -> None  (CPU AutoDock Vina; score_only for scoring, dock for seeds)
      "docking_score" -> the GPU config dict (QuickVina2-CUDA)
    Raises on an unknown mode or a missing GPU config.
    """
    if not target_resource:
        return None
    docking_cfg = target_resource.get("docking", {}) or {}
    scoring_mode = str(docking_cfg.get("scoring_mode", "vina_score")).strip().lower()
    if scoring_mode == "vina_score":
        return None
    if scoring_mode == "docking_score":
        gpu_config = docking_cfg.get("gpu")
        if not gpu_config:
            raise ValueError(
                "docking.scoring_mode is 'docking_score' but docking.gpu config is missing."
            )
        return gpu_config
    raise ValueError(
        f"Unsupported docking.scoring_mode: {scoring_mode!r} "
        "(expected 'vina_score' or 'docking_score')."
    )


def compute_vina_affinity(gen_mol, ligand_filename: str, protein_root: str, tmp_dir: str, exhaustiveness: int, gpu_config: Optional[Dict] = None) -> float:
    ligand_filename = normalize_dataset_relative_path(protein_root, ligand_filename)
    vina_task = VinaDockingTask.from_generated_mol(
        gen_mol,
        ligand_filename,
        protein_root=protein_root,
        test=True,
        tmp_dir=tmp_dir,
        gpu_config=gpu_config,
    )
    # gpu_config takes over the engine inside run(); mode is ignored on the GPU path.
    vina_results = vina_task.run(mode="score_only", exhaustiveness=exhaustiveness)
    return float(vina_results[0]["affinity"]), vina_results[0].get("docked_sdf_path", "")


def _compute_via_explicit_target(
    mol,
    pdb_path: str,
    sdf_path: str,
    tmp_dir: str,
    exhaustiveness: int,
    gpu_config: Optional[Dict] = None,
) -> float:
    protein_root = os.path.dirname(os.path.abspath(pdb_path))
    ligand_filename = os.path.basename(sdf_path)
    protein_filename = os.path.basename(pdb_path)
    vina_task = VinaDockingTask.from_generated_mol(
        mol,
        ligand_filename,
        protein_filename=protein_filename,
        protein_root=protein_root,
        test=True,
        tmp_dir=tmp_dir,
        gpu_config=gpu_config,
    )
    # gpu_config takes over the engine inside run(); mode is ignored on the GPU path.
    vina_results = vina_task.run(mode="score_only", exhaustiveness=exhaustiveness)
    return float(vina_results[0]["affinity"]), vina_results[0].get("docked_sdf_path", "")


def dock_mol_to_sdf(
    mol,
    target_resource: Dict,
    output_sdf_path: str,
    ligand_sdf_path: str = None,
    center=None,
    box_size=None,
    tmp_dir: str = None,
    exhaustiveness: int = 16,
):
    tmp_dir = tmp_dir or target_resource.get("tmp_root", _DEFAULT_TMP_DIR)
    protein_path = target_resource["pdb"]
    sdf_path = ligand_sdf_path or target_resource.get("sdf")
    if not sdf_path:
        raise ValueError("dock_mol_to_sdf requires ligand_sdf_path when target_resource has no reference SDF")
    protein_root = os.path.dirname(os.path.abspath(protein_path))
    ligand_filename = os.path.basename(sdf_path)
    protein_filename = os.path.basename(protein_path)
    # In docking_score mode the LBDD seed is docked on the GPU (QuickVina2-CUDA);
    # otherwise CPU AutoDock Vina (mode="dock"). Either way the seed's real SDF
    # pose is used to prepare the ligand.
    gpu_config = _resolve_gpu_config(target_resource)
    vina_task = VinaDockingTask.from_generated_mol(
        mol,
        ligand_filename,
        protein_filename=protein_filename,
        protein_root=protein_root,
        test=True,
        tmp_dir=tmp_dir,
        center=center,
        box_size=box_size,
        gpu_config=gpu_config,
    )
    vina_results = vina_task.run(mode="dock", exhaustiveness=exhaustiveness)
    docked_sdf_path = vina_results[0].get("docked_sdf_path", "")
    if not docked_sdf_path or not os.path.exists(docked_sdf_path):
        raise FileNotFoundError("Docking completed without producing a docked SDF pose")
    os.makedirs(os.path.dirname(os.path.abspath(output_sdf_path)), exist_ok=True)
    shutil.copy2(docked_sdf_path, output_sdf_path)
    return output_sdf_path, float(vina_results[0]["affinity"])


def compute_vina_score(
    mol,
    protein_path: Optional[str] = None,
    target_resource: Optional[Dict] = None,
    ligand_sdf_path: Optional[str] = None,
    tmp_dir: str = None,
    exhaustiveness: int = 16,
):
    if target_resource is not None:
        tmp_dir = tmp_dir or target_resource.get("tmp_root", _DEFAULT_TMP_DIR)
        protein_path = target_resource["pdb"]
        sdf_path = target_resource.get("sdf") or ligand_sdf_path
        if not sdf_path:
            raise ValueError(
                "compute_vina_score requires target_resource.sdf or ligand_sdf_path "
                "to name the ligand input for docking."
            )
        # Scoring mode selection:
        #   "vina_score"    -> CPU AutoDock Vina score_only (preserved).
        #   "docking_score" -> Uni-Dock (GPU, reproducible via fixed seed).
        gpu_config = _resolve_gpu_config(target_resource)

        if gpu_config is not None:
            smiles = Chem.MolToSmiles(mol)
            score = _unidock(smiles, protein_path, seed=42)
            return float(score) if score is not None else float("inf"), ""

        protein_root = os.path.dirname(os.path.abspath(sdf_path))
        ligand_filename = os.path.basename(sdf_path)
        expected_protein = _default_test_protein_name(ligand_filename)

        if (
            os.path.abspath(os.path.dirname(protein_path)) == protein_root
            and os.path.basename(protein_path) == expected_protein
        ):
            return compute_vina_affinity(
                gen_mol=mol,
                ligand_filename=ligand_filename,
                protein_root=protein_root,
                tmp_dir=tmp_dir,
                exhaustiveness=exhaustiveness,
                gpu_config=None,
            )
        return _compute_via_explicit_target(
            mol=mol,
            pdb_path=protein_path,
            sdf_path=sdf_path,
            tmp_dir=tmp_dir,
            exhaustiveness=exhaustiveness,
            gpu_config=gpu_config,
        )

    if protein_path is None:
        raise ValueError("Either protein_path or target_resource must be provided")
    tmp_dir = tmp_dir or _DEFAULT_TMP_DIR
    raise ValueError(
        "compute_vina_score now expects target_resource for SBDD-style scoring. "
        "Pass the resolved target_resource dict instead of only protein_path."
    )
