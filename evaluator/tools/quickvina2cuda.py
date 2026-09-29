from typing import Literal, List, Optional

from contextlib import contextmanager

import os
import tempfile
import subprocess
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb


class QuickVina2Cuda:
    def __init__(
        self,
        prepare_receptor: Literal["meeko", "mgltools"] = "mgltools",
        prepare_ligand: Literal["meeko", "obabel"] = "meeko",
        path_to_tool: str = "/path/to/Vina-CUDA/QuickVina2-CUDA",
        path_to_tool_environment: str = "/path/to/envs/vina-cuda",
        thread: int = 8192,
        cache_dir: str | None = None,
    ):
        self.path_to_tool = path_to_tool
        self.path_to_tool_environment = path_to_tool_environment
        self.thread = thread
        self.cache_dir = cache_dir

        self._init_prepare_receptor(prepare_receptor)
        self._init_prepare_ligand(prepare_ligand)

    # region init
    def _init_prepare_receptor(self, method: str):
        if method == "meeko":
            self.prepare_receptor = self.prepare_receptor_meeko
        elif method == "mgltools":
            self.prepare_receptor = self.prepare_receptor_mgltools
        else:
            raise ValueError(f"'prepare_receptor' is not recognized: '{method}'")

    def _init_prepare_ligand(self, method: str):
        if method == "meeko":
            self.prepare_ligand = self.prepare_ligand_meeko
        elif method == "obabel":
            self.prepare_ligand = self.prepare_ligand_obabel
        else:
            raise ValueError(f"'prepare_ligand' is not recognized: '{method}'")
    # endregion

    @contextmanager
    def _working_dir(self):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            yield self.cache_dir
        else:
            with tempfile.TemporaryDirectory() as tmp:
                yield tmp

    @staticmethod
    def extract_vina_score(pdbqt_file: str) -> Optional[float]:
        with open(pdbqt_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    return float(line.split()[3])
        return None

    # region prepare receptor
    def prepare_receptor_meeko(self, receptor_pdb: str, receptor_pdbqt: str) -> None:
        cmd = [
            "mk_prepare_receptor.py",
            "-i",
            receptor_pdb,
            "-p",
            "-a",
            "--delete_residues",
            "HOH",
            "-o",
            receptor_pdbqt.replace(".pdbqt", ""),
            "--default_altloc",
            "A",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def prepare_receptor_mgltools(self, receptor_pdb: str, receptor_pdbqt: str) -> None:
        cmd = [
            "conda",
            "run",
            "-n",
            "mgltools",
            "prepare_receptor4.py",
            "-r",
            receptor_pdb,
            "-o",
            receptor_pdbqt,
        ]
        subprocess.run(cmd, check=True)
    # endregion prepare receptor

    # region prepare ligand
    def prepare_ligand_meeko(self, ligand_sdf: str, ligand_pdbqt: str) -> None:
        cmd = [
            "mk_prepare_ligand.py",
            "-i",
            ligand_sdf,
            "-o",
            ligand_pdbqt,
            "--rigid_macrocycles",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def prepare_ligand_obabel(self, ligand_sdf: str, ligand_pdbqt: str) -> None:
        cmd = [
            "obabel",
            ligand_sdf,
            "-O",
            ligand_pdbqt,
        ]
        subprocess.run(cmd, check=True)
    # endregion

    def smiles_to_sdf(
        self,
        smi: str,
        center: list,
        ligand_sdf: str
    ):
        mol = Chem.MolFromSmiles(smi) # pylint: disable=E1101
        mol = Chem.AddHs(mol, addCoords=True) # pylint: disable=E1101
        AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # pylint: disable=E1101
        AllChem.MMFFOptimizeMolecule(mol) # pylint: disable=E1101

        conf = mol.GetConformer()
        ligand_center = np.mean(
            [conf.GetAtomPosition(j) for j in range(mol.GetNumAtoms())],
            axis=0,
        )
        translation = center - ligand_center

        for j in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(j)
            conf.SetAtomPosition(
                j,
                (
                    pos.x + translation[0],
                    pos.y + translation[1],
                    pos.z + translation[2],
                ),
            )

        writer = Chem.SDWriter(ligand_sdf) # pylint: disable=E1101
        writer.write(mol)
        writer.close()

    def dock(
        self,
        smi: str,
        pdb_path: str,
        grid_size: int | List[int] | None = 20,
    ):
        MIN_GRID_SIZE = 20

        with self._working_dir() as work_dir:
            # set receptor box size
            ppdb = PandasPdb().read_pdb(pdb_path)
            coords = ppdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]]
            center = coords.mean().values.tolist()

            # set docking box size
            if isinstance(grid_size, int):
                box_size = [grid_size, grid_size, grid_size]
            elif isinstance(grid_size, list):
                box_size = grid_size
            else:
                raw_size = coords.max().values - coords.min().values + 10 # use 10 as buffer
                box_size = np.minimum(raw_size, MIN_GRID_SIZE).tolist()

            receptor_pdbqt = os.path.join(work_dir, "receptor.pdbqt")
            self.prepare_receptor(pdb_path, receptor_pdbqt)

            inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

            ligand_sdf = os.path.join(work_dir, f"{inchikey}_ligand.sdf")
            ligand_pdbqt = os.path.join(work_dir, f"{inchikey}_ligand.pdbqt")

            self.smiles_to_sdf(smi, center, ligand_sdf)
            self.prepare_ligand(ligand_sdf, ligand_pdbqt)

            # run quickvina2-cuda
            out_pdbqt = os.path.join(work_dir, f"{inchikey}_out.pdbqt")
            binary = os.path.join(self.path_to_tool, "QuickVina2-GPU-2-1-CUDA")
            command = (
                f'"{binary}"'
                f' --ligand "{ligand_pdbqt}"'
                f' --receptor "{receptor_pdbqt}"'
                f' --thread {self.thread}'
                f' --center_x {center[0]:.3f}'
                f' --center_y {center[1]:.3f}'
                f' --center_z {center[2]:.3f}'
                f' --size_x {box_size[0]:.1f}'
                f' --size_y {box_size[1]:.1f}'
                f' --size_z {box_size[2]:.1f}'
                f' --out "{out_pdbqt}"'
            )
            subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL)
            score = self.extract_vina_score(out_pdbqt)

            return score

def run_quickvina2cuda(
    smiles: str,
    pdb_path: str,
    grid_size: int | List[int] | None = 20,
    prepare_receptor: Literal["meeko", "mgltools"] = "meeko",
    prepare_ligand: Literal["meeko", "obabel"] = "meeko",
    path_to_tool: str = "/path/to/Vina-CUDA/QuickVina2-CUDA",
    path_to_tool_environment: str = "/path/to/envs/maddie-eval",
    thread: int = 8192,
    cache_dir: str | None = None,
) -> Optional[float]:
    vina = QuickVina2Cuda(
        prepare_receptor=prepare_receptor,
        prepare_ligand=prepare_ligand,
        path_to_tool=path_to_tool,
        path_to_tool_environment=path_to_tool_environment,
        thread=thread,
        cache_dir=cache_dir,
    )
    return vina.dock(
        smi=smiles,
        pdb_path=pdb_path,
        grid_size=grid_size,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run QuickVina2-CUDA docking from the command line.")
    parser.add_argument("smiles", type=str, help="SMILES string of the ligand")
    parser.add_argument("receptor", type=str, help="Path to the receptor PDB file")
    parser.add_argument("--grid_size", type=int, default=20, help="Docking grid size in Angstroms (default: 20)")
    parser.add_argument("--path_to_tool", type=str, default="./Vina-CUDA/QuickVina2-CUDA", help="Path to the QuickVina2-CUDA tool directory")
    parser.add_argument("--path_to_tool_environment", type=str, default="/path/to/envs/vina-cuda", help="Path to the conda environment for QuickVina2-CUDA")
    parser.add_argument("--thread", type=int, default=8192, help="Number of GPU threads (default: 8192)")
    parser.add_argument("--prepare_receptor", choices=["meeko", "mgltools"], default="meeko", help="Receptor preparation method (default: meeko)")
    parser.add_argument("--prepare_ligand", choices=["meeko", "obabel"], default="meeko", help="Ligand preparation method (default: meeko)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache intermediate files (default: temporary directory)")
    args = parser.parse_args()

    score = run_quickvina2cuda(
        smiles=args.smiles,
        pdb_path=args.receptor,
        grid_size=args.grid_size,
        prepare_receptor=args.prepare_receptor,
        prepare_ligand=args.prepare_ligand,
        path_to_tool=args.path_to_tool,
        path_to_tool_environment=args.path_to_tool_environment,
        thread=args.thread,
        cache_dir=args.cache_dir,
    )
    print(f"Docking score: {score:.3f} kcal/mol")
