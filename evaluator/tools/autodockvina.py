from typing import Literal, List, Optional

from contextlib import contextmanager

import os
import tempfile
import subprocess
import numpy as np
import pandas as pd

from vina import Vina

from rdkit import Chem
from rdkit.Chem import AllChem
from biopandas.pdb import PandasPdb


class AutoDockVina:
    def __init__(
        self,
        prepare_receptor: Literal["meeko", "mgltools"] = "meeko",
        prepare_ligand: Literal["meeko", "obabel"] = "meeko",
        cache_dir: str | None = None,
    ):
        self.cache_dir = cache_dir

        self._init_prepare_receptor(prepare_receptor)
        self._init_prepare_ligand(prepare_ligand)

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

    @contextmanager
    def _working_dir(self):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            yield self.cache_dir
        else:
            with tempfile.TemporaryDirectory() as tmp:
                yield tmp

    def prepare_receptor_meeko(self, receptor_pdb: str, receptor_pdbqt: str):
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
        subprocess.run(cmd, check=True, capture_output=False)

    def prepare_receptor_mgltools(self, receptor_pdb: str, receptor_pdbqt: str):
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

    def prepare_ligand_meeko(self, ligand_sdf: str, ligand_pdbqt: str):
        cmd = [
            "mk_prepare_ligand.py",
            "-i",
            ligand_sdf,
            "-o",
            ligand_pdbqt,
            "--rigid_macrocycles",
        ]
        subprocess.run(cmd, check=True)

    def prepare_ligand_obabel(self, ligand_sdf: str, ligand_pdbqt: str):
        cmd = [
            "obabel",
            ligand_sdf,
            "-O",
            ligand_pdbqt,
        ]
        subprocess.run(cmd, check=True)

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
        n_poses: int = 5,
        exhaustiveness: int = 8
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

            # run vina
            out_pdbqt = os.path.join(work_dir, f"{inchikey}_out.pdbqt")
            v = Vina(sf_name="vina")
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            v.compute_vina_maps(center=center, box_size=box_size)
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            v.write_poses(out_pdbqt, n_poses=n_poses, overwrite=True)
            score = v.energies(n_poses=1)[0][0]

            return score

def run_autodockvina(
    smiles: str,
    pdb_path: str,
    grid_size: int | List[int] | None = 20,
    n_poses: int = 5,
    exhaustiveness: int = 8,
    prepare_receptor: Literal["meeko", "mgltools"] = "meeko",
    prepare_ligand: Literal["meeko", "obabel"] = "meeko",
    cache_dir: str | None = None,
) -> Optional[float]:
    vina = AutoDockVina(
        prepare_receptor=prepare_receptor,
        prepare_ligand=prepare_ligand,
        cache_dir=cache_dir,
    )
    return vina.dock(
        smi=smiles,
        pdb_path=pdb_path,
        grid_size=grid_size,
        n_poses=n_poses,
        exhaustiveness=exhaustiveness,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoDock Vina docking from the command line.")
    parser.add_argument("smiles", type=str, help="SMILES string of the ligand")
    parser.add_argument("receptor", type=str, help="Path to the receptor PDB file")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Exhaustiveness of the search (default: 8)")
    parser.add_argument("--n_poses", type=int, default=5, help="Number of poses to generate (default: 5)")
    parser.add_argument("--grid_size", type=int, default=20, help="Docking grid size in Angstroms (default: 20)")
    parser.add_argument("--prepare_receptor", choices=["meeko", "mgltools"], default="meeko", help="Receptor preparation method (default: meeko)")
    parser.add_argument("--prepare_ligand", choices=["meeko", "obabel"], default="meeko", help="Ligand preparation method (default: meeko)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache intermediate files (default: temporary directory)")
    args = parser.parse_args()

    score = run_autodockvina(
        smiles=args.smiles,
        pdb_path=args.receptor,
        grid_size=args.grid_size,
        n_poses=args.n_poses,
        exhaustiveness=args.exhaustiveness,
        prepare_receptor=args.prepare_receptor,
        prepare_ligand=args.prepare_ligand,
        cache_dir=args.cache_dir,
    )
    print(f"Docking score: {score:.3f} kcal/mol")
