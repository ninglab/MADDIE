import os
import sys

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit import Geometry
from rdkit.Chem.rdchem import BondType
from torch.utils.data import Dataset
from torch_geometric.data import Batch


_HIT_IDENTIFIER_ROOT = os.environ.get(
    "HIT_IDENTIFIER_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
CONDITAR_ROOT = os.path.join(_HIT_IDENTIFIER_ROOT, "vendor", "conditar")
if CONDITAR_ROOT not in sys.path:
    sys.path.insert(0, CONDITAR_ROOT)

from datasets.mol_data import LigandData, PocketData, parse_ligand_pocket, parse_protein_pocket, torchify_dict
from models.pocket_modelAE import PocketModel


ATOM_FAMILIES = [
    "Acceptor",
    "Donor",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
]
ATOM_FAMILIES_ID = {name: idx for idx, name in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
AA_NAME_NUMBER = {
    name: idx
    for idx, name in enumerate([
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
    ])
}
UNKNOWN_AA_INDEX = len(AA_NAME_NUMBER)
BACKBONE_NAMES = {"CA", "C", "N", "O"}


def _torchify_dict(data):
    output = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            output[key] = torch.from_numpy(value)
            if output[key].dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
                output[key] = output[key].to(torch.float32)
        elif isinstance(value, float):
            output[key] = torch.tensor(value, dtype=torch.float32)
        else:
            output[key] = value
    return output


def _parse_pdb_all_atoms(pdb_path: str):
    ptable = Chem.GetPeriodicTable()
    element = []
    pos = []
    is_backbone = []
    atom_name = []
    atom_to_aa_type = []

    with open(pdb_path, "r") as handle:
        for line in handle:
            record = line[0:6].strip()
            if record == "ENDMDL":
                break
            if record not in {"ATOM", "HETATM"}:
                continue

            name = line[12:16].strip()
            res_name = line[17:20].strip()
            element_sym = line[76:78].strip().capitalize() or name[0].capitalize()
            atomic_number = ptable.GetAtomicNumber(element_sym)
            if atomic_number <= 0:
                continue

            element.append(atomic_number)
            pos.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            is_backbone.append(name in BACKBONE_NAMES)
            atom_name.append(name)
            atom_to_aa_type.append(AA_NAME_NUMBER.get(res_name, UNKNOWN_AA_INDEX))

    if not pos:
        raise ValueError(f"No atoms could be parsed from PDB: {pdb_path}")

    return {
        "element": np.asarray(element, dtype=np.int64),
        "molecule_name": os.path.basename(pdb_path),
        "pos": np.asarray(pos, dtype=np.float32),
        "is_backbone": np.asarray(is_backbone, dtype=np.bool_),
        "atom_name": atom_name,
        "atom_to_aa_type": np.asarray(atom_to_aa_type, dtype=np.int64),
    }


def _center_mol(mol):
    conformer = mol.GetConformer()
    pos = np.asarray(conformer.GetPositions(), dtype=np.float32)
    center = pos.mean(axis=0)
    for atom in mol.GetAtoms():
        atom_pos = conformer.GetAtomPosition(atom.GetIdx())
        conformer.SetAtomPosition(
            atom.GetIdx(),
            Geometry.Point3D(
                float(atom_pos.x - center[0]),
                float(atom_pos.y - center[1]),
                float(atom_pos.z - center[2]),
            ),
        )
    return center


def _load_ligand_mol(sdf_path: str):
    last_exc = None
    for sanitize in (True, False):
        try:
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=sanitize)
            mol = next((item for item in supplier if item is not None), None)
        except Exception as exc:
            last_exc = exc
            mol = None
        if mol is not None and mol.GetNumConformers() > 0:
            return mol
    if last_exc is not None:
        raise last_exc
    raise ValueError(f"Failed to read ligand molecule from {sdf_path}")


def _parse_rdkit_mol(mol, center=None):
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    num_atoms = mol.GetNumAtoms()
    feat_mat = np.zeros([num_atoms, len(ATOM_FAMILIES)], dtype=np.int64)
    for feat in factory.GetFeaturesForMol(mol):
        feat_mat[list(feat.GetAtomIds()), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    hybridization = []
    element = []
    for atom in mol.GetAtoms():
        hybridization.append(str(atom.GetHybridization()))
        element.append(atom.GetAtomicNum())

    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += [BOND_TYPES[bond.GetBondType()]] * 2

    edge_index = np.asarray([rows, cols], dtype=np.int64)
    edge_type = np.asarray(edge_types, dtype=np.int64)
    if edge_index.size > 0:
        perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_type = np.empty((0,), dtype=np.int64)

    return {
        "smiles": Chem.MolToSmiles(mol),
        "element": np.asarray(element, dtype=np.int32),
        "pos": pos,
        "bond_index": edge_index,
        "bond_type": edge_type,
        "atom_feature": feat_mat,
        "hybridization": hybridization,
        "center": center,
    }


def _parse_sdf_file(sdf_path: str):
    mol = _load_ligand_mol(sdf_path)
    center = _center_mol(mol)
    return _parse_rdkit_mol(mol, center=center), mol


class FixedPocketDataset(Dataset):
    def __init__(self, config, files, ligand_transform):
        self.config = config
        self.files = files
        self.hydrogen = config.data.hydrogen
        self.ligand_transform = ligand_transform
        self.ckpt_path = config.model.checkpoint_pocket
        self._build_model()

    def _build_model(self):
        self.model = PocketModel(config=self.config.pocket, hydrogen=self.config.data.hydrogen)
        checkpoint = torch.load(self.ckpt_path)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except Exception:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pdb_path, sdf_path = self.files[idx]
        protein_dict = _parse_pdb_all_atoms(pdb_path)
        ligand_dict = _parse_sdf_file(sdf_path)[0] if sdf_path else None

        data = {}
        for key, value in protein_dict.items():
            data[f"protein_{key}"] = value
        if ligand_dict is not None:
            for key, value in ligand_dict.items():
                data[f"ligand_{key}"] = value

        pocket_mean = data["protein_pos"].mean(0)
        data_pocket = parse_protein_pocket(data, hydrogen=self.hydrogen)

        pocket_data = PocketData.from_pocket_dicts(torchify_dict(data_pocket))
        batch = Batch.from_data_list([pocket_data])
        with torch.no_grad():
            embd_scalar, embd_vector = self.model.get_embds(batch)

        data_pocket["embd_scalar"] = np.asarray(embd_scalar)
        data_pocket["embd_vector"] = np.asarray(embd_vector)

        pocket_data = PocketData.from_pocket_dicts(torchify_dict(data_pocket))
        ligand_data = None
        if ligand_dict is not None:
            data_ligand = parse_ligand_pocket(_torchify_dict(data))
            data_ligand["filename"] = sdf_path
            data_ligand["protein_pos_center"] = pocket_mean
            ligand_data = LigandData.from_ligand_dicts(torchify_dict(data_ligand))
            if self.ligand_transform is not None:
                ligand_data = self.ligand_transform(ligand_data)

        pocket_data.id = idx
        if ligand_data is not None:
            ligand_data.id = idx
        return {"pocket": pocket_data, "ligand": ligand_data, "protein_pos_center": torch.from_numpy(pocket_mean)}


def build_dataset(config, ligand_transform, pdb_path: str, sdf_path: str = ""):
    files = [(pdb_path, sdf_path or "")]
    return FixedPocketDataset(config=config, files=files, ligand_transform=ligand_transform)
