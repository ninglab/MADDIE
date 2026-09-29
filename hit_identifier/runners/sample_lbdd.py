import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from functools import partial

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch_geometric.transforms import Compose

HIT_IDENTIFIER_ROOT = os.environ.get(
    "HIT_IDENTIFIER_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DIFFSMOL_ROOT = os.path.join(HIT_IDENTIFIER_ROOT, "vendor", "diffsmol")
for _p in [HIT_IDENTIFIER_ROOT, DIFFSMOL_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openbabel.openbabel as _ob
_ob.obErrorLog.SetOutputLevel(-1)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import utils.misc as misc
import utils.transforms as trans
import utils.data as utils_data
from utils.shape import build_point_shapeAE_model, get_pointcloud_from_mol
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion_with_pocket import (
    construct_dist_mat,
    evaluate_pocket,
    get_voxel_size,
    sample_atom_nums,
    sample_diffusion_ligand,
)
from data.lbdd_data import build_dataset


def _resolve_config_path(config_dir: str, path_value: str) -> str:
    if not path_value or os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(config_dir, path_value))


def _resolve_lbdd_config_paths(config, config_dir: str):
    if "model" in config and "checkpoint" in config.model:
        config.model.checkpoint = _resolve_config_path(config_dir, config.model.checkpoint)
    if "model" in config and "train_config" in config.model:
        config.model["train_config"] = _resolve_config_path(config_dir, config.model["train_config"])
    if "data" in config and "shape" in config.data and "checkpoint" in config.data.shape:
        config.data.shape.checkpoint = _resolve_config_path(config_dir, config.data.shape.checkpoint)
    return config


def load_config_and_checkpoint(config_path: str, device: str):
    config = misc.load_config(config_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    config = _resolve_lbdd_config_paths(config, config_dir)
    misc.seed_all(config.sample.seed)

    ckpt = torch.load(config.model.checkpoint, map_location=torch.device(device))
    if "train_config" in config.model:
        train_config = misc.load_config(config.model["train_config"])
        train_config_dir = os.path.dirname(os.path.abspath(config.model["train_config"]))
        ckpt["config"] = _resolve_lbdd_config_paths(train_config, train_config_dir)
    return config, ckpt


def build_ligand_transform(ckpt):
    if "transform" in ckpt["config"].data:
        ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    else:
        ligand_atom_mode = "full"

    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    return ligand_featurizer, transform


def build_model(config, ckpt, ligand_featurizer, device: str):
    model = ScorePosNet3D(
        ckpt["config"].model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False if "train_config" in config.model else True)
    model.eval()
    return model


def build_shapeae_if_needed(config, device: str):
    if config.sample.use_grad:
        return build_point_shapeAE_model(config.data.shape, device=device, detach=False)
    return None


def build_sample_func(dist_path: str, data):
    import pickle

    dists = pickle.load(open(dist_path, "rb"))
    voxel_shape = get_voxel_size(data["ligand_smiles"])
    atom_nums = {}
    for key in dists.keys():
        if key < voxel_shape + 200 and key > voxel_shape - 200:
            atom_nums.update(dists[key])

    if len(atom_nums) == 0:
        return data["mol"].GetNumAtoms()

    atom_num_keys = list(atom_nums.keys())
    total_num = sum(atom_nums[key] for key in atom_num_keys)
    sample_atom_dist = [atom_nums[num] / total_num for num in atom_num_keys]
    return partial(sample_atom_nums, atom_nums=atom_num_keys, atom_dist=sample_atom_dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--protein_ligand_dist", type=str, default="../data/crossdocked/protein_ligand_dist.txt")
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--protein_root", type=str, required=True)
    parser.add_argument("--pdb_filename", type=str, required=True)
    parser.add_argument("--sdf_filename", type=str, required=True)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    args = parser.parse_args()
    if not os.path.isabs(args.protein_ligand_dist):
        args.protein_ligand_dist = os.path.normpath(
            os.path.join(DIFFSMOL_ROOT, args.protein_ligand_dist)
        )

    logger = misc.get_logger("sample_lbdd")
    config, ckpt = load_config_and_checkpoint(args.config, args.device)
    config.sample.num_samples = args.num_samples

    ligand_featurizer, transform = build_ligand_transform(ckpt)
    pdb_path = os.path.join(args.protein_root, args.pdb_filename)
    sdf_path = os.path.join(args.protein_root, args.sdf_filename)
    test_set = build_dataset(config, transform, pdb_path, sdf_path)
    model = build_model(config, ckpt, ligand_featurizer, args.device)
    shapeae = build_shapeae_if_needed(config, args.device)

    atom_dist_path = os.path.join(DIFFSMOL_ROOT, "data", "MOSES2", "MOSES2_training_val_shape_atomnum_dict.pkl")
    os.makedirs(args.result_path, exist_ok=True)

    for i in range(len(test_set)):
        data = test_set[i]

        if config.sample.use_pointcloud:
            data["point_cloud"] = data["point_cloud"].cpu()
            atom_pos = np.array(data["ligand_pos"])
            point_clouds = get_pointcloud_from_mol(atom_pos)
            kdtree = KDTree(point_clouds)
            use_pointcloud_data = (point_clouds, kdtree, config.sample.use_pointcloud_radius)
        else:
            use_pointcloud_data = None

        if "use_pocket" in config.sample and config.sample.use_pocket:
            data["protein_pos"] = data["protein_pos"].cpu().numpy()
            data["protein_element"] = data["protein_element"].cpu().numpy()
            kdtree = KDTree(data["protein_pos"])

            with open(args.protein_ligand_dist, "r") as f:
                protein_ligand_dist = {}
                for line in f.readlines():
                    atom1, atom2, dist = line.strip().split(" ")
                    protein_ligand_dist[(atom1, atom2)] = float(dist)
                protein_ligand_tensor = construct_dist_mat(
                    protein_ligand_dist,
                    config.data.transform.ligand_atom_mode,
                )
            use_pocket_data = (
                data["protein_pos"],
                data["protein_element"],
                kdtree,
                protein_ligand_tensor,
                config.sample.pocket_grad_step,
                config.sample.pocket_threshold,
            )
        else:
            use_pocket_data = None
            protein_ligand_dist = None

        sample_func = build_sample_func(atom_dist_path, data)
        if not config.sample.use_grad:
            config.sample.grad_lr = 0

        (
            pred_pos,
            pred_v,
            pred_bond,
            pred_pos_traj,
            pred_v_traj,
            pred_v0_traj,
            pred_vt_traj,
            time_list,
            pred_pos_cond_traj,
            pred_v_cond_traj,
        ) = sample_diffusion_ligand(
            model,
            data,
            config.sample.num_samples,
            batch_size=args.batch_size,
            device=args.device,
            threshold_type=config.sample.threshold_CFG.type if "threshold_CFG" in config.sample else None,
            threshold_args=config.sample.threshold_CFG if "threshold_CFG" in config.sample else None,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            use_grad=config.sample.use_grad,
            grad_step=config.sample.grad_step,
            grad_lr=config.sample.grad_lr,
            shape_AE=shapeae,
            use_mesh_data=None,
            use_pointcloud_data=use_pointcloud_data,
            use_pocket_data=use_pocket_data,
            use_mesh_gap=config.sample.use_mesh_gap,
            sample_num_atoms=config.sample.sample_num_atoms,
            sample_func=sample_func,
            init_scale=config.sample.init_scale,
            guide_stren=config.sample.guide_stren,
            protein_ligand_dist=protein_ligand_dist,
        )

        result = {
            "data": data,
            "pred_ligand_bond": pred_bond,
            "pred_ligand_pos": pred_pos,
            "pred_ligand_v": pred_v,
            "pred_ligand_pos_traj": pred_pos_traj,
            "pred_ligand_v_traj": pred_v_traj,
            "time": time_list,
            "pred_ligand_pos_cond_traj": pred_pos_cond_traj,
            "pred_ligand_v_cond_traj": pred_v_cond_traj,
        }
        evaluate_pocket(result, config.evaluate, args.result_path)


if __name__ == "__main__":
    main()
