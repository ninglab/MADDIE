import argparse
import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

HIT_IDENTIFIER_ROOT = os.environ.get(
    "HIT_IDENTIFIER_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
CONDITAR_ROOT = os.path.join(HIT_IDENTIFIER_ROOT, "vendor", "conditar")
CONDITAR_SCRIPTS = os.path.join(CONDITAR_ROOT, "scripts", "conDitar")
for _p in [HIT_IDENTIFIER_ROOT, CONDITAR_ROOT, CONDITAR_SCRIPTS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import openbabel.openbabel as _ob
_ob.obErrorLog.SetOutputLevel(-1)

from data.sbdd_data import (
    build_dataset,
)
import utils.misc as misc  # from conDitar-dev
from utils import atom_num
import utils.transforms as trans  # from conDitar-dev
import utils.data as utils_data  # from conDitar-dev
from datasets.mol_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from sample_diffusion import sample_diffusion_ligand
from utils import reconstruct


def _resolve_config_path(config_dir, path_value):
    if not path_value or os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(config_dir, path_value))


def sample_diffusion_pocket_only(
    model,
    data,
    num_samples,
    batch_size=16,
    device="cuda:0",
    num_steps=None,
    center_pos_mode="none",
):
    all_pred_pos = []
    all_pred_v = []
    all_pred_pos_traj = []
    all_pred_v_traj = []
    all_pred_pos_cond_traj = []
    all_pred_v_cond_traj = []
    all_pred_v0_traj = []
    all_pred_vt_traj = []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))

    pocket_pos = data["pocket"].pocket_coordinate.detach().cpu().numpy()
    pocket_size = atom_num.get_space_size(pocket_pos)

    for batch_idx in range(num_batch):
        n_data = batch_size if batch_idx < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        pocket_batch = Batch.from_data_list(
            [data["pocket"].clone() for _ in range(n_data)],
            follow_batch=list(FOLLOW_BATCH),
        ).to(device)
        ligand_num_atoms = [atom_num.sample_atom_num(pocket_size, has_ligand=False) for _ in range(n_data)]
        batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
        init_ligand_pos = torch.randn(sum(ligand_num_atoms), 3).to(device)
        if model.v_mode == "gaussian":
            init_ligand_v = torch.randn(len(batch_ligand), model.num_classes).to(device)
        else:
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v = log_sample_categorical(uniform_logits)

        with torch.no_grad():
            result = model.sample_diffusion(
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                pocket_data=pocket_batch,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,
            )

        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos = result["pos"].detach().cpu().numpy().astype(np.float64)
        ligand_v = result["v"].detach().cpu()
        for sample_idx in range(n_data):
            start = ligand_cum_atoms[sample_idx]
            end = ligand_cum_atoms[sample_idx + 1]
            all_pred_pos.append(ligand_pos[start:end])
            all_pred_v.append(ligand_v[start:end])

        def _split_traj(traj):
            split = [[] for _ in range(n_data)]
            for step_value in traj:
                step_array = step_value.detach().cpu()
                for sample_idx in range(n_data):
                    start = ligand_cum_atoms[sample_idx]
                    end = ligand_cum_atoms[sample_idx + 1]
                    split[sample_idx].append(step_array[start:end])
            return split

        all_pred_pos_traj.extend(_split_traj(result["pos_traj"]))
        all_pred_v_traj.extend(_split_traj(result["v_traj"]))
        all_pred_v0_traj.extend(_split_traj(result["v0_traj"]))
        all_pred_vt_traj.extend(_split_traj(result["vt_traj"]))
        all_pred_pos_cond_traj.extend(_split_traj(result["pos_cond_traj"]))
        all_pred_v_cond_traj.extend(_split_traj(result["v_cond_traj"]))

    return (
        all_pred_pos,
        all_pred_v,
        all_pred_pos_traj,
        all_pred_v_traj,
        all_pred_v0_traj,
        all_pred_vt_traj,
        time_list,
        all_pred_pos_cond_traj,
        all_pred_v_cond_traj,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--atom_enc_mode", type=str, default="add_aromatic")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--pdb_path", type=str, required=True)
    parser.add_argument("--sdf_path", type=str, default="")
    args = parser.parse_args()

    logger = misc.get_logger("sample_sbdd")
    config = misc.load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    config.model.checkpoint = _resolve_config_path(config_dir, config.model.checkpoint)
    config.model.checkpoint_pocket = _resolve_config_path(config_dir, config.model.checkpoint_pocket)
    runtime_seed = random.SystemRandom().randrange(1, 1_000_000_000)
    misc.seed_all(runtime_seed)

    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    ligand_atom_mode = "full"
    if "transform" in ckpt["config"].data:
        ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    ligand_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])

    dataset = build_dataset(config, ligand_transform, args.pdb_path, args.sdf_path)
    data = dataset[0]
    model = ScorePosNet3D(
        ckpt["config"],
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES),
    ).to(args.device)
    model.load_state_dict(ckpt["model"], strict=False if "train_config" in config.model else True)

    sample_func = sample_diffusion_pocket_only if data["ligand"] is None else sample_diffusion_ligand
    sample_kwargs = {
        "batch_size": args.batch_size,
        "device": args.device,
        "num_steps": config.sample.num_steps,
        "center_pos_mode": config.sample.center_pos_mode,
    }
    if data["ligand"] is not None:
        sample_kwargs["sample_num_atoms"] = "pocket"

    (
        pred_pos,
        pred_v,
        pred_pos_traj,
        pred_v_traj,
        pred_v0_traj,
        pred_vt_traj,
        time_list,
        pred_pos_cond_traj,
        pred_v_cond_traj,
    ) = sample_func(
        model,
        data,
        args.num_samples,
        **sample_kwargs,
    )

    os.makedirs(args.result_path, exist_ok=True)
    if data["ligand"] is None:
        center = data["protein_pos_center"].numpy()
    else:
        center = data["ligand"].ligand_protein_pos_center.numpy()
    all_pred_ligand_pos = pred_pos_traj
    all_pred_ligand_v = pred_v_traj
    smiles_path = os.path.join(args.result_path, "SMILES.txt")
    with open(smiles_path, "w") as smiles_f:
        for sample_idx, (pred_pos_all, pred_v_all) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos_sample, pred_v_sample = pred_pos_all[-1], pred_v_all[-1]
            pred_pos_sample = pred_pos_sample + center
            pred_atom_type = trans.get_atomic_number_from_index(pred_v_sample, mode=args.atom_enc_mode)
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v_sample, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos_sample, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if "." in smiles:
                continue
            smiles_f.write(smiles + "\n")
            writer = Chem.SDWriter(os.path.join(args.result_path, f"generated_{sample_idx}.sdf"))
            writer.write(mol)
            writer.close()


if __name__ == "__main__":
    main()
