import pdb
import os, sys
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import shutil
from glob import glob
import time
import pickle
import utils.misc as misc
import utils.transforms as trans
from utils.shape import get_voxel_shape, get_pointcloud_from_mesh, get_pointcloud_from_mol, get_mesh, get_atom_stamp, build_point_shapeAE_model
from utils.reconstruct import reconstruct_from_generated
import utils.data as utils_data
from datasets import get_dataset
from functools import partial
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean
from datasets.shape_mol_data import FOLLOW_BATCH
import trimesh
from sklearn.neighbors import KDTree
import numpy as np
from utils.docking_vina import VinaDockingTask

from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from preprocess.mose_training_val_dataset_generation import conformer_generation
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from utils import scoring_func
from utils import reconstruct
from utils import transforms
from utils.docking import QVinaDockingTask
import multiprocessing as mp

def get_voxel_size(smiles):
    mol = conformer_generation(smiles)
    atom_stamp = get_atom_stamp(0.5, 4)
    voxel_size = np.sum(get_voxel_shape(mol, atom_stamp, 0.5, 11))
    return voxel_size

def sample_atom_nums(batch_size, atom_nums, atom_dist):
    return np.random.choice(atom_nums, batch_size, p=atom_dist).tolist()
    

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

def construct_dist_mat(protein_ligand_dist, atom_mode):
    tbl = Chem.GetPeriodicTable()
    if atom_mode == "add_aromatic":
        ligand_atom_indices = trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC
    else:
        ligand_atom_indices = trans.MAP_INDEX_TO_ATOM_TYPE_ONLY

    max_atom_index = max([tbl.GetAtomicNumber(atom) for atom_tuple in protein_ligand_dist for atom in atom_tuple]) + 1
    protein_ligand_dist_mat = np.ones((len(ligand_atom_indices), max_atom_index)) * 4
    ligand_atom_map = {}
    for ligand_atom_idx in ligand_atom_indices:
        ligand_atom = ligand_atom_indices[ligand_atom_idx]
        if ligand_atom[0] not in ligand_atom_map: ligand_atom_map[ligand_atom[0]] = []
        ligand_atom_map[ligand_atom[0]].append(ligand_atom_idx)
    
    for atom1, atom2 in protein_ligand_dist:
        dist = protein_ligand_dist[(atom1, atom2)]
        atom1_num, atom2_num = tbl.GetAtomicNumber(atom1), tbl.GetAtomicNumber(atom2)
        if atom1_num in ligand_atom_map:
            for ligand_atom_idx in ligand_atom_map[atom1_num]:
                protein_ligand_dist_mat[ligand_atom_idx, atom2_num] = dist
        
        if atom2_num in ligand_atom_map:
            for ligand_atom_idx in ligand_atom_map[atom2_num]:
                protein_ligand_dist_mat[ligand_atom_idx, atom1_num] = dist            
    
    return protein_ligand_dist_mat
    

def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='none', sample_func=None,
                            threshold_type=None, threshold_args=None, sample_num_atoms='prior', use_grad=False,
                            grad_step=1000, grad_lr=1, shape_AE=None, use_mesh_data=None, use_mesh_gap=None, 
                            use_pointcloud_data=None, use_pocket_data=None, init_scale=False, guide_stren=0,
                            protein_ligand_dist=None):
    all_pred_pos, all_pred_v, all_pred_bond = [], [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_pos_cond_traj, all_pred_v_cond_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    
    current_i = data.ligand_element.shape[0]
    n_range = list(range(current_i // 2, int(1.5 * current_i)))
    collate_exclude_keys = ['mol', 'ligand_index', 'id']
    for i in (range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], exclude_keys=collate_exclude_keys, follow_batch=list(FOLLOW_BATCH) + ['bound']).to(device)
        
        batch_bounds = batch.bound.view(len(batch.ligand_smiles), -1, 2)
        t1 = time.time()
        #with torch.no_grad():
        #ligand_cum_atoms = [0]
        if sample_num_atoms == 'size':
            if type(sample_func) is int:
                ligand_num_atoms = [sample_func] * n_data
            else:
                ligand_num_atoms = sample_func(n_data)
            ligand_cum_atoms = [0]
            for natom in ligand_num_atoms: ligand_cum_atoms.append(ligand_cum_atoms[-1] + natom)
            batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            ligand_cum_atoms = [0] + [natom for natom in ligand_num_atoms]
        else:
            raise ValueError

        # print('Atom List!!!!')
        # print(ligand_num_atoms)
        
        # init ligand pos
        all_ligand_atoms = sum(ligand_num_atoms)
        
        init_ligand_pos = torch.randn(all_ligand_atoms, 3).to(device)
        
        # init ligand v
        if pos_only:
            # init_ligand_v = F.one_hot(batch.ligand_atom_feature_full, num_classes=model.num_classes).float()
            init_ligand_v = batch.ligand_atom_feature_full
        else:
            if model.v_mode == 'gaussian':
                init_ligand_v = torch.randn(len(batch_ligand), model.num_classes).to(device)
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
        
        
        protein_pos = batch.protein_pos
        r = model.sample_diffusion(
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            ligand_shape=batch.shape_emb,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            grad_step=grad_step,
            use_mesh_data=use_mesh_data,
            use_pointcloud_data=use_pointcloud_data,
            use_pocket_data=use_pocket_data,
            guide_stren=guide_stren,
            pred_bond=True,
            bounds=batch_bounds,
            protein_pos=protein_pos,
        )
        ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
        ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
        ligand_pos_cond_traj = r['pos_cond_traj']
        ligand_v_cond_traj = r['v_cond_traj']
        ligand_bond = r['bond']


        if ligand_bond is not None:
            ligand_bond_array = ligand_bond.cpu().numpy().astype(np.int)
            ligand_bond_idx = np.argsort(ligand_bond_array[:, 0])
            ligand_bond_array = ligand_bond_array[ligand_bond_idx, :]

            n_idx, st_b_idx = 0, 0
            for b_idx in range(len(ligand_bond_array)):
                idx1, _, _ = ligand_bond_array[b_idx]
                if idx1 >= ligand_cum_atoms[n_idx+1]:
                    single_bond_indices = ligand_bond_array[st_b_idx:b_idx, :2] - ligand_cum_atoms[n_idx]
                    single_bond = np.concatenate([single_bond_indices, ligand_bond_array[st_b_idx:b_idx, -1].reshape(-1, 1)], axis=1)
                    
                    all_pred_bond.append(single_bond) #ligand_bond_array[st_b_idx:b_idx, :])
                    st_b_idx = b_idx
                    n_idx += 1

            single_bond_indices = ligand_bond_array[st_b_idx:, :2] - ligand_cum_atoms[n_idx]
            single_bond = np.concatenate([single_bond_indices, ligand_bond_array[st_b_idx:, -1].reshape(-1, 1)], axis=1)
            all_pred_bond.append(single_bond)

        # unbatch pos
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
        try:
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]  # num_samples * [num_atoms_i, 3]
        except:
            pdb.set_trace()

        all_step_pos = [[] for _ in range(n_data)]
        all_step_cond_pos = [[] for _ in range(n_data)]
        #all_step_uncond_pos = [[] for _ in range(n_data)]

        for pos, cond_pos in zip(ligand_pos_traj, ligand_pos_cond_traj):  # step_i
            p_array = pos.detach().cpu().numpy().astype(np.float64)
            cond_p_array = cond_pos.detach().cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])
                all_step_cond_pos[k].append(cond_p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])

        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        all_step_cond_pos = [np.stack(step_pos) for step_pos in all_step_cond_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        
        all_pred_pos_traj += [p for p in all_step_pos]
        all_pred_pos_cond_traj += [p for p in all_step_cond_pos]

        # unbatch v
        ligand_v_array = ligand_v.cpu().numpy()
        all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]

        all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += [v for v in all_step_v]

        all_step_cond_v = unbatch_v_traj(ligand_v_cond_traj, n_data, ligand_cum_atoms)
        all_pred_v_cond_traj += [v for v in all_step_cond_v]

        if not pos_only:
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_bond, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, \
        all_pred_pos_cond_traj, all_pred_v_cond_traj


def evaluate_pocket(ligand_results, config, result_path):
    all_pred_ligand_pos = ligand_results['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = ligand_results['pred_ligand_v_traj']
    num_samples = len(all_pred_ligand_pos)

    all_mols = []

    for sample_idx, (pred_pos, pred_v) in enumerate(
            zip(all_pred_ligand_pos, all_pred_ligand_v)):
        pred_pos, pred_v = pred_pos[config.eval_step], pred_v[config.eval_step]
        if config.pos_offset:
            #protein_pos = pocket_results['data'].protein_pos
            #center_pos = torch.mean(protein_pos, dim=0, keepdim=True)
            center = ligand_results['data'].point_cloud_center + ligand_results['data'].ligand_center 
            pred_pos = pred_pos + center.numpy()

        pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=config.atom_enc_mode)
        # reconstruction
        try:
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=config.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic,
                                                         basic_mode=config.basic_mode)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            if config.verbose:
                logger.warning('Reconstruct failed %s' % f'{sample_idx}')
            continue

        if '.' in smiles:
            continue

        all_mols.append(mol)

        writer = Chem.SDWriter(f"{result_path}/generated{len(all_mols)}.sdf")
        writer.write(mol)
        writer.close()

    return all_mols

    # logger.info(f'Evaluate done! {num_samples} samples in total. {len(all_mols)} eval success!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int, default=0)
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--protein_ligand_dist', type=str, default='../data/crossdocked/protein_ligand_dist.txt')

    parser.add_argument('--result_path', type=str, default='/path/to/diffsmol/results')
    parser.add_argument('--protein_root', type=str, default='/path/to/proteins')

    parser.add_argument('--pdb_filename', type=str, default='SDF/14gs_protein.pdb')
    parser.add_argument('--sdf_filename', type=str, default='SDF/Cpd2-Best-GenAI-Sln-PDL1-ligand-Asof07042025.sdf')


    args = parser.parse_args()

    
    sdf_path = os.path.join(args.protein_root, args.sdf_filename)
    pdb_path = os.path.join(args.protein_root, args.pdb_filename)
    files = [(pdb_path, sdf_path)]

    logger = misc.get_logger('evaluate')
    
    # Load config
    config = misc.load_config(args.config)
    # logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=torch.device(args.device))
    if 'train_config' in config.model:
        # logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])
    # logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    if 'transform' in ckpt['config'].data:
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    else:
        ligand_atom_mode = 'full'

    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])


    # Load dataset
    test_set = get_dataset(
        config=config.data,
        transform=transform,
        files=files
    )

    # print(test_set[0])
    # logger.info(f'Successfully load the dataset (size: {len(test_set)})!')
    
    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES)

    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    # logger.info(f'Successfully load the model! {config.model.checkpoint}')
    
    dists = pickle.load(open("../data/MOSES2/MOSES2_training_val_shape_atomnum_dict.pkl", 'rb'))
    
    if config.sample.use_grad:
        shapeae = build_point_shapeAE_model(config.data.shape, device=args.device, detach=False)
    else:
        shapeae = None

    for i in tqdm(range(len(test_set))): 
        # result_path = os.path.join(args.result_root, 'INPP5D')

        os.makedirs(args.result_path, exist_ok=True)

        data = test_set[i]
        # print(data.ligand_filename, data.protein_filename)
        # print(pdb_list[i])

        # print(data.ligand_element)
        
        if config.sample.use_mesh:
            print('Use Mesh!!!!!!!!!')
            mesh = get_mesh(data['mol'], probe_radius=0.5)
            point_clouds = np.array(data['point_cloud'].squeeze(0))
            kdtree = KDTree(point_clouds)
            mesh = trimesh.Trimesh(mesh[0], mesh[1])
            use_mesh_data = (mesh, point_clouds, kdtree)
            config.sample.use_pointcloud = False
        else:
            use_mesh_data = None

        if config.sample.use_pointcloud:
            print('Use Pointcloud!!!!!!!!!')
            data['point_cloud'] = data['point_cloud'].cpu()
            atom_pos = np.array(data['ligand_pos'])
            point_clouds = get_pointcloud_from_mol(atom_pos)
            
            kdtree = KDTree(point_clouds)
            use_pointcloud_data = (point_clouds, kdtree, config.sample.use_pointcloud_radius)
        else:
            use_pointcloud_data = None

        if "use_pocket" in config.sample and config.sample.use_pocket:
            print('Use Pocket!!!!!!!!!')
            data['protein_pos'] = data['protein_pos'].cpu().numpy()
            data['protein_element'] = data['protein_element'].cpu().numpy()
            kdtree = KDTree(data['protein_pos'])
            
            with open(args.protein_ligand_dist, 'r') as f:
                protein_ligand_dist = {}
                for line in f.readlines():
                    atom1, atom2, dist = line.strip().split(" ")
                    protein_ligand_dist[(atom1, atom2)] = float(dist)
                protein_ligand_tensor = construct_dist_mat(protein_ligand_dist, config.data.transform.ligand_atom_mode)
            use_pocket_data = (data['protein_pos'], data['protein_element'], kdtree, protein_ligand_tensor, config.sample.pocket_grad_step, config.sample.pocket_threshold)
            
        else:
            use_pocket_data = None
            protein_ligand_dist = None

        voxel_shape = get_voxel_size(data['ligand_smiles'])
        print(data['ligand_smiles'])
        atom_nums = {}
        for key in dists.keys():
            if key < voxel_shape + 200 and key > voxel_shape - 200:
                atom_nums.update(dists[key])

        print('atom_nums')
        print(atom_nums)


        if len(atom_nums) == 0:
            # logger.info("failed to build atom dists as the molecules have shape volume %d and atom num %d" % (voxel_shape, data['mol'].GetNumAtoms()))
            sample_func = data['mol'].GetNumAtoms()
        else:
            atom_num_keys = list(atom_nums.keys())
            total_num = sum([atom_nums[key] for key in atom_num_keys])
            sample_atom_dist = [atom_nums[num]/total_num for num in atom_num_keys]
            sample_func = partial(sample_atom_nums, atom_nums=atom_num_keys, atom_dist = sample_atom_dist)

            print('sample_atom_dist')
            print(sample_atom_dist)
        
        if not config.sample.use_grad:
            config.sample.grad_lr = 0

        pred_pos, pred_v, pred_bond, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, \
            pred_pos_cond_traj, pred_v_cond_traj = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            threshold_type=config.sample.threshold_CFG.type if 'threshold_CFG' in config.sample else None,
            threshold_args=config.sample.threshold_CFG if 'threshold_CFG' in config.sample else None,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            use_grad=config.sample.use_grad,
            grad_step=config.sample.grad_step,
            grad_lr=config.sample.grad_lr,
            shape_AE=shapeae,
            use_mesh_data=use_mesh_data,
            use_pointcloud_data=use_pointcloud_data,
            use_pocket_data=use_pocket_data,
            use_mesh_gap=config.sample.use_mesh_gap,
            sample_num_atoms=config.sample.sample_num_atoms,
            sample_func=sample_func,
            init_scale=config.sample.init_scale,
            guide_stren=config.sample.guide_stren,
            protein_ligand_dist=protein_ligand_dist
        )
        print('time: ', time_list)
        result = {
            'data': data,
            'pred_ligand_bond': pred_bond,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list,
            'pred_ligand_pos_cond_traj': pred_pos_cond_traj,
            'pred_ligand_v_cond_traj': pred_v_cond_traj,
            # 'pred_ligand_v0_traj': pred_v0_traj,
            # 'pred_ligand_vt_traj': pred_vt_traj
        }
        # shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))

        mols = evaluate_pocket(result, config.evaluate, args.result_path)

        for mol in mols:
            print(Chem.MolToSmiles(mol))



        # vina_results = []
        # vina_min_results = []
        # qed = []
        # sa = []
        # logp = []
        # lipinski = []
        # smiles_results = []
        # name = []

        # supplier = Chem.SDMolSupplier(sdf_path)
        # mol = supplier[0]

        # vina_task = VinaDockingTask.from_generated_mol(
        #             mol, result['data'].ligand_filename, protein_filename=args.pdb_filename, protein_root=args.protein_root, tmp_dir='/path/to/tmp')
                        
        # score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, cpu=10)
        # minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness, cpu=10)
        # vina_results.append((score_only_results[0]['affinity']))
        # vina_min_results.append((minimize_results[0]['affinity']))

        # chem_results = scoring_func.get_chem(mol)
        # qed.append(chem_results['qed'])
        # sa.append(chem_results['sa'])
        # logp.append(chem_results['logp'])
        # lipinski.append(chem_results['lipinski'])
        # smiles_results.append(result['data'].ligand_smiles)
        # name.append('reference')

        # for i, mol in enumerate(mols):
        #     vina_task = VinaDockingTask.from_generated_mol(
        #         mol, result['data'].ligand_filename, protein_filename=args.pdb_filename, protein_root=args.protein_root, tmp_dir='/path/to/tmp')
                    
        #     score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, cpu=10)
        #     score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, cpu=10)
        #     minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness, cpu=10)
        #     vina_results.append((score_only_results[0]['affinity']))
        #     vina_min_results.append((minimize_results[0]['affinity']))

        #     chem_results = scoring_func.get_chem(mol)
        #     qed.append(chem_results['qed'])
        #     sa.append(chem_results['sa'])
        #     logp.append(chem_results['logp'])
        #     lipinski.append(chem_results['lipinski'])
        #     smiles_results.append(Chem.MolToSmiles(mol))
        #     name.append(f'generated{i}')

        # print(f'!!!!!!!!Average Vina Score is {sum(vina_results)/len(vina_results)}!!!!!!!!')
        # print(f'!!!!!!!!Average Vina Min Score is {sum(vina_min_results)/len(vina_min_results)}!!!!!!!!')
        # print(f'!!!!!!!!Average QED is {sum(qed)/len(qed)}!!!!!!!!')
        # print(f'!!!!!!!!Average SA is {sum(sa)/len(sa)}!!!!!!!!')

        # # import pandas as pd
        # # df = pd.DataFrame({
        # #         "Name": name,
        # #         "SMILES": smiles_results,
        # #         "Vina S": vina_results,
        # #         "Vina M": vina_min_results,
        # #         "QED": qed,
        # #         "SA": sa,
        # #         "LogP": logp,
        # #         "Lipinski": lipinski
        # #         })
        # # df_path = os.path.join(args.result_path, 'stat.csv')
        # # df.to_csv(df_path, index=False)

        # result_path = args.result_path
        # shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        # shutil.copyfile(glob(os.path.join(os.path.dirname(os.path.dirname(config.model.checkpoint)), '*.yml'))[0],
        #                 os.path.join(result_path, 'training.yml'))

        # torch.save(result, os.path.join(result_path, f'result.pt'))


