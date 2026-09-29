import os
import oddt
import pickle
import lmdb
import time
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import copy
from utils.data import *
from .shape_mol_data import ShapeMolData, torchify_dict
from utils.shape import *
from functools import partial
from multiprocessing import Pool
from utils.subproc_shapeAE import SubprocShapeAE


class ScriptDataset(Dataset):
    def __init__(self, config, transform, files):
        super().__init__()
        self.config = config
        self.transform = transform
        self.files = files


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        pocket_dict, ligand_dict, rdmol = parse_pdb_and_sdf(self.files[idx][0], self.files[idx][1])
        # print('!!!!!!!!!!!!!!!')
        # print(data)
        shape_func, subproc_voxelae = get_shape_func(self.config.shape)

        remove_idxs, batch_shape_embs, batch_bounds, batch_pointclouds, batch_pointcloud_centers = shape_func([rdmol])

        data = ShapeMolData.from_ligand_dicts(
                            ligand_dict=torchify_dict(ligand_dict),
                            protein_dict=torchify_dict(pocket_dict)
                        )
        data.shape_emb = batch_shape_embs[0]
        
        data.ligand_pos = data.ligand_pos - batch_pointcloud_centers[0]
        data.protein_pos = data.protein_pos - batch_pointcloud_centers[0] - data.ligand_center
        data.bound = batch_bounds[0]

        data.point_cloud = batch_pointclouds[0]
        data.point_cloud_center = batch_pointcloud_centers[0]
        data.mol = rdmol

        data.protein_filename = self.files[idx][0]
        data.ligand_filename = self.files[idx][1]

        if self.transform is not None:
            data = self.transform(data)

        return data
        # data = ShapeMolData.from_ligand_dicts(
        #                 ligand_dict=None,
        #                 protein_dict=torchify_dict(data)
        #             ).to_dict()
        

def get_shape_func(config):
    if config.shape_type == 'electroshape':
        shape_func = get_electro_shape_emb
    elif config.shape_type == 'voxelAE_shape':
        atom_stamp = get_atom_stamp(grid_resolution=config.grid_resolution, max_dist=4)
        if config.shape_parallel: shapeae = SubprocShapeAE(config)
        else: shapeae = build_voxel_shapeAE_model(config, device='cuda')
        
        shape_func = partial(get_voxelAE_shape_emb,
                             model=shapeae,
                             atom_stamp=atom_stamp,
                             grid_resolution=config.grid_resolution,
                             max_dist=config.max_dist,
                             batch_size=config.batch_size,
                             shape_parallel=config.shape_parallel
                             )
    elif config.shape_type == 'pointAE_shape':
        if config.shape_parallel: shapeae = SubprocShapeAE(config)
        else: shapeae = build_point_shapeAE_model(config, device='cuda')
        shape_func = partial(get_pointAE_shape_emb,
                             model=shapeae,
                             point_cloud_samples=config.point_cloud_samples,
                             config=config,
                             batch_size=config.batch_size,
                             shape_parallel=config.shape_parallel
                             )
    return shape_func, shapeae
