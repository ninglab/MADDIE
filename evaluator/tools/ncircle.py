# Taken from https://github.com/yutxie/chem-measure/blob/main/measures.py
# Note: use run_circle(...) if you want to compute NCircle of a list of SMILES strng

import rdkit
import random
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

CHUNK_SIZE = 1000

class Measure():
    def __init__(self):
        pass

    def update(self, mols=[]):
        raise NotImplementedError
        
    def measure(self, mols=[]):
        self.update(mols)
        return self.report()

    def report(self):
        raise NotImplementedError
    
    
class GoldenStandard(Measure):
    def __init__(self):
        super().__init__()
        self.target_set = set()
        
    def update(self, mols=[]):
        for mol in mols:
            self.target_set.add(mol.GetProp('Target'))
        
    def report(self):
        return len(self.target_set)


class DissimilarityBasedMeasure(Measure):
    def __init__(self, 
        vectorizer=None, 
        sim_mat_func=None, 
        *args, **kargs
    ):
        super().__init__(*args, **kargs)
        self.vectorizer = vectorizer
        self.sim_mat_func = sim_mat_func
        self.dis_mat = np.zeros((0, 0))
        self.vecs = []

    def update(self, mols=[], is_vec=False):
        n = len(self.vecs)
        new_n = n + len(mols)
        new_dis_mat = np.zeros((new_n, new_n))
        new_dis_mat[:n, :n] = self.dis_mat
        
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        sim_mat = self.sim_mat_func(vecs, vecs)
        new_dis_mat[n:, n:] = 1 - np.array(sim_mat)
        
        if n > 0:
            sim_mat = self.sim_mat_func(self.vecs, vecs)
            new_dis_mat[:n, n:] = 1 - np.array(sim_mat)
            new_dis_mat[n:, :n] = 1 - np.array(sim_mat).T
        
        self.dis_mat = new_dis_mat
        self.vecs += vecs

    def report(self):
        raise NotImplementedError
        
def diversity_map(args):
    vec, vecs, sim_mat_func = args
    dists = sim_mat_func([vec], vecs)[0]
    return dists.mean()

class Diversity(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
    def measure(self, mols=[], is_vec=False):
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        args = zip(vecs, [vecs]*len(vecs), [self.sim_mat_func]*len(vecs))
        avg_dists = process_map(diversity_map, args, chunksize=CHUNK_SIZE)
        return 1. - np.sum(avg_dists) / len(avg_dists)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return (m.sum(axis=-1) / (n - 1)).mean().item()

class SumDiversity(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return (m.sum(axis=-1) / (n - 1)).sum().item()
    
class Bottleneck(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        m = m + np.eye(n) * 1e9
        return m.min().item()
    
def sumbot_map(args):
    vec, vecs, sim_mat_func = args
    dists = 1.-sim_mat_func([vec], vecs)[0]
    return np.partition(dists, 1)[1]

class SumBottleneck(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
    def measure(self, mols=[], is_vec=False):
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        
        args = zip(vecs, [vecs]*len(vecs), [self.sim_mat_func]*len(vecs))
        min_dists = process_map(sumbot_map, args, chunksize=CHUNK_SIZE)
        return np.sum(min_dists)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        m = m + np.eye(n) * 1e9
        return m.min(axis=-1).sum().item()

class Diameter(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return m.max().item()

class SumDiameter(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return m.max(axis=-1).sum().item()
    
class DPP(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        
    def report(self):
        sim_mat = 1 - self.dis_mat
        return np.linalg.det(sim_mat)

    
class ReferenceBasedMeasure(Measure):
    def __init__(self):
        super().__init__()

import pandas as pd
# from cal_ifg_atom import CollectFG
# from cal_ringsystem import CollectRingSystems
# class NFragment(ReferenceBasedMeasure):
#     def __init__(self, frag=None):
#         super().__init__()
#         if  frag == 'FG': self.collect_fn = CollectFG
#         elif frag == 'RS': self.collect_fn = CollectRingSystems
#         self.df = pd.DataFrame(columns=['frags'])
        
#     def update(self, mols=[]):
#         frags = self.collect_fn(mols)
#         self.df = self.df.append(pd.DataFrame(frags, columns=['frags']))
        
#     def report(self):
#         self.df.drop_duplicates('frags', inplace=True)
#         return len(self.df)
    
# def bm_map(mol):
#     core = MurckoScaffold.GetScaffoldForMol(mol)
#     smi = Chem.MolToSmiles(core)
#     return smi

from rdkit.Chem.Scaffolds import MurckoScaffold
class NBM(ReferenceBasedMeasure):
    def __init__(self):
        super().__init__()
        self.smiles = set()
        
    def measure(self, mols=[]):
        smiles = process_map(bm_map, mols, chunksize=CHUNK_SIZE)
        return len(set(smiles))
        
    def update(self, mols=[]):
        for mol in tqdm(mols):
            core = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(core)
            self.smiles.add(smi)
    
    def report(self):
        return len(self.smiles)

def get_circles(args, silent=True):
    vecs, sim_mat_func, t = args
    
    circs = []
    for vec in vecs:
        if len(circs) > 0:
            dists = 1. - sim_mat_func([vec], circs)
            if dists.min() <= t: continue
        circs.append(vec)
    return circs

class NCircles(Measure):
    def __init__(self, 
        vectorizer=None, 
        sim_mat_func=None, 
        threshold=0.75,
        *args, **kargs
    ):
        super().__init__(*args, **kargs)
        self.vectorizer = vectorizer
        self.sim_mat_func = sim_mat_func
        self.t = threshold
        self.vecs = []
        
    def measure(self, mols=[], is_vec=False, n_chunk=64):
        # print('entering the func')
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        # print('splitting the list')
        
        mol_to_vec_dict = None
        if isinstance(vecs, dict):
            mol_to_vec_dict = vecs
            vecs = list(mol_to_vec_dict.values())

        for i in range(3):
            # print(n_chunk//(2**i))
            vecs_list = [list(c) for c in mit.divide(n_chunk//(2**i), vecs)]
            # print('parallelly computing n circles')
            args = zip(vecs_list, 
                       [self.sim_mat_func]*len(vecs_list), 
                       [self.t]*len(vecs_list))
            # print(len(vecs_list[0]))
            circs_list = process_map(get_circles, args)
            # print(len(circs_list[0]))
            vecs = [c for ls in circs_list for c in ls]
            vecs.sort(key=lambda fp: fp.ToBitString())
        vecs = get_circles(
            (vecs, self.sim_mat_func, self.t), 
            silent=False)
        if mol_to_vec_dict is not None:
            return len(vecs), (vecs, mol_to_vec_dict)
        else:
            return len(vecs), vecs

    def update(self, mols=[], is_vec=False):
        if is_vec: vecs = mols
        else: vecs = self.vectorizer(mols)
        for vec in vecs:
            if len(self.vecs) > 0:
                dists = 1. - self.sim_mat_func([vec], self.vecs)
                if dists.min() <= self.t: continue
            self.vecs.append(vec)

    def report(self):
        return len(self.vecs)
    
class Richness(Measure):
    def __init__(self):
        super().__init__()
        self.smiles = set()
        
    def update(self, mols=[]):
        for mol in mols:
            smi = Chem.MolToSmiles(mol)
            self.smiles.add(smi)
    
    def report(self):
        return len(self.smiles)
    
import os
from typing import List, Tuple

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) # pylint: disable=E1101

def fingerprints(mols):
    return [fingerprint(mol) for mol in mols]

def fingerprints_dict(mols):
    return {mol: fingerprint(mol) for mol in mols}

def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1] # pylint: disable=E1101
    return np.array(similarities)


#---TOOL FUNCTION---#
def run_ncircle(smi_list: List[str], d: float) -> Tuple[int, List[str]]:
    smiles = smi_list
    if len(smiles) > 0:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles] # pylint: disable=E1101
        mols = [m for m in mols if m is not None]
        measure = NCircles(
            vectorizer=fingerprints_dict,
            sim_mat_func=similarity_matrix_tanimoto,
            threshold=d,
        )
        value, circs = measure.measure(mols, n_chunk=64)
        # calculate NCircles from circs instead since value is based on bit vectors
        # Example, if d=0 NCircle should calculate # of unique molecules, not bit vectors

        vecs, mol_to_vec_dict = circs
        vecs = [v.ToBitString() for v in vecs]
        mol_to_vec_dict = {k: v.ToBitString() for k, v in mol_to_vec_dict.items()}

        vec_to_mol_dict = {}
        for k, v in mol_to_vec_dict.items():
            vec_to_mol_dict.setdefault(v, []).append(k)
        vec_to_mol_dict = {v: keys for v, keys in vec_to_mol_dict.items()}

        circs_mols = []
        for vec in vecs:
            circs_mols += vec_to_mol_dict[vec]

        circs_smi_unique = list(set([Chem.MolToSmiles(mol, canonical=True) for mol in circs_mols]))
        return len(circs_smi_unique), circs_smi_unique
    else:
        return 0, []
#---TOOL FUNCTION---#

def read_smiles_file(path: str, column_name: str = "SMILES") -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if column_name not in df.columns:
            raise ValueError(
                f"Column '{column_name}' not found in {path}. Available columns: {list(df.columns)}"
            )
        return df[column_name].dropna().astype(str).tolist()
    elif ext == ".smi":
        smiles = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    smiles.append(line.split()[0])  # first token, ignore any id/name
        return smiles
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Expected .smi or .csv")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute the NCircles diversity measure for a file of SMILES strings.")
    parser.add_argument("input_file", type=str, help="Path to the input file (.smi or .csv)")
    parser.add_argument("-d", "--threshold", type=float, default=0.75, help="Distance threshold for circle separation (default: 0.75)")
    parser.add_argument("-c", "--column_name", type=str, default="SMILES", help="Column name for SMILES (only used for .csv input, default: SMILES)")
    args = parser.parse_args()

    smi_list = read_smiles_file(args.input_file, column_name=args.column_name)
    n_circ, circs = run_ncircle(smi_list, d=args.threshold)
    print(f"Number of circles: {n_circ}")
    print("Circles:")
    for smi in circs:
        print(f"  {smi}")