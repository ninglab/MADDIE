import json
import os
import pandas as pd
import shutil

class Memory(object):
    def __init__(self):
        self.stream = {}
        self.history = []
        self.id_counters = {}
        self.history_id = 0

    def _generate_id(self, identifier: str):
        current_id = self.id_counters.get(identifier, 0)
        val = identifier + f"{current_id:03}"
        self.id_counters[identifier] = current_id + 1
        return val

    def _add_block(self, mem_id: str, block):
        self.stream[mem_id] = block

    def _compact_generation_metadata(self, metadata):
        if not isinstance(metadata, dict):
            return metadata
        compact = {
            key: metadata[key]
            for key in [
                "method",
                "vs_id",
                "sample_size",
                "top_k",
                "selected_candidate_ids",
                "selected_smiles",
                "skipped_seed_candidates",
                "screened_drug_ids",
            ]
            if key in metadata
        }
        if "seed_runs" in metadata:
            compact["seed_runs"] = [
                {
                    key: seed_run[key]
                    for key in ["seed_candidate_id", "seed_docked_vina"]
                    if key in seed_run
                }
                for seed_run in (metadata.get("seed_runs") or [])
            ]
        if "seed_candidate_id" in metadata:
            compact["seed_candidate_id"] = metadata["seed_candidate_id"]
        return compact

    def _persist_candidate_files(self, mem_id: str, data: pd.DataFrame, storage_root: str, block_kwargs):
        if "SDF Path" not in data.columns:
            raise KeyError(
                "Candidate persistence requires an 'SDF Path' column so memory can store stable molecule files."
            )

        generation_metadata = block_kwargs.get("generation_metadata", {})
        os.makedirs(storage_root, exist_ok=True)
        persisted_paths = []
        source_paths_to_cleanup = []
        for _, row in data.iterrows():
            candidate_id = row["Candidate ID"]
            source_sdf = row["SDF Path"]
            if not isinstance(source_sdf, str) or not source_sdf:
                raise KeyError(f"Candidate {candidate_id} is missing a valid source SDF Path")
            if not os.path.exists(source_sdf):
                raise FileNotFoundError(f"Candidate {candidate_id} source SDF does not exist: {source_sdf}")

            candidate_dir = os.path.join(storage_root, candidate_id)
            os.makedirs(candidate_dir, exist_ok=True)
            stored_sdf = os.path.join(candidate_dir, "molecule.sdf")
            shutil.copy2(source_sdf, stored_sdf)
            stored_protein = ""
            source_protein = row["Protein Path"] if "Protein Path" in row else ""
            if isinstance(source_protein, str) and source_protein and os.path.exists(source_protein):
                stored_protein = os.path.join(candidate_dir, "protein.pdb")
                shutil.copy2(source_protein, stored_protein)

            metadata_path = os.path.join(candidate_dir, "metadata.json")
            evaluation = {}
            for key in ["Heavy Atom Count", "Vina Score", "QED", "SA Score"]:
                if key in row and pd.notna(row[key]):
                    evaluation[key] = self._json_safe_value(row[key])
            metadata = {
                "candidate_id": candidate_id,
                "molecule_set_id": mem_id,
                "SMILES": row["SMILES"] if "SMILES" in row else "",
                "stored_sdf_path": stored_sdf,
                "stored_protein_path": stored_protein,
                "source_action": block_kwargs.get("action", ""),
            }
            if evaluation:
                metadata["evaluation"] = evaluation
            predicted_binding = {}
            for key in ["affinity_probability_binary", "affinity_pred_value"]:
                if key in row and pd.notna(row[key]):
                    predicted_binding[key] = self._json_safe_value(row[key])
            if predicted_binding:
                metadata["predicted_binding"] = predicted_binding
            if generation_metadata:
                generation = {
                    "method": generation_metadata.get("method", ""),
                }
                if generation_metadata.get("seed_runs"):
                    seed_run = generation_metadata["seed_runs"][0]
                    generation["seed_molecule"] = seed_run.get("seed_candidate_id", "")
                    if "seed_docked_vina" in seed_run:
                        generation["seed_docked_vina"] = self._json_safe_value(seed_run.get("seed_docked_vina"))
                elif generation_metadata.get("seed_candidate_id"):
                    generation["seed_molecule"] = generation_metadata.get("seed_candidate_id", "")
                metadata["generation"] = generation
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            persisted_paths.append(stored_sdf)
            source_paths_to_cleanup.append(source_sdf)

        for source_sdf in sorted(set(source_paths_to_cleanup)):
            if os.path.exists(source_sdf):
                os.remove(source_sdf)
            self._cleanup_empty_parent_dirs(source_sdf)

        data = data.copy()
        data["SDF Path"] = persisted_paths
        if "Protein Path" in data.columns:
            protein_paths = []
            for _, row in data.iterrows():
                candidate_dir = os.path.join(storage_root, row["Candidate ID"])
                stored_protein = os.path.join(candidate_dir, "protein.pdb")
                protein_paths.append(stored_protein if os.path.exists(stored_protein) else "")
            data["Protein Path"] = protein_paths
        return data

    def _json_safe_value(self, value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _cleanup_empty_parent_dirs(self, path: str):
        current_dir = os.path.dirname(path)
        for _ in range(2):
            if not current_dir or not os.path.isdir(current_dir):
                return
            if os.listdir(current_dir):
                return
            os.rmdir(current_dir)
            current_dir = os.path.dirname(current_dir)

    def add_mol(self, data: pd.DataFrame, return_id: bool = False, **kwargs):
        mem_id = self._generate_id("MOL")
        data = data.copy()
        if "Candidate ID" not in data.columns:
            data.insert(0, "Candidate ID", [f"{mem_id}_ROW{i:03}" for i in range(len(data))])
        storage_root = kwargs.pop("storage_root", kwargs.pop("artifact_root", ""))
        if storage_root:
            data = self._persist_candidate_files(mem_id, data, storage_root, kwargs)
        block = {"type": "MOL", "data": data}
        for key, val in kwargs.items():
            if key == "generation_metadata":
                val = self._compact_generation_metadata(val)
            block[key] = val
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_vs(self, data: pd.DataFrame, return_id: bool = False, **kwargs):
        mem_id = self._generate_id("VS")
        block = {"type": "VS", "data": data}
        for key, val in kwargs.items():
            if key == "generation_metadata":
                val = self._compact_generation_metadata(val)
            block[key] = val
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_pocket(self, filename: str, return_id: bool = False, **kwargs):
        mem_id = "POCKET"
        block = {"type": "POCKET", "filename": filename}
        for key, val in kwargs.items():
            block[key] = val
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_action(self, action_id: str, desc: str, return_id: bool = False, **kwargs):
        if "id" in kwargs.keys():
            mem_id = kwargs["id"]
        else:
            mem_id = self._generate_id(action_id)  
        block = {"type": action_id, "desc": desc}    
        for key, val in kwargs.items():
            block[key] = val  
        self._add_block(mem_id, block)
        if return_id:
            return mem_id

    def add_history(self, **kwargs):
        block = {"id": self.history_id}
        for key, val in kwargs.items():
            if key == "metadata":
                val = self._compact_history_metadata(val)
            block[key] = val
        self.history.append(block)  
        self.history_id += 1

    def _compact_history_metadata(self, metadata):
        return self._compact_generation_metadata(metadata)

    def get_used_seed_candidate_ids(self, action_prefix: str = ""):
        used = set()
        for item in self.history:
            action_id = str(item.get("action_id", ""))
            if action_prefix and not action_id.startswith(action_prefix):
                continue

            action_input = item.get("action_input", []) or []
            for value in action_input:
                if isinstance(value, str) and "_ROW" in value:
                    used.add(value)

            metadata = item.get("metadata", {})
            candidate_id = metadata.get("seed_candidate_id", "")
            if candidate_id and candidate_id != "REFMOL":
                used.add(candidate_id)

            for seed_run in metadata.get("seed_runs", []) or []:
                candidate_id = seed_run.get("seed_candidate_id", "")
                if candidate_id and candidate_id != "REFMOL":
                    used.add(candidate_id)
        return used

    def get_screened_drug_ids(self):
        used = set()
        for block in self.stream.values():
            if block.get("type") != "VS":
                continue
            data = block.get("data")
            if data is not None and "drug_id" in data.columns:
                used.update(str(value) for value in data["drug_id"].dropna().tolist())
            metadata = block.get("generation_metadata", {}) or {}
            used.update(str(value) for value in metadata.get("screened_drug_ids", []) or [])
        return used
