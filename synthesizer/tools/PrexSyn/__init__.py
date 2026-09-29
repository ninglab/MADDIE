from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, List
import uuid, sys
import pandas

from ...utils.timestamp import timestamp
from ..base import Tool
from ...data.route import Route, RouteSource

docs = """`PrexSyn(molecules="targets.smi", top=3, num_samples=28, batch_size_limit=64, seed=123, device="cuda")`
PrexSyn projects one or more target molecules into synthesizable chemical space
and returns synthesis route for the top projections.  Please note, the top projections may be synthesizable analogs to the target molecule,
rather than the target molecule itself.  This tool runs relatively quickly, in less than a minute per target molecule,
but the synthesizable analogs may have different properties than the original molecule.
This tool is most useful when molecules are expected to be difficult to synthesize, or to expand strong hits by identifying synthesizable analogs.

Specify a molecules file with molecules="filename.smi".
The molecules file should contain one target SMILES per line.
Each returned `Route` corresponds to one ranked PrexSyn proposal:
    - `route.product` is the proposed synthesizable molecule returned by PrexSyn
    - `route.reactions` contains reaction strings extracted from the synthesis tree, preferring `reactant1.reactant2>>product`
    - `route.source.parameters` includes the original `target_smiles`, the `similarity` score, and the `rank`
Use top to control how many synthesizable molecules and routes are returned per input molecule, ranked by their similarity score, num_samples to control how many candidates are sampled, batch_size_limit to control how many molecules PrexSyn batches together during inference, and seed to make the stochastic sampling reproducible when needed.
Set device="cuda" to require GPU inference. Use "cpu" only when you intentionally want CPU inference.
Always clearly indicate that these are synthesizable analogs when responding to the user.
"""


class PrexSyn(Tool):
    IMAGE_NAME = "prexsyn.sif"
    DEFAULT_CONFIG = "data/trained_models/enamine2310_rxn115_202511.yml"
    DEFAULT_IMAGE = Path("/path/to/synthesizer-assets/prexsyn.sif")

    def documentation(self):
        return docs

    def _apptainer(self) -> str:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        if runtime is None:
            raise RuntimeError("PrexSyn requires apptainer or singularity on PATH.")
        return runtime

    def _tool_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _repo_dir(self) -> Path:
        configured = self.setup_config.get("repo_dir") or os.environ.get("PREXSYN_REPO")
        if configured:
            return Path(str(configured)).expanduser().resolve()
        return (self._tool_dir() / "prexsyn").resolve()

    def _image_path(self) -> Path:
        configured = self.setup_config.get("image") or os.environ.get("PREXSYN_IMAGE")
        candidates = []
        if configured:
            candidates.append(Path(str(configured)).expanduser().resolve())
        candidates.extend(
            [
                self.DEFAULT_IMAGE,
                (self._tool_dir() / self.IMAGE_NAME).resolve(),
                Path("/tmp/prexsyn.sif"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _apptainer_exec_base(self, *, device: str) -> list[str]:
        command = [
            self._apptainer(),
            "exec",
            "--cleanenv",
            "--containall",
            "--no-home",
        ]
        if device.lower().startswith("cuda"):
            command.append("--nv")
        return command

    def _check_image(self) -> Path:
        image = self._image_path()
        if not image.exists():
            raise FileNotFoundError(
                f"PrexSyn image not found: {image}. Build it from "
                "synthesizer/synthesizer/tools/PrexSyn/prexsyn/prexsyn.def."
            )

        result = subprocess.run(
            self._apptainer_exec_base(device="cpu")
            + [
                str(image),
                "python",
                "-c",
                "import prexsyn, prexsyn_engine",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"PrexSyn image is not usable under apptainer: {image}")
        return image

    def setup_tool(self):
        repo_dir = self._repo_dir()
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"PrexSyn repository directory not found: {repo_dir}")
        self._runtime_image = self._check_image()

    def _smiles_list(self, molecules: Any) -> tuple[list[str], str]:
        if molecules is not None:
            path = Path(str(molecules)).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"PrexSyn molecules file not found: {molecules}")
            if not path.is_file():
                raise ValueError(f"PrexSyn molecules input must be a file: {molecules}")
            smiles_list = [line.strip() for line in path.read_text().splitlines() if line.strip().lower() not in ["", "smiles", "mols", "mol", "smi", "molecules"]]
            if not smiles_list:
                raise ValueError(f"PrexSyn molecules file is empty: {path}")
            return smiles_list, str(path)

        raise ValueError('Please specify molecules="filename.smi".')

    def _flatten_reactions(self, tree: dict[str, Any]) -> list[str]:
        reactions: list[str] = []

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            product = node.get("SMILES")
            reaction = node.get("Reaction")
            precursors = node.get("Precursors")
            if isinstance(precursors, list):
                for precursor_group in precursors:
                    if isinstance(precursor_group, dict):
                        reactants: list[str] = []
                        for child in precursor_group.values():
                            if isinstance(child, dict):
                                child_smiles = child.get("SMILES")
                                if isinstance(child_smiles, str) and child_smiles:
                                    reactants.append(child_smiles)
                            walk(child)
                        if isinstance(product, str) and product and reactants:
                            reactions.append(".".join(reactants) + ">>" + product)
                        elif isinstance(reaction, str):
                            reactions.append(reaction)
            elif isinstance(reaction, str):
                reactions.append(reaction)

        walk(tree)
        return reactions
    
    def csv_string(self, routes : List[Route]):
        data = {
            "route_id": [],
            "route_length": [],
            "product_molecule": [],
            "analog_of": [],
            "target_synthesizable_decision": [],
            "analog_synthesizable_decision": [],
            "analog_similarity": [],
            "analog_rank": [],
            "source": [],
            "reactions": []
        }
        for r in routes:
            data["route_id"].append(r.route_id)
            data["route_length"].append(r.length)
            data["product_molecule"].append(r.product)
            data["analog_of"].append(r.source.parameters["target_smiles"])
            data["analog_similarity"].append(r.source.parameters["similarity"])
            data["analog_rank"].append(r.source.parameters["rank"])
            data["source"].append(r.source.planner)
            data["reactions"].append("||".join(r.reactions))
        df : pandas.DataFrame = pandas.DataFrame.from_dict(data)
        return df.to_csv(index=False)

    def csv_string(self, routes: List[Route]) -> str:
        data = {
            "route_id": [],
            "route_length": [],
            "product_molecule": [],
            "analog_of": [],
            "target_synthesizable_decision": [],
            "analog_synthesizable_decision": [],
            "analog_similarity": [],
            "analog_rank": [],
            "source": [],
            "reactions": []
        }
        for r in routes:
            data["route_id"].append(r.route_id)
            data["route_length"].append(r.length)
            data["product_molecule"].append(r.product)
            target_smiles = r.source.parameters["target_smiles"]
            data["analog_of"].append(target_smiles)
            data["target_synthesizable_decision"].append("YES" if r.product == target_smiles else "NO")
            data["analog_synthesizable_decision"].append("YES")
            data["analog_similarity"].append(r.source.parameters["similarity"])
            data["analog_rank"].append(r.source.parameters["rank"])
            data["source"].append(r.source.planner)
            data["reactions"].append("||".join(r.reactions) if r.reactions else "")
        df: pandas.DataFrame = pandas.DataFrame.from_dict(data)
        return df.to_csv(index=False)

    def _route_from_result(
        self,
        *,
        target_smiles: str,
        rank: int,
        result: dict[str, Any],
        input_source: str,
        config_path: str,
        num_samples: int,
        top: int,
        seed: int | None,
        device: str,
    ) -> Route:
        product = str(result["product_smiles"])
        tree = result["tree"]
        reactions = self._flatten_reactions(tree)
        parameters = {
            "target_smiles": target_smiles,
            "rank": rank,
            "similarity": result["similarity"],
            "input_source": input_source,
            "config": config_path,
            "num_samples": num_samples,
            "top": top,
            "device": device,
            "tree": tree,
        }
        if seed is not None:
            parameters["seed"] = seed
        return Route(
            route_id=f"prexsyn:{target_smiles}:{rank}:{product}",
            product=product,
            synthesizable=True,
            reactions=reactions,
            source=RouteSource(
                planner="PrexSyn",
                parameters=parameters,
            ),
        )

    def __call__(self, **params):
        molecules_param = params.get("molecules")
        try:
            smiles_list, input_source = self._smiles_list(molecules_param)
        except Exception as exc:
            return None, f"⚠️ Failed with {exc.__class__.__name__}: {exc}"

        top = max(1, int(params.get("top", 3)))
        num_samples = max(top, int(params.get("num_samples", max(8, top))))
        batch_size_limit = max(1, int(params.get("batch_size_limit", 64)))
        seed_param = params.get("seed", 0)
        seed = None if seed_param is None else int(seed_param)
        device = str(params.get("device", "cpu"))
        config_path = str(params.get("config", self.DEFAULT_CONFIG))
        repo_dir = self._repo_dir()
        image = getattr(self, "_runtime_image", None) or self._check_image()

        io_dir = Path(self.agent_output_dir)
        created_io_dir = False
        output_path: Path | None = None
        try:
            if io_dir.exists() and not io_dir.is_dir():
                raise ValueError(f"PrexSyn output path exists and is not a directory: {io_dir}")
            created_io_dir = not io_dir.exists()
            io_dir.mkdir(parents=True, exist_ok=True)

            input_path = io_dir / "prexsyn_inputs.json"
            output_path = io_dir / "prexsyn_results.json"
            input_path.write_text(
                json.dumps(
                    {
                        "smiles": smiles_list,
                        "config": config_path,
                        "top": top,
                        "num_samples": num_samples,
                        "batch_size_limit": batch_size_limit,
                        "seed": seed,
                        "device": device,
                    }
                )
            )

            script = r"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from prexsyn.shortcuts import AllInOneLoader, MoleculeProjector

payload = json.loads(Path("/tmp/prexsyn_io/prexsyn_inputs.json").read_text())
requested_device = payload["device"]
seed = payload.get("seed")
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if requested_device.lower().startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested for PrexSyn, but torch.cuda.is_available() is false inside the container.")

loader = AllInOneLoader(payload["config"])
model = loader.model().to(requested_device).eval()
actual_device = str(model.device)
if requested_device.lower().startswith("cuda") and not actual_device.startswith("cuda"):
    raise RuntimeError(
        f"CUDA was requested for PrexSyn, but the model was placed on {actual_device!r}."
    )

projector = MoleculeProjector(
    model=model,
    detokenizer=loader.detokenizer(),
    descriptor="ecfp4",
    num_samples=payload["num_samples"],
    batch_size_limit=payload["batch_size_limit"],
)

batched_result = projector.many(payload["smiles"])
all_results = []
for target, result in zip(payload["smiles"], batched_result.results, strict=True):
    items = []
    for rank, item in enumerate(result.items[: payload["top"]], start=1):
        items.append(
            {
                "rank": rank,
                "product_smiles": item.molecule.smiles(),
                "similarity": item.similarity,
                "tree": item.get_tree(),
            }
        )
    all_results.append({"target_smiles": target, "items": items})

Path("/tmp/prexsyn_io/prexsyn_results.json").write_text(json.dumps(all_results))
"""

            command = self._apptainer_exec_base(device=device) + [
                "--writable-tmpfs",
                "--env",
                "PYTHONNOUSERSITE=1",
                "--cwd",
                "/opt/prexsyn",
                "--bind",
                f"{repo_dir}:/opt/prexsyn",
                "--bind",
                f"{io_dir}:/tmp/prexsyn_io",
                str(image),
                "python",
                "-c",
                script,
            ]

            sys.stderr.write("\n".join([timestamp(), "[synthesizer/tools/PrexSyn/__init__.py:293]", "[DEBUG]", " ".join(command)]))

            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
                return None, f"⚠️ Failed with RuntimeError: {detail}"

            payload = json.loads(output_path.read_text())
        finally:
            for path in [input_path, output_path]:
                if path is not None:
                    path.unlink(missing_ok=True)

        routes: list[Route] = []
        for target_entry in payload:
            target_smiles = str(target_entry["target_smiles"])
            result_items = list(target_entry.get("items") or [])
            for result in result_items:
                routes.append(
                    self._route_from_result(
                        target_smiles=target_smiles,
                        rank=int(result["rank"]),
                        result=result,
                        input_source=input_source,
                        config_path=config_path,
                        num_samples=num_samples,
                        top=top,
                        seed=seed,
                        device=device,
                    )
                )
        if not routes:
            return None, "✅ Executed successfully; no synthesis routes returned."
        else:
            datastr = self.csv_string(routes)
            _stem = getattr(self, "_output_stem_override", None) or (Path(molecules_param).stem if molecules_param else "prexsyn_output")
            rtfile = str(io_dir / f"{_stem}_prexsyn.csv")
            with open(rtfile, "w") as f:
                f.write(datastr)
            if len(routes) == 1: # For grammatical correctness
                return routes[0], f"✅ Generated 1 synthesizable analog with a route using PrexSyn.  Results are also stored in {rtfile}."
        return routes, f"✅ Generated {len(routes)} synthesizable analogs with routes using PrexSyn. Results are also stored in {rtfile}."
