from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List
import pandas

from ..base import Tool
from ...data.route import Route, RouteSource
from ..PrexSyn import PrexSyn

docs = """`MEEA(molecules="filename.smi", iter_limit=500, starting_materials=["eMoleculesBB"])`
MEEA will run a general-purpose synthesis planner on every target in the specified SMILES file,
returning the first synthesis route found for each target, if any.  This tool may take a few minutes per product,
but will always plan a route for the specified products.  MEEA is generally a good first-line tool to plan routes
for molecules that are expected to be synthesizable.

The target product molecules file must be specified, wrapped in quotes. It should contain one target SMILES per line, and no other information, that is, no header lines or SA scores.
By default, MEEA will run under a limit of 500 iterations. Time limit and parallelism are set automatically.
MEEA will stop planning when a route to starting materials is found.  By default, this is eMoleculesBB, but this can be configured to any combination of the following building block sets:
    eMoleculesBB: Over 23M commercially available molecules
"""
# Option for future expansion: EnamineREAL as building blocks.

class MEEA(Tool):
    IMAGE_NAME = "meea.sif"
    REQUIRED_LARGEFILES = (
        "policy_model.ckpt",
        "value_pc.pt",
        "template_rules.dat",
        "eMoleculesBB.txt",
        "EnamineREAL.txt",
    )
    STARTING_MATERIAL_ALIASES = {
        "eMoleculesBB": "eMoleculesBB.txt",
        "eMoleculesBB.txt": "eMoleculesBB.txt",
        "EnamineREAL": "EnamineREAL.txt",
        "EnamineREAL.txt": "EnamineREAL.txt",
    }
    agent_output_dir = None

    def documentation(self):
        return docs

    def _apptainer(self) -> str:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        if runtime is None:
            raise RuntimeError("MEEA requires apptainer or singularity on PATH.")
        return runtime

    def _tool_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _image_path(self) -> Path:
        configured = os.environ.get("MEEA_IMAGE")
        if configured:
            return Path(configured).expanduser().resolve()
        return self._tool_dir() / self.IMAGE_NAME

    def _largefile_map(self) -> dict[str, Path]:
        largefiles = self.setup_config.get("largefiles")
        if not isinstance(largefiles, dict):
            raise ValueError('MEEA requires setup_config["largefiles"] to be a dictionary of file paths.')

        resolved: dict[str, Path] = {}
        for key, value in largefiles.items():
            candidate: Any = value
            if isinstance(value, dict):
                candidate = value.get("path") or value.get("filepath") or value.get("file")
            if candidate is None:
                continue

            path = Path(candidate).expanduser().resolve()
            resolved[str(key)] = path
            resolved[path.name] = path
        return resolved

    def _required_largefiles(self) -> dict[str, Path]:
        largefiles = self._largefile_map()
        missing = [name for name in self.REQUIRED_LARGEFILES if name not in largefiles]
        if missing:
            raise FileNotFoundError("MEEA is missing required large files: " + ", ".join(sorted(missing)))

        required = {name: largefiles[name] for name in self.REQUIRED_LARGEFILES}
        absent = [name for name, path in required.items() if not path.exists()]
        if absent:
            raise FileNotFoundError("MEEA large files do not exist on disk: " + ", ".join(sorted(absent)))
        return required

    def _starting_material_paths(self, starting_materials: Any) -> list[Path]:
        if starting_materials is None:
            names = ["eMoleculesBB"]
        elif isinstance(starting_materials, str):
            names = [starting_materials]
        else:
            names = list(starting_materials)

        largefiles = self._largefile_map()
        resolved: list[Path] = []
        for name in names:
            key = self.STARTING_MATERIAL_ALIASES.get(str(name), str(name))
            path = largefiles.get(key, Path(key).expanduser().resolve())
            if not path.exists():
                raise FileNotFoundError(f"Starting materials file not found: {name}")
            resolved.append(path)

        if not resolved:
            raise ValueError("MEEA requires at least one starting materials file.")
        return resolved

    def _check_image(self) -> Path:
        image = self._image_path()
        if not image.exists():
            raise FileNotFoundError(
                f"MEEA image not found: {image}. Build it with synthesizer/synthesizer/tools/MEEA/build.sh."
            )

        result = subprocess.run(
            [
                self._apptainer(),
                "exec",
                "--cleanenv",
                "--containall",
                "--no-home",
                str(image),
                "python",
                "-c",
                "import importlib.machinery",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MEEA image is not usable under apptainer: {image}")
        return image

    def setup_tool(self):
        self._runtime_image = self._check_image()
        self._required_largefiles()

    def _products_path(self, molecules: Any) -> Path:
        path = Path(str(molecules)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"MEEA molecules file not found: {molecules}")
        if not path.is_file():
            raise ValueError(f"MEEA molecules input must be a file: {molecules}")
        return path

    def _products(self, products_path: Path) -> list[str]:
        import csv as _csv, io as _io
        text = products_path.read_text()
        first_line = text.splitlines()[0].strip() if text.strip() else ""
        # CSV detection: if first line has commas and contains a smiles field, extract that column
        if "," in first_line:
            reader = _csv.DictReader(_io.StringIO(text))
            fieldnames = reader.fieldnames or []
            smiles_col = next((f for f in fieldnames if f.strip().lower() == "smiles"), None)
            if smiles_col:
                products = [row[smiles_col].strip() for row in reader if row.get(smiles_col, "").strip()]
                if products:
                    return products
        # Plain SMILES file: one SMILES per line
        products = []
        for line in text.splitlines():
            product = line.strip()
            if product and product.lower() not in ["smiles", "mols", "molecules", "product", "products", "mol"]:
                products.append(product)
        if products:
            return products
        raise ValueError(f"MEEA molecules file is empty: {products_path}")

    def _prepared_products_input(self, products: list[str]) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".smi",
            prefix="meea_products_",
            delete=False,
        ) as handle:
            handle.write("\n".join(products))
            handle.write("\n")
            return Path(handle.name)

    def _output_dir(self, products_path: Path) -> Path:
        if self.agent_output_dir:
            return Path(str(self.agent_output_dir)).expanduser().resolve()
        return products_path.parent.resolve()

    def _route_from_payload(
        self,
        payload: dict[str, Any],
        products_path: Path,
        iter_limit: int,
        time_limit_min: float,
        simulations: int,
        cpuct: float,
        parallel: int,
        starting_materials: list[Path],
    ) -> Route:
        reactions = payload.get("reactions") or None
        product = payload["product"]  # must match the route product; sorted back into input order later
        return Route(
            route_id=f"meea:{product}",
            product=product,
            synthesizable=bool(reactions),
            reactions=reactions,
            source=RouteSource(
                planner="MEEA",
                parameters={
                    "molecules": str(products_path),
                    "iter_limit": iter_limit,
                    "time_limit_min": time_limit_min,
                    "simulations": simulations,
                    "cpuct": cpuct,
                    "parallel": parallel,
                    "starting_materials": [path.name for path in starting_materials],
                },
            ),
        )
    
    def _run_prexsyn_backup(self, no_products: List[str], input_stem: str) -> "dict[str, list[tuple[str, str]]]":
        analogs_by_product: dict[str, list[tuple[str, str]]] = {}
        if not no_products:
            return analogs_by_product
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".smi", prefix="meea_prexsyn_backup_", delete=False,
        ) as handle:
            handle.write("\n".join(no_products) + "\n")
            backup_path = Path(handle.name)
        try:
            prexsyn = PrexSyn()
            prexsyn.setup_tool()
            prexsyn.agent_output_dir = str(self._output_dir(backup_path))
            prexsyn._output_stem_override = input_stem
            backup_routes, _ = prexsyn(molecules=str(backup_path), top=5, seed=0, device="cuda")
            if isinstance(backup_routes, Route):
                backup_route_list = [backup_routes]
            elif isinstance(backup_routes, list):
                backup_route_list = [r for r in backup_routes if isinstance(r, Route)]
            else:
                backup_route_list = []
            for backup_route in backup_route_list:
                source = backup_route.source.parameters if backup_route.source is not None else {}
                target_smiles = source.get("target_smiles")
                rank = source.get("rank")
                if not isinstance(target_smiles, str) or not isinstance(rank, int):
                    continue
                analogs_by_product.setdefault(target_smiles, []).append(
                    (backup_route.product, "||".join(backup_route.reactions) if backup_route.reactions else "")
                )
            for target_smiles in analogs_by_product:
                analogs_by_product[target_smiles] = analogs_by_product[target_smiles][:5]
        except Exception as exc:
            print(f"⚠️ PrexSyn backup failed: {exc}", flush=True)
            analogs_by_product = {}
        finally:
            backup_path.unlink(missing_ok=True)
        return analogs_by_product

    def csv_string(self, routes: List[Route], products: List[str], prexsyn_backup: bool = True, input_stem: str = "prexsyn_backup", analogs_by_product: "dict[str, list[tuple[str, str]]] | None" = None) -> str:
        product_order = {product: index for index, product in enumerate(products)}
        sorted_routes = sorted(
            routes,
            key=lambda route: (product_order.get(route.product, len(product_order)), route.route_id),
        )
        if analogs_by_product is None:
            if prexsyn_backup:
                no_products = [r.product for r in sorted_routes if not r.reactions]
                analogs_by_product = self._run_prexsyn_backup(no_products, input_stem)
            else:
                analogs_by_product = {}
        data = {
            "smiles": [],
            "synthesizable_decision": [],
            "source": [],
            "analog_smiles_1": [],
            "analog_smiles_2": [],
            "analog_smiles_3": [],
            "analog_smiles_4": [],
            "analog_smiles_5": [],
        }
        for r in sorted_routes:
            data["smiles"].append(r.product)
            analogs = analogs_by_product.get(r.product, []) if not r.reactions else []
            has_exact_prexsyn_match = any(product_value == r.product for product_value, _ in analogs)
            if r.reactions:
                decision, src = "YES", "MEEA"
            elif has_exact_prexsyn_match:
                decision, src = "YES", "PrexSyn"
            else:
                decision, src = "NO", ""
            data["synthesizable_decision"].append(decision)
            data["source"].append(src)
            display_analogs = [] if decision == "YES" else analogs
            for idx in range(5):
                product_value = display_analogs[idx][0] if idx < len(display_analogs) else ""
                data[f"analog_smiles_{idx + 1}"].append(product_value)
        df: pandas.DataFrame = pandas.DataFrame.from_dict(data)
        return df.to_csv(index=False)

    def _write_synthesis_routes(self, routes: List[Route], output_dir: Path, stem: str, analogs_by_product: "dict[str, list[tuple[str, str]]] | None" = None) -> "Path | None":
        rows = []
        for r in routes:
            if r.reactions:
                rows.append({"smiles": r.product, "source": "MEEA", "reactions": "||".join(r.reactions)})
        if analogs_by_product:
            for analogs in analogs_by_product.values():
                for analog_smiles, reactions_str in analogs:
                    if reactions_str:
                        rows.append({"smiles": analog_smiles, "source": "PrexSyn", "reactions": reactions_str})
        if not rows:
            return None
        out_file = output_dir.parent.parent / "synthesis_routes.csv"
        new_df = pandas.DataFrame(rows)
        if out_file.exists():
            existing = pandas.read_csv(out_file)
            combined = pandas.concat([existing, new_df]).drop_duplicates(subset="smiles", keep="first")
        else:
            combined = new_df.drop_duplicates(subset="smiles", keep="first")
        combined.to_csv(out_file, index=False)
        return out_file

    def __call__(self, **params):
        if "molecules" not in params:
            return 'Please specify the target product molecules file with an explicit keyword molecules="filename.smi".', '⚠️ Failed; no molecules="filename.smi".'

        products_path = self._products_path(params["molecules"])
        products = self._products(products_path)
        iter_limit = int(params.get("iter_limit", 500))
        time_limit_min = 0.5  # hardcoded: 30s per molecule
        simulations = int(params.get("simulations", 100))
        cpuct = float(params.get("cpuct", 4.0))
        parallel = 2  # hardcoded: optimal for small batches
        max_parallel = os.environ.get("MEEA_MAX_PARALLEL")
        if max_parallel is not None:
            parallel = min(parallel, max(1, int(max_parallel)))
        time_limit_sec = max(1, int(time_limit_min * 60))
        starting_materials = self._starting_material_paths(params.get("starting_materials"))
        required_largefiles = self._required_largefiles()
        image = getattr(self, "_runtime_image", None) or self._check_image()
        prexsyn_backup = not bool(params.get("SKIP_PREXSYN_BACKUP", False)) # Hidden parameter

        command = [
            self._apptainer(),
            "exec",
            "--cleanenv",
            "--containall",
            "--no-home",
            "--writable-tmpfs",
            "--env",
            "PYTHONNOUSERSITE=1",
        ]
        if os.environ.get("MEEA_APPTAINER_NV", "1").lower() not in {"0", "false", "no"}:
            command.append("--nv")

        prepared_products_path = self._prepared_products_input(products)
        try:
            command.extend(
                [
                    "--cwd",
                    "/opt/meea/MEEA",
                    "--bind",
                    f"{prepared_products_path}:/opt/meea/resources/products.txt:ro",
                    "--bind",
                    f"{required_largefiles['policy_model.ckpt']}:/opt/meea/MEEA/saved_model/policy_model.ckpt:ro",
                    "--bind",
                    f"{required_largefiles['value_pc.pt']}:/opt/meea/MEEA/saved_model/value_pc.pt:ro",
                    "--bind",
                    f"{required_largefiles['template_rules.dat']}:/opt/meea/MEEA/saved_model/template_rules.dat:ro",
                ]
            )
            for path in starting_materials:
                command.extend(["--bind", f"{path}:/opt/meea/resources/starting/{path.name}:ro"])

            command.extend(
                [
                    str(image),
                    "/bin/bash",
                    "meea.sh",
                    "/opt/meea/resources/products.txt",
                    "--iterations",
                    str(iter_limit),
                    "--simulations",
                    str(simulations),
                    "--starting-materials",
                    *[f"/opt/meea/resources/starting/{path.name}" for path in starting_materials],
                    "--time-limit",
                    str(time_limit_sec),
                    "--parallel",
                    str(parallel),
                ]
            )
            stdout_lines: list[str] = []
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    stdout_lines.append(line)
                returncode = process.wait()
        finally:
            prepared_products_path.unlink(missing_ok=True)

        if returncode != 0:
            return None, "⚠️ Failed; MEEA execution error."

        payloads = []
        for line in stdout_lines:
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not payloads:
            return None, "⚠️ Failed; MEEA returned no parseable JSON route output."

        routes = [
            self._route_from_payload(
                payload=payload,
                products_path=products_path,
                iter_limit=iter_limit,
                time_limit_min=time_limit_min,
                simulations=simulations,
                cpuct=cpuct,
                parallel=parallel,
                starting_materials=starting_materials,
            )
            for payload in payloads
        ]
        product_order = {p: i for i, p in enumerate(products)}
        no_products = sorted(
            [r.product for r in routes if not r.reactions],
            key=lambda p: product_order.get(p, len(product_order)),
        )
        analogs_by_product = self._run_prexsyn_backup(no_products, products_path.stem) if prexsyn_backup else {}
        found = sum(
            1 for r in routes
            if r.reactions or any(s == r.product for s, _ in analogs_by_product.get(r.product, []))
        )
        datastr = self.csv_string(routes, products, analogs_by_product=analogs_by_product)
        output_dir = self._output_dir(products_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        rtfile = output_dir / f"{products_path.stem}_meea.csv"
        with open(rtfile, "w") as f:
            f.write(datastr)
        self._write_synthesis_routes(routes, output_dir, products_path.stem, analogs_by_product=analogs_by_product)
        if len(routes) == 1:
            route = routes[0]
            if found:
                return route, f"✅ Route found, stored in {rtfile}."
            return route, f"✅ Executed successfully; no route found. Results stored in {rtfile}."
        if found:
            return routes, f"✅ Routes found for {found}/{len(routes)} targets, stored in {rtfile}."
        return routes, f"✅ Executed successfully; no routes found for {len(routes)} targets. Analog candidates stored in {rtfile}."
