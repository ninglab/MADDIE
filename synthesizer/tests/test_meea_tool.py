from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
import csv
from unittest.mock import patch

from synthesizer.tools.MEEA import MEEA
from synthesizer.data.route import Route, RouteSource


REAL_ASSET_DIR = Path("/path/to/synthesizer-assets")
REAL_IMAGE = REAL_ASSET_DIR / "meea.sif"
REAL_LARGEFILES = {
    "eMoleculesBB.txt": REAL_ASSET_DIR / "eMoleculesBB.txt",
    "EnamineREAL.txt": REAL_ASSET_DIR / "EnamineREAL.txt",
    "policy_model.ckpt": REAL_ASSET_DIR / "policy_model.ckpt",
    "template_rules.dat": REAL_ASSET_DIR / "template_rules.dat",
    "value_pc.pt": REAL_ASSET_DIR / "value_pc.pt",
}


class MEEAToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.largefiles = {}
        for name in MEEA.REQUIRED_LARGEFILES:
            path = self.base / name
            path.write_text("stub\n")
            self.largefiles[name] = str(path)
        self.tool = MEEA(largefiles=self.largefiles)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def real_tool(self) -> MEEA:
        missing = [str(path) for path in [REAL_IMAGE, *REAL_LARGEFILES.values()] if not path.exists()]
        if missing:
            self.skipTest("MEEA integration assets are missing: " + ", ".join(missing))

        previous_image = os.environ.get("MEEA_IMAGE")
        os.environ["MEEA_IMAGE"] = str(REAL_IMAGE)
        if previous_image is None:
            self.addCleanup(os.environ.pop, "MEEA_IMAGE", None)
        else:
            self.addCleanup(os.environ.__setitem__, "MEEA_IMAGE", previous_image)

        return MEEA(largefiles={key: str(path) for key, path in REAL_LARGEFILES.items()})

    def test_starting_materials_default_to_emolecules(self) -> None:
        paths = self.tool._starting_material_paths(None)
        self.assertEqual([path.name for path in paths], ["eMoleculesBB.txt"])

    def test_starting_materials_accept_aliases_and_explicit_paths(self) -> None:
        custom = self.base / "custom.txt"
        custom.write_text("custom\n")

        paths = self.tool._starting_material_paths(["EnamineREAL", str(custom)])

        self.assertEqual(paths[0], Path(self.largefiles["EnamineREAL.txt"]).resolve())
        self.assertEqual(paths[1], custom.resolve())

    def test_required_largefiles_raise_when_missing(self) -> None:
        incomplete = dict(self.largefiles)
        incomplete.pop("policy_model.ckpt")
        tool = MEEA(largefiles=incomplete)

        with self.assertRaises(FileNotFoundError) as ctx:
            tool._required_largefiles()

        self.assertIn("policy_model.ckpt", str(ctx.exception))

    def test_call_requires_molecules_keyword(self) -> None:
        route, status = self.tool()

        self.assertIn('molecules="filename.smi"', route)
        self.assertIn('molecules="filename.smi"', status)

    def test_products_path_requires_existing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.tool._products_path(self.base / "missing.smi")

    def test_products_reads_nonempty_lines(self) -> None:
        products = self.base / "products.smi"
        products.write_text("\nCCC\nCCO\n")

        self.assertEqual(self.tool._products(products), ["CCC", "CCO"])

    def test_prepared_products_input_strips_header_rows(self) -> None:
        products = self.base / "products.smi"
        products.write_text("smiles\nCCC\nCCO\n")

        values = self.tool._products(products)
        prepared = self.tool._prepared_products_input(values)
        self.addCleanup(prepared.unlink, missing_ok=True)

        self.assertEqual(prepared.read_text(), "CCC\nCCO\n")

    def test_route_from_payload_records_parallel_batch_parameters(self) -> None:
        products = self.base / "products.smi"
        products.write_text("CCC\nCCO\n")
        starting_materials = self.tool._starting_material_paths(None)

        route = self.tool._route_from_payload(
            payload={"product": "CCO", "reactions": ["CCO>>CC"]},
            fallback_product="CCC",
            products_path=products.resolve(),
            iter_limit=1,
            time_limit_min=1,
            simulations=10,
            cpuct=4.0,
            parallel=2,
            starting_materials=starting_materials,
        )

        self.assertEqual(route.route_id, "meea:CCO")
        self.assertEqual(route.product, "CCO")
        self.assertTrue(route.synthesizable)
        self.assertEqual(route.source.parameters["parallel"], 2)

    def test_csv_string_includes_yes_no_synthesizable_decision(self) -> None:
        routes = [
            Route(
                route_id="meea:CCO",
                product="CCO",
                synthesizable=True,
                reactions=["CC>>CCO"],
                source=RouteSource(planner="MEEA"),
            ),
            Route(
                route_id="meea:CCC",
                product="CCC",
                synthesizable=False,
                reactions=None,
                source=RouteSource(planner="MEEA"),
            ),
        ]

        csv_text = self.tool.csv_string(routes, ["CCO", "CCC"], prexsyn_backup=False)
        rows = list(csv.DictReader(csv_text.splitlines()))

        self.assertEqual(rows[0]["product_molecule"], "CCO")
        self.assertEqual(rows[0]["synthesizable_decision"], "YES")
        self.assertEqual(rows[1]["product_molecule"], "CCC")
        self.assertEqual(rows[1]["synthesizable_decision"], "NO")

    def test_csv_string_marks_yes_when_prexsyn_exact_match_exists(self) -> None:
        class _FakePrexSyn:
            def setup_tool(self):
                return None

            def __call__(self, **kwargs):
                return [
                    Route(
                        route_id="prexsyn:CCC:1:CCC",
                        product="CCC",
                        synthesizable=True,
                        reactions=["CC>>CCC"],
                        source=RouteSource(
                            planner="PrexSyn",
                            parameters={"target_smiles": "CCC", "rank": 1, "similarity": 1.0},
                        ),
                    )
                ], "ok"

        with patch("synthesizer.tools.MEEA.PrexSyn", _FakePrexSyn):
            routes = [
                Route(
                    route_id="meea:CCC",
                    product="CCC",
                    synthesizable=False,
                    reactions=None,
                    source=RouteSource(planner="MEEA"),
                ),
            ]
            csv_text = self.tool.csv_string(routes, ["CCC"], prexsyn_backup=True)

        rows = list(csv.DictReader(csv_text.splitlines()))
        self.assertEqual(rows[0]["product_molecule"], "CCC")
        self.assertEqual(rows[0]["synthesizable_decision"], "YES")
        self.assertEqual(rows[0]["analog_1_product"], "CCC")

    def test_call_runs_container_for_multiple_targets(self) -> None:
        tool = self.real_tool()
        products = self.base / "products.smi"
        products.write_text("CCC\nCCO\n")

        routes, status = tool(
            molecules=str(products),
            iter_limit=1,
            time_limit_min=1,
            parallel=2,
            starting_materials=["eMoleculesBB"],
        )

        self.assertIsInstance(routes, list, status)
        self.assertEqual({route.product for route in routes}, {"CCC", "CCO"})
        self.assertTrue(status.startswith("✅"), status)
        for route in routes:
            self.assertEqual(route.source.planner, "MEEA")
            self.assertEqual(route.source.parameters["parallel"], 2)
            self.assertEqual(route.source.parameters["molecules"], str(products.resolve()))

    def test_setup_tool_uses_real_environment(self) -> None:
        self.real_tool().setup_tool()

    def test_call_uses_real_meea_execution(self) -> None:
        tool = self.real_tool()
        products = self.base / "products.smi"
        products.write_text("CCC1=CC2=CC=CC(=C2C(=C1)C)C\n")

        route, status = tool(
            molecules=str(products),
            iter_limit=1,
            time_limit_min=1,
            parallel=2,
            starting_materials=["eMoleculesBB"],
        )

        self.assertIsNotNone(route, status)
        self.assertIn(
            status[:10],
            {"✅ Route found."[:10], "✅ Executed successfully; no route found."[:10]},
        )
        self.assertEqual(route.route_id, "meea:CCC1=CC2=CC=CC(=C2C(=C1)C)C")
        self.assertEqual(route.product, "CCC1=CC2=CC=CC(=C2C(=C1)C)C")
        self.assertEqual(route.source.planner, "MEEA")
        self.assertEqual(route.source.parameters["molecules"], str(products.resolve()))
        self.assertEqual(route.source.parameters["iter_limit"], 1)
        self.assertEqual(route.source.parameters["time_limit_min"], 1)
        self.assertEqual(route.source.parameters["parallel"], 2)
        self.assertEqual(
            route.source.parameters["starting_materials"],
            ["eMoleculesBB.txt"]
        )
        if "stored in " in status:
            json_path = Path(status.removesuffix(".").rsplit("stored in ", 1)[1])
            self.addCleanup(json_path.unlink, missing_ok=True)
            self.assertTrue(json_path.is_file(), status)
            self.assertEqual(json.loads(json_path.read_text()), route.to_dict())
        # TODO: Add assertion checking the JSON file
