from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import csv

from synthesizer.tools.PrexSyn import PrexSyn
from synthesizer.data.route import Route, RouteSource


DEFAULT_REAL_REPO_DIR = Path(__file__).resolve().parents[1] / "tools" / "PrexSyn" / "prexsyn"
REAL_REPO_DIR = Path(os.environ.get("PREXSYN_REPO", str(DEFAULT_REAL_REPO_DIR)))
REAL_IMAGE = Path(
    os.environ.get("PREXSYN_IMAGE", "/path/to/synthesizer-assets/prexsyn.sif")
)


class PrexSynToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self.repo = self.base / "prexsyn"
        self.repo.mkdir()
        self.image = self.base / "prexsyn.sif"
        self.image.write_text("stub\n")
        self.tool = PrexSyn(repo_dir=str(self.repo), image=str(self.image))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def real_tool(self) -> PrexSyn:
        missing = [str(path) for path in [REAL_REPO_DIR, REAL_IMAGE] if not path.exists()]
        if missing:
            self.skipTest("PrexSyn integration assets are missing: " + ", ".join(missing))
        if shutil.which("apptainer") is None and shutil.which("singularity") is None:
            self.skipTest("PrexSyn integration test requires apptainer or singularity on PATH.")

        tool = PrexSyn(repo_dir=str(REAL_REPO_DIR), image=str(REAL_IMAGE))
        try:
            tool.setup_tool()
        except RuntimeError as exc:
            self.skipTest(f"PrexSyn integration setup failed: {exc}")
        return tool

    def test_smiles_list_accepts_single_smiles(self) -> None:
        smiles, source = self.tool._smiles_list("CCO", None)
        self.assertEqual(smiles, ["CCO"])
        self.assertEqual(source, "smiles")

    def test_smiles_list_reads_molecules_file(self) -> None:
        molecules = self.base / "molecules.smi"
        molecules.write_text("\nCCO\nc1ccccc1O\n")

        smiles, source = self.tool._smiles_list(None, molecules)

        self.assertEqual(smiles, ["CCO", "c1ccccc1O"])
        self.assertEqual(source, str(molecules.resolve()))

    def test_smiles_list_rejects_missing_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.tool._smiles_list(None, None)

    def test_smiles_list_rejects_both_input_modes(self) -> None:
        molecules = self.base / "molecules.smi"
        molecules.write_text("CCO\n")

        with self.assertRaises(ValueError):
            self.tool._smiles_list("CCO", molecules)

    def test_io_dir_uses_prexsyn_extension_for_molecules_file(self) -> None:
        molecules = self.base / "molecules.smi"
        molecules.write_text("CCO\n")

        self.assertEqual(self.tool._io_dir(None, molecules), self.base / "molecules.prexsyn")

    def test_io_dir_uses_temp_fallback_for_direct_smiles(self) -> None:
        self.assertIsNone(self.tool._io_dir("CCO", None))

    def test_apptainer_exec_base_uses_nv_for_cuda(self) -> None:
        command = self.tool._apptainer_exec_base(device="cuda")

        self.assertIn("--nv", command)
        self.assertNotIn(str(self.image), command)

    def test_apptainer_exec_base_skips_nv_for_cpu(self) -> None:
        command = self.tool._apptainer_exec_base(device="cpu")

        self.assertNotIn("--nv", command)
        self.assertNotIn(str(self.image), command)

    def test_flatten_reactions_collects_nested_reaction_names(self) -> None:
        tree = {
            "SMILES": "A",
            "Reaction": "rxn-top",
            "Precursors": [
                {
                    "r1": {"SMILES": "B", "BuildingBlock": "BB1"},
                    "r2": {
                        "SMILES": "C",
                        "Reaction": "rxn-child",
                        "Precursors": [{"r3": {"SMILES": "D", "BuildingBlock": "BB2"}}],
                    },
                }
            ],
        }

        self.assertEqual(self.tool._flatten_reactions(tree), ["rxn-top", "rxn-child"])

    def test_call_requires_explicit_input(self) -> None:
        route, status = self.tool()
        self.assertIsNone(route)
        self.assertIn('molecules="filename.smi"', status)

    def test_route_from_result_records_similarity_metadata(self) -> None:
        route = self.tool._route_from_result(
            target_smiles="CCO",
            rank=1,
            result={
                "rank": 1,
                "product_smiles": "CCCO",
                "similarity": 0.5,
                "tree": {"SMILES": "CCCO", "BuildingBlock": "BB1"},
            },
            input_source="smiles",
            config_path="data/trained_models/enamine2310_rxn115_202511.yml",
            num_samples=8,
            top=3,
            device="cpu",
        )

        self.assertEqual(route.route_id, "prexsyn:CCO:1:CCCO")
        self.assertEqual(route.product, "CCCO")
        self.assertTrue(route.synthesizable)
        self.assertEqual(route.reactions, [])
        self.assertEqual(route.source.planner, "PrexSyn")
        self.assertEqual(route.source.parameters["similarity"], 0.5)
        self.assertEqual(route.source.parameters["tree"]["BuildingBlock"], "BB1")

    def test_csv_string_includes_yes_no_decision_columns(self) -> None:
        routes = [
            Route(
                route_id="prexsyn:target:1:analog",
                product="CCCO",
                synthesizable=True,
                reactions=["CCO>>CCCO"],
                source=RouteSource(
                    planner="PrexSyn",
                    parameters={"target_smiles": "CCO", "similarity": 0.7, "rank": 1},
                ),
            )
        ]

        csv_text = self.tool.csv_string(routes)
        rows = list(csv.DictReader(csv_text.splitlines()))

        self.assertEqual(rows[0]["analog_of"], "CCO")
        self.assertEqual(rows[0]["target_synthesizable_decision"], "NO")
        self.assertEqual(rows[0]["analog_synthesizable_decision"], "YES")

    def test_csv_string_marks_target_yes_when_analog_matches_target(self) -> None:
        routes = [
            Route(
                route_id="prexsyn:target:1:target",
                product="CCO",
                synthesizable=True,
                reactions=["CC>>CCO"],
                source=RouteSource(
                    planner="PrexSyn",
                    parameters={"target_smiles": "CCO", "similarity": 1.0, "rank": 1},
                ),
            )
        ]

        csv_text = self.tool.csv_string(routes)
        rows = list(csv.DictReader(csv_text.splitlines()))

        self.assertEqual(rows[0]["analog_of"], "CCO")
        self.assertEqual(rows[0]["target_synthesizable_decision"], "YES")
        self.assertEqual(rows[0]["analog_synthesizable_decision"], "YES")

    def test_call_uses_real_prexsyn_execution(self) -> None:
        tool = self.real_tool()

        route, status = tool(smiles="CCO", top=1, num_samples=4, device="cpu")

        self.assertIsNotNone(route, status)
        self.assertTrue(status.startswith("✅"), status)
        self.assertEqual(route.source.planner, "PrexSyn")
        self.assertEqual(route.source.parameters["target_smiles"], "CCO")
        self.assertEqual(route.source.parameters["rank"], 1)
        self.assertEqual(route.source.parameters["device"], "cpu")
        self.assertGreaterEqual(route.source.parameters["similarity"], 0.0)
        self.assertIn("tree", route.source.parameters)
