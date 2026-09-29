from __future__ import annotations

import contextlib
import io
import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from synthesizer.agent.agent import Agent
from synthesizer.data.route import Route
from synthesizer.llm.base import LLM
from synthesizer.tools.MEEA import MEEA
from synthesizer.tools.base import Tool
from synthesizer.tools.code import Code


REAL_ASSET_DIR = Path(os.environ.get("MEEA_ASSET_DIR", "/path/to/synthesizer-assets"))
REAL_IMAGE = REAL_ASSET_DIR / "meea.sif"
REAL_LARGEFILES = {
    "eMoleculesBB.txt": REAL_ASSET_DIR / "eMoleculesBB.txt",
    "EnamineREAL.txt": REAL_ASSET_DIR / "EnamineREAL.txt",
    "policy_model.ckpt": REAL_ASSET_DIR / "policy_model.ckpt",
    "template_rules.dat": REAL_ASSET_DIR / "template_rules.dat",
    "value_pc.pt": REAL_ASSET_DIR / "value_pc.pt",
}


class ScriptedLLM(LLM):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []
        self.response_count = 0

    def get_single_response(self, user, system=None) -> tuple[str, dict]:
        self.calls.append((user, system))
        if not self.responses:
            raise AssertionError("ScriptedLLM received more calls than expected")
        self.response_count += 1
        response_text = self.responses.pop(0)
        metadata = {
            "id": f"resp-{self.response_count}",
            "usage": {
                "prompt_tokens": self.response_count * 10,
                "completion_tokens": self.response_count * 3,
                "total_tokens": self.response_count * 13,
            },
        }
        return response_text, metadata


class DelayedScriptedLLM(ScriptedLLM):
    def __init__(self, responses: list[str], delays: list[float]):
        super().__init__(responses)
        self.delays = list(delays)

    def get_single_response(self, user, system=None) -> tuple[str, dict]:
        if self.delays:
            import time
            time.sleep(self.delays.pop(0))
        return super().get_single_response(user, system)


class SetupTrackingTool(Tool):
    def __init__(self):
        super().__init__()
        self.setup_calls = 0

    def setup_tool(self):
        self.setup_calls += 1

    def documentation(self) -> str:
        return "`SetupTrackingTool()`"

    def __call__(self, **params):
        return None, "ok"


def routes_from_history(agent: Agent) -> list[Route]:
    routes = []
    for result, status in agent.action_history:
        if "✅" not in status:
            continue
        result_routes = result if isinstance(result, list) else [result]
        routes.extend(route for route in result_routes if isinstance(route, Route))
    return routes


class AgentTests(unittest.TestCase):
    def test_agent_initializes_enabled_tools(self) -> None:
        tool = SetupTrackingTool()

        Agent([tool], llm=ScriptedLLM(['Thought: done\nAction: `Finish(response="ok")`\n<PAUSE>']))

        self.assertEqual(tool.setup_calls, 1)

    def test_agent_extracts_action_when_argument_contains_backticks(self) -> None:
        llm = ScriptedLLM(
            ['Thought: I can finish directly.\nAction: `Finish(response="Use `PrexSyn` for follow-up.")`\n<PAUSE>']
        )
        agent = Agent([], llm=llm)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                response, metadata = agent("Answer the task.", tmpdir, turn_limit=1)

        self.assertEqual(response, "Use `PrexSyn` for follow-up.")
        self.assertTrue(agent.finished)
        self.assertEqual(len(metadata["messages"]), 1)

    def test_agent_returns_metadata_with_token_usage_for_finish_only(self) -> None:
        llm = ScriptedLLM(
            ['Thought: I can answer immediately.\nAction: `Finish(response="Done.")`\n<PAUSE>']
        )
        agent = Agent([], llm=llm)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()):
                response, metadata = agent("Answer the task.", tmpdir, turn_limit=1)

        self.assertEqual(response, "Done.")
        self.assertTrue(agent.finished)
        self.assertEqual(
            metadata,
            {
                "messages": [
                    {
                        "id": "resp-1",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                        },
                    }
                ]
            },
        )
        self.assertEqual(metadata["messages"][0]["usage"]["total_tokens"], 13)

    def test_agent_iteration_limit_requests_plaintext_summary_report(self) -> None:
        llm = ScriptedLLM(
            [
                'Thought: I should inspect a tool first.\nAction: `SetupTrackingTool()`\n<PAUSE>',
                "Summary report: attempted one tool call and produced no final answer yet.",
            ]
        )
        tool = SetupTrackingTool()
        agent = Agent([tool], llm=llm)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                response, metadata = agent("Answer the task.", tmpdir, turn_limit=1)

        self.assertFalse(agent.finished)
        self.assertEqual(
            response,
            "The agent reached its iteration limit attempting to complete the task."
            "Summary report: attempted one tool call and produced no final answer yet.",
        )
        self.assertEqual(len(metadata["messages"]), 2)
        self.assertEqual(metadata["messages"][1]["id"], "resp-2")
        self.assertEqual(len(llm.calls), 2)
        self.assertIsNone(llm.calls[1][1])

        finalprompt = llm.calls[1][0]
        self.assertIn("within the allotted 1 iterations", finalprompt)
        self.assertIn("Original user prompt: Answer the task.", finalprompt)
        self.assertIn(f"Directory for inputs/outputs: {tmpdir}", finalprompt)
        self.assertIn("Action History:", finalprompt)
        self.assertIn("SetupTrackingTool() → ok", finalprompt)
        self.assertIn("Generate your response as a plaintext report", finalprompt)

    def test_agent_timeout_requests_plaintext_summary_report(self) -> None:
        llm = DelayedScriptedLLM(
            [
                "1",
                'Thought: I should inspect a tool first.\nAction: `SetupTrackingTool()`\n<PAUSE>',
                "Summary report: attempted one tool call before timing out.",
            ],
            [0.0, 0.0, 1.2],
        )
        tool = SetupTrackingTool()
        agent = Agent([tool], llm=llm)

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                response, metadata = agent("Answer the task.", tmpdir, turn_limit=3)

        self.assertFalse(agent.finished)
        self.assertEqual(
            response,
            "The agent reached its time limit attempting to complete the task."
            "Summary report: attempted one tool call before timing out.",
        )
        self.assertEqual(len(metadata["responses"]), 3)
        self.assertEqual(metadata["responses"][2]["id"], "resp-3")
        self.assertEqual(len(llm.calls), 3)
        self.assertIsNone(llm.calls[2][1])

        finalprompt = llm.calls[2][0]
        self.assertIn("within the allotted 3 iterations", finalprompt)
        self.assertIn("Original user prompt: Answer the task.", finalprompt)
        self.assertIn(f"Directory for inputs/outputs: {tmpdir}", finalprompt)
        self.assertIn("Action History:", finalprompt)
        self.assertIn("SetupTrackingTool() → ok", finalprompt)
        self.assertIn("Generate your response as a plaintext report", finalprompt)

    def real_meea_tool(self) -> MEEA:
        missing = [str(path) for path in [REAL_IMAGE, *REAL_LARGEFILES.values()] if not path.exists()]
        if missing:
            self.skipTest("MEEA integration assets are missing: " + ", ".join(missing))
        if shutil.which("apptainer") is None and shutil.which("singularity") is None:
            self.skipTest("MEEA integration test requires apptainer or singularity on PATH.")

        previous_image = os.environ.get("MEEA_IMAGE")
        os.environ["MEEA_IMAGE"] = str(REAL_IMAGE)
        if previous_image is None:
            self.addCleanup(os.environ.pop, "MEEA_IMAGE", None)
        else:
            self.addCleanup(os.environ.__setitem__, "MEEA_IMAGE", previous_image)

        tool = MEEA(largefiles={key: str(path) for key, path in REAL_LARGEFILES.items()})
        try:
            tool.setup_tool()
        except RuntimeError as exc:
            self.skipTest(f"MEEA integration setup failed: {exc}")
        return tool

    def test_agent_runs_scripted_code_and_meea_actions_across_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target_paths = []
            for index, smiles in enumerate(["CCC", "CCO", "CCN", "c1ccccc1", "CCCl"], start=1):
                path = Path(tmpdir) / f"targets_{index}.smi"
                path.write_text(smiles + "\n")
                target_paths.append(path)

            code_actions = [
                'Code(code="value = 1 + 1")',
                'Code(code="items = [name.upper() for name in [\'cc\', \'co\']]")',
                'Code(code="import math\\narea = math.pi * 4")',
                f'Code(code="from pathlib import Path\\npath = Path({str(target_paths[3])!r}).resolve()")',
                'Code(code="scores = {\'route\': 3}\\nbest = max(scores.values())")',
            ]
            meea_actions = [
                (
                    f'MEEA(molecules="{path}", iter_limit={index}, time_limit_min=1, '
                    f'simulations=1, parallel=1, starting_materials=["eMoleculesBB"])'
                )
                for index, path in enumerate(target_paths, start=1)
            ]
            actions = [
                code_actions[0],
                meea_actions[0],
                code_actions[1],
                meea_actions[1],
                code_actions[2],
                meea_actions[2],
                code_actions[3],
                meea_actions[3],
                code_actions[4],
                meea_actions[4],
                'Finish(response="Planned five targets after five code checks.")',
            ]
            responses = [
                f"Thought: First inspect simple arithmetic before planning.\nAction: `{actions[0]}`\n<PAUSE>",
                f"Thought: The first check succeeded, so plan the first target.\nAction: `{actions[1]}`\n<PAUSE>",
                f"Thought: The route came back; normalize a small list next.\nAction: `{actions[2]}`\n<PAUSE>",
                f"Thought: With normalized names ready, run the second MEEA search.\nAction: `{actions[3]}`\n<PAUSE>",
                f"Thought: Two turns of evidence are enough to compute a quick constant.\nAction: `{actions[4]}`\n<PAUSE>",
                f"Thought: Use that context to launch the third planning call.\nAction: `{actions[5]}`\n<PAUSE>",
                f"Thought: Three routes are available; check filesystem handling.\nAction: `{actions[6]}`\n<PAUSE>",
                f"Thought: The path check worked, so try another target file.\nAction: `{actions[7]}`\n<PAUSE>",
                f"Thought: One final score calculation before the last planner call.\nAction: `{actions[8]}`\n<PAUSE>",
                f"Thought: The score check passed; run the fifth MEEA call.\nAction: `{actions[9]}`\n<PAUSE>",
                f"Thought: Five code checks and five MEEA calls are complete.\nAction: `{actions[10]}`\n<PAUSE>",
            ]
            llm = ScriptedLLM(responses)
            meea = self.real_meea_tool()
            agent = Agent([Code(), meea], llm=llm)

            with contextlib.redirect_stdout(io.StringIO()):
                response, metadata = agent("Plan these targets.", tmpdir, turn_limit=12)
            tmpdir_text = Path(tmpdir).as_posix()
            expected_products = [path.read_text().strip() for path in target_paths]
        routes = routes_from_history(agent)

        self.assertEqual(llm.responses, [])
        self.assertTrue(agent.finished)
        self.assertIn("Planned five targets", response)
        self.assertEqual(len(metadata["messages"]), len(actions))
        self.assertTrue(all("usage" in message for message in metadata["messages"]))
        self.assertEqual(
            [message["usage"]["total_tokens"] for message in metadata["messages"]],
            [13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143],
        )
        self.assertEqual(
            [message["usage"]["prompt_tokens"] for message in metadata["messages"]],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        )
        self.assertEqual(
            [message["usage"]["completion_tokens"] for message in metadata["messages"]],
            [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        )
        self.assertEqual(len(routes), 5)
        self.assertEqual([route.product for route in routes], expected_products)
        self.assertEqual([route.source.planner for route in routes], ["MEEA"] * 5)
        self.assertEqual([route.source.parameters["iter_limit"] for route in routes], [1, 2, 3, 4, 5])
        self.assertEqual(len(agent.action_history), 11)
        self.assertEqual(sum("Code(" in history for _, history in agent.action_history), 5)
        self.assertEqual(sum("MEEA(" in history for _, history in agent.action_history), 5)
        self.assertTrue(
            all("Full result of last action:" in system for _, system in llm.calls[1:])
        )
        self.assertTrue(all(tmpdir_text in user for user, _ in llm.calls))


if __name__ == "__main__":
    unittest.main()
