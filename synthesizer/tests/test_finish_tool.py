from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthesizer.tools.finish import Finish


class _DummyAgent:
    def __init__(self) -> None:
        self.finished = False
        self.response = None


class FinishToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _DummyAgent()
        self.tool = Finish(agent=self.agent)

    def test_call_requires_response_keyword(self) -> None:
        route, status = self.tool()

        self.assertIn('response="Natural language response..."', route)
        self.assertIn('response="Natural language response..."', status)
        self.assertFalse(self.agent.finished)
        self.assertIsNone(self.agent.response)

    def test_call_marks_agent_finished_and_stores_response(self) -> None:
        route, status = self.tool(response="All done.")

        self.assertIsNone(route)
        self.assertEqual(status, "🏁🏁🏁")
        self.assertTrue(self.agent.finished)
        self.assertEqual(self.agent.response, "All done.")
