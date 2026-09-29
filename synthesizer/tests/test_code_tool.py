import unittest

from synthesizer.tools.code import Code


class CodeToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = Code()

    def test_call_requires_code_keyword(self) -> None:
        result, status = self.tool()

        self.assertIn('code="..."', result)
        self.assertIn('code="..."', status)

    def test_call_returns_stdout_as_result_and_status_summary(self) -> None:
        result, status = self.tool(code='print("hello")')

        self.assertEqual(result, "hello\n")
        self.assertEqual(status, "✅ Executed successfully; 6 chars of stdout.")

    def test_call_succeeds_without_stdout(self) -> None:
        result, status = self.tool(code="x = 1 + 1")

        self.assertIsNone(result)
        self.assertEqual(status, "✅ Executed successfully; no stdout.")

    def test_call_returns_stdout_when_exception_occurs_after_print(self) -> None:
        result, status = self.tool(code='print("before"); raise ValueError("bad")')

        self.assertEqual(result, "before\n")
        self.assertIn("⚠️Failed with ValueError: bad", status)
        self.assertIn("Stdout:\nbefore\n", status)

    def test_call_returns_none_when_exception_occurs_without_stdout(self) -> None:
        result, status = self.tool(code='raise RuntimeError("boom")')

        self.assertIsNone(result)
        self.assertEqual(status, "⚠️Failed with RuntimeError: boom")

    def test_call_returns_descriptive_syntax_error(self) -> None:
        result, status = self.tool(code="if True print('missing colon')")

        self.assertIsNone(result)
        self.assertIn("⚠️Failed with SyntaxError:", status)
        self.assertIn("line 1", status)
        self.assertIn("if True print('missing colon')", status)
        self.assertIn("^", status)


if __name__ == "__main__":
    unittest.main()
