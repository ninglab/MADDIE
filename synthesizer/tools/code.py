from contextlib import redirect_stdout
from io import StringIO
import pandas as pd
from rdkit import Chem
import numpy as np

from .base import Tool

docs = """`Code(code="python code")`
Code executes Python code in the current environment and returns anything printed to standard output in the action status.
You may import rdkit, pandas, numpy, etc.  You may read and write files from the local environment, provided you use the full path to the environment.
This tool is most useful for basic file maintanenance, sorting and filtering results, etc.
"""


class Code(Tool):
    def documentation(self):
        return docs

    @staticmethod
    def _format_syntax_error(exc: SyntaxError) -> str:
        parts = [f"⚠️Failed with SyntaxError: {exc.msg}"]
        if exc.lineno is not None:
            location = f"line {exc.lineno}"
            if exc.offset is not None:
                location += f", column {exc.offset}"
            parts[0] += f" at {location}"

        text = (exc.text or "").rstrip("\n")
        if text:
            parts.append(text)
            if exc.offset is not None and exc.offset > 0:
                parts.append(" " * (exc.offset - 1) + "^")

        return "\n".join(parts)

    def __call__(self, **params):
        if "code" not in params:
            return 'Please specify Python code using the explicit keyword code="..."', '⚠️Failed, no code="..."'

        output = StringIO()
        namespace = {}
        try:
            compiled = compile(params["code"], "<Code>", "exec")
        except SyntaxError as exc:
            return None, self._format_syntax_error(exc)

        try:
            with redirect_stdout(output):
                exec(compiled, namespace, namespace)
        except Exception as exc:
            stdout = output.getvalue()
            if stdout:
                return stdout, f"⚠️Failed with {exc.__class__.__name__}: {exc}\nStdout:\n{stdout}"
            return None, f"⚠️Failed with {exc.__class__.__name__}: {exc}"

        stdout = output.getvalue()
        if stdout:
            return stdout, f"✅ Executed successfully; {len(stdout)} chars of stdout."
        return None, "✅ Executed successfully; no stdout."
