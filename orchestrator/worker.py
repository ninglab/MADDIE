from abc import ABC
from datetime import datetime
from typing import Literal

import os
import sys
import tempfile
import subprocess
import threading


class BaseWorker(ABC):
    def __init__(
        self,
        cli: str,
        env_dir: str,
        context_dir: str,
        model: str,
        log_dir: str | None = None,
        extra_args: list[str] | None = None,
    ):
        """Initialize the worker with paths needed to invoke a CLI agent.

        Args:
            cli: Path to the CLI script that runs the agent.
            env_dir: Path to the conda/virtual environment used to run the agent.
            context_dir: Path to the directory containing context files for the agent.
            log_dir: Directory in which to store the agent's stdout/stderr for each
                call (one ``.out`` and one ``.error`` file per call). When None, no
                per-call log files are written.
            extra_args: Additional CLI arguments appended verbatim to each invocation.
        """
        self.cli = cli
        self.env_dir = env_dir
        self.context_dir = context_dir
        self.model = model
        self.log_dir = log_dir
        self.extra_args: list[str] = extra_args or []

    def _stream(self, pipe, dest, lines: list, log_fh=None):
        for line in pipe:
            dest.write(line)
            dest.flush()
            lines.append(line)
            # each stream tees to its own file, so no locking is needed
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()

    def _run(self, cmd: list[str], out_fh=None, err_fh=None) -> tuple[int, str, str]:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout_lines, stderr_lines = [], []
        t_out = threading.Thread(
            target=self._stream, args=(process.stdout, sys.stdout, stdout_lines, out_fh)
        )
        t_err = threading.Thread(
            target=self._stream, args=(process.stderr, sys.stderr, stderr_lines, err_fh)
        )
        t_out.start()
        t_err.start()
        t_out.join()
        t_err.join()
        process.wait()
        return process.returncode, "".join(stdout_lines), "".join(stderr_lines)

    def run(
        self, prompt: str, option: Literal["python", "conda"] = "conda"
    ) -> str:
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt") as tmp:
            if option == "python":
                py_dir = os.path.join(self.env_dir, "bin", "python")
                cmd = [
                    py_dir,
                    self.cli,
                    "--input_prompt",
                    prompt,
                    "--context_dir",
                    self.context_dir,
                    "--output",
                    tmp.name,
                    "--model",
                    self.model
                ] + self.extra_args
            elif option == "conda":
                cmd = [
                    "conda",
                    "run",
                    "-p",
                    self.env_dir,
                    "--no-capture-output",
                    "python",
                    self.cli,
                    "--input_prompt",
                    prompt,
                    "--context_dir",
                    self.context_dir,
                    "--output",
                    tmp.name,
                    "--model",
                    self.model
                ] + self.extra_args

            # open per-call log files (stdout -> .out, stderr -> .error) when log_dir is set
            out_fh = err_fh = None
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # have the agent dump its run metadata alongside the per-call logs
                cmd += [
                    "--path_to_metadata",
                    os.path.join(self.log_dir, f"{ts}.metadata.json"),
                ]
                header = (
                    f"# agent: {os.path.basename(self.log_dir.rstrip('/'))}\n"
                    f"# model: {self.model}\n"
                    f"# time: {ts}\n"
                    f"# prompt:\n{prompt}\n"
                    "===== START =====\n"
                )
                out_fh = open(os.path.join(self.log_dir, f"{ts}.out"), "w")
                err_fh = open(os.path.join(self.log_dir, f"{ts}.error"), "w")
                for fh in (out_fh, err_fh):
                    fh.write(header)
                    fh.flush()

            returncode = None
            response = ""
            try:
                returncode, stdout, stderr = self._run(cmd, out_fh, err_fh)
                if returncode != 0:
                    response = (
                        f"Agent failed to run during execution. Error message:\n{stderr}."
                    )
                else:
                    response = tmp.read()
            finally:
                # record the final response in the .out log before closing
                if out_fh is not None:
                    out_fh.write(f"\n===== RESPONSE =====\n{response}\n")
                for fh in (out_fh, err_fh):
                    if fh is not None:
                        fh.write(f"\n===== END (returncode={returncode}) =====\n")
                        fh.close()

            return response


_BASE = os.environ.get(
    "MADDIE_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

class SynthesizerWorker(BaseWorker):
    def __init__(
        self,
        model: str,
        context_dir: str,
        cli: str = f"{_BASE}/cli_synthesizer.py",
        env_dir: str = os.environ.get("MADDIE_ENV_SYNTHESIZER", "/path/to/envs/maddie-synthesizer"),
        log_dir: str | None = None,
    ):
        super().__init__(cli=cli, env_dir=env_dir, context_dir=context_dir, model=model, log_dir=log_dir)


class EvaluatorWorker(BaseWorker):
    def __init__(
        self,
        model: str,
        context_dir: str,
        cli: str = f"{_BASE}/cli_evaluator.py",
        env_dir: str = os.environ.get("MADDIE_ENV_EVALUATOR", "/path/to/envs/maddie-evaluator"),
        log_dir: str | None = None,
    ):
        super().__init__(cli=cli, env_dir=env_dir, context_dir=context_dir, model=model, log_dir=log_dir)


class LeadOptimizerWorker(BaseWorker):
    def __init__(
        self,
        model: str,
        context_dir: str,
        cli: str = f"{_BASE}/cli_lead_optimizer.py",
        env_dir: str = os.environ.get("MADDIE_ENV_LEAD_OPTIMIZER", "/path/to/envs/maddie-lead-optimizer"),
        log_dir: str | None = None,
        num_outputs: int = 5,
    ):
        super().__init__(
            cli=cli, env_dir=env_dir, context_dir=context_dir, model=model,
            log_dir=log_dir, extra_args=["--num_outputs", str(num_outputs)],
        )


class HitIdentifierWorker(BaseWorker):
    def __init__(
        self,
        model: str,
        context_dir: str,
        cli: str = f"{_BASE}/cli_hit_identifier.py",
        env_dir: str = os.environ.get("MADDIE_ENV_HIT_IDENTIFIER", "/path/to/envs/maddie-hit-identifier"),
        log_dir: str | None = None,
    ):
        super().__init__(cli=cli, env_dir=env_dir, context_dir=context_dir, model=model, log_dir=log_dir)


if __name__ == "__main__":
    pass
