import errno
import os
import pty
import subprocess
import sys
from collections import deque
from typing import Deque, Dict, List


def _append_tail(buffer: Deque[str], text: str, limit: int) -> None:
    if not text:
        return
    buffer.append(text)
    total = sum(len(part) for part in buffer)
    while total > limit and buffer:
        removed = buffer.popleft()
        total -= len(removed)


def run_with_live_output(cmd: List[str], cwd: str, tail_limit: int = 4000, env: Dict[str, str] = None) -> Dict[str, object]:
    """Run a child process on a PTY so tqdm-style progress bars render live."""
    master_fd, slave_fd = pty.openpty()
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        env=child_env,
        text=False,
    )
    os.close(slave_fd)

    output_tail: Deque[str] = deque()
    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            sys.stderr.write(text)
            sys.stderr.flush()
            _append_tail(output_tail, text, tail_limit)
    finally:
        os.close(master_fd)

    returncode = process.wait()
    tail_text = "".join(output_tail)
    return {
        "returncode": returncode,
        "stdout_tail": tail_text,
        "stderr_tail": tail_text,
    }
