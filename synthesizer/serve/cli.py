from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from synthesizer.serve.__main__ import DEFAULT_ENV_DIR, _build_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synthesizer.serve.cli",
        description="Run the Synthesizer agent from the command line.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt to send to the Synthesizer agent. If omitted, reads from stdin or prompts interactively.",
    )
    parser.add_argument(
        "--environment-path",
        default=str(DEFAULT_ENV_DIR),
        help="Writable directory containing prompt-referenced files and where outputs should be stored.",
    )
    parser.add_argument(
        "--turn-limit", '-t',
        type=int,
        default=10,
        help="Maximum number of agent turns.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON object containing the response and metadata.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print metadata after the response.",
    )
    return parser


def resolve_prompt(raw_prompt: str | None) -> str:
    if raw_prompt is not None:
        return raw_prompt
    if not sys.stdin.isatty():
        stdin_prompt = sys.stdin.read().strip()
        if stdin_prompt:
            return stdin_prompt
    return input("Prompt: ").strip()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    prompt = resolve_prompt(args.prompt)
    if not prompt:
        parser.error("a non-empty prompt is required")
    environment_path = Path(args.environment_path).expanduser().resolve()
    environment_path.mkdir(parents=True, exist_ok=True)
    turn_limit = max(1, args.turn_limit)

    agent = _build_agent()
    response, metadata = agent(prompt, str(environment_path), turn_limit=turn_limit)

    if args.json:
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "environment_path": str(environment_path),
                    "turn_limit": turn_limit,
                    "response": response,
                    "metadata": metadata,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(response)
    if args.show_metadata:
        print("\n--- metadata ---")
        print(json.dumps(metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
