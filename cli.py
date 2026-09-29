import argparse
import csv
import json
import os
from pathlib import Path

from orchestrator.agent import Orchestrator, WorkerEntry
from orchestrator.worker import (
    SynthesizerWorker,
    EvaluatorWorker,
    LeadOptimizerWorker,
    HitIdentifierWorker,
)

BASE = os.path.dirname(os.path.abspath(__file__))

WORKER_CLI_SCRIPTS = {
    "Synthesizer": f"{BASE}/cli_synthesizer.py",
    "HitIdentifier": f"{BASE}/cli_hit_identifier.py",
    "LeadOptimizer": f"{BASE}/cli_lead_optimizer.py",
    "Evaluator": f"{BASE}/cli_evaluator.py",
}

WORKER_READMES = {
    "Synthesizer": f"{BASE}/synthesizer/README.md",
    "HitIdentifier": f"{BASE}/hit_identifier/README.md",
    "LeadOptimizer": f"{BASE}/lead_optimizer/README.md",
    "Evaluator": f"{BASE}/evaluator/README.md",
}

# subdirectory under --log where each worker stores its per-call stdout/stderr
WORKER_LOG_NAMES = {
    "Synthesizer": "synthesizer",
    "HitIdentifier": "hit_identifier",
    "LeadOptimizer": "lead_optimizer",
    "Evaluator": "evaluator",
}

WORKER_ENV_DIRS = {
    "Synthesizer": "synthesizer",
    "HitIdentifier": "hit_identifier",
    "LeadOptimizer": "lead_optimizer",
    "Evaluator": "evaluator",
}


def write_interactions_csv(states_path: str, out_path: str | None = None) -> str | None:
    """Read a states.json, extract its 'interactions', and write them to a flat
    interactions.csv (next to states.json unless out_path is given).

    Columns are taken from the interaction keys. Returns the path written, or
    None if there are no interactions.
    """
    with open(states_path) as f:
        state = json.load(f)

    interactions = state.get("interactions", [])
    if not interactions:
        return None

    if out_path is None:
        out_path = os.path.join(os.path.dirname(states_path), "interactions.csv")

    # all interaction keys except "response" (too large/verbose for the csv)
    columns = [k for k in interactions[0].keys() if k != "response"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for interaction in interactions:
            row = dict(interaction)
            # payload is a list (or None); serialize so it fits one CSV cell
            if isinstance(row.get("payload"), (list, dict)):
                row["payload"] = json.dumps(row["payload"], default=str)
            writer.writerow(row)

    return out_path

if __name__ == "__main__":
    # region argument
    parser = argparse.ArgumentParser(description="Script to run MADDIE")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "run"],
        default="run",
        help="Specify what mode the Orchestrator should run.",
    )
    parser.add_argument(
        "--input_prompt",
        type=str,
        required=True,
        help="Input Prompt to MADDIE.",
    )
    # region directory-related
    parser.add_argument(
        "--env_dir",
        type=str,
        default="./env",
        help="Path to directory where data and files are uploaded",
    )
    parser.add_argument(
        "--eps_mem_dir",
        type=str,
        default="./ltm/episodic",
        help="Path to episodic memory directory.",
    )
    parser.add_argument(
        "--sem_mem_dir",
        type=str,
        default="./ltm/semantic",
        help="Path to semantic memory directory.",
    )
    parser.add_argument(
        "--log",
        type=str,
    )
    # endregion
    # region llm-related
    parser.add_argument(
        "--llm",
        type=str,
        choices=[
            "claude-sonnet-4-6",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "gpt-5.4-mini",
        ],
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="LLM Provider for MADDIE.",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=None,
        help="URL to vLLM. Must be specified if 'llm' uses neither Claude nor OpenAI.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable extended thinking for the Orchestrator's LLM (requires temperature=1).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        choices=["first"],
        default=None,
        help="Prompt caching strategy. 'first' caches system+first message.",
    )
    # endregion
    # region run-related
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10,
        help="Number of max iterations orchestrator can run its action.",
    )
    parser.add_argument(
        "--optimizer_num_outputs",
        type=int,
        default=5,
        help="Number of optimized variants LeadOptimizer generates per input molecule (default: 5).",
    )
    parser.add_argument(
        "--working_memory_threshold",
        type=int,
        default=None,
        help="Number of maximum tokens in Orchestrator Working Memory. By default, if tokens exceed this threshold, it will automatically trigger compaction."
    )
    # endregion

    args = parser.parse_args()
    # endregion

    # region store arguments
    # store the arguments used for this run early, so a crash mid-run still
    # leaves a record of exactly what was invoked.
    if args.log:
        os.makedirs(args.log, exist_ok=True)
        args_path = os.path.join(args.log, "args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    # endregion

    # region workers
    sem_mem_dir = Path(args.sem_mem_dir)

    # per-worker log dir (stdout/stderr per call) lives under --log when provided
    def _worker_log_dir(name: str) -> str | None:
        return os.path.join(args.log, "worker", WORKER_LOG_NAMES[name]) if args.log else None
    
    def _worker_env_dir(name: str) -> str:
        return os.path.join(args.env_dir, WORKER_ENV_DIRS[name])

    synthesizer_worker_entry = WorkerEntry(
        worker=SynthesizerWorker(model=args.llm, log_dir=_worker_log_dir("Synthesizer"), context_dir=_worker_env_dir("Synthesizer")),
        profile=sem_mem_dir / "workers/synthesizer/PROFILE.md",
    )

    hit_identifier_worker_entry = WorkerEntry(
        worker=HitIdentifierWorker(model=args.llm, log_dir=_worker_log_dir("HitIdentifier"), context_dir=_worker_env_dir("HitIdentifier")),
        profile=sem_mem_dir / "workers/hit_identifier/PROFILE.md",
    )

    evaluator_worker_entry = WorkerEntry(
        worker=EvaluatorWorker(model=args.llm, log_dir=_worker_log_dir("Evaluator"), context_dir=_worker_env_dir("Evaluator")),
        profile=sem_mem_dir / "workers/evaluator/PROFILE.md",
    )

    lead_optimizer_worker_entry = WorkerEntry(
        worker=LeadOptimizerWorker(model=args.llm, log_dir=_worker_log_dir("LeadOptimizer"), context_dir=_worker_env_dir("LeadOptimizer"), num_outputs=args.optimizer_num_outputs),
        profile=sem_mem_dir / "workers/lead_optimizer/PROFILE.md",
    )

    workers = {
        "Synthesizer": synthesizer_worker_entry,
        "HitIdentifier": hit_identifier_worker_entry,
        "LeadOptimizer": lead_optimizer_worker_entry,
        "Evaluator": evaluator_worker_entry,
    }
    # endregion

    # region setup llm arguments
    llm_args = {}
    if args.thinking:
        llm_args["thinking"] = True
        llm_args["temperature"] = 1
    if args.cache:
        llm_args["cache"] = args.cache

    # CHECKPOINT: check if args.llm match with the models in url.
    if args.llm not in ("claude-sonnet-4-6", "gpt-5.4-mini"):
        assert args.llm_url is not None
        llm_args["base_url"] = args.llm_url

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
        _client = OpenAI(api_key="none", base_url=args.llm_url)
        available = [m.id for m in _client.models.list().data]
        if args.llm not in available:
            raise ValueError(
                f"Model '{args.llm}' not found at {args.llm_url}. "
                f"Available models: {available}"
            )

    # CHECKPOINT: specific only for Qwen/Qwen3-30B-A3B-Instruct-2507
    if "Qwen3-30B-A3B-Instruct-2507" in args.llm:
        llm_args["temperature"] = 0.7
        llm_args["top_p"] = 0.8
        llm_args["extra_body"] = {"top_k": 20, "min_p": 0.0}
    # endregion

    ## setup orchestrator
    try:
        wm_args = {}
        if args.working_memory_threshold:
            wm_args["threshold"] = args.working_memory_threshold
    
        orchestrator = Orchestrator(
            llm=args.llm,
            env_dir=args.env_dir,
            eps_mem_dir=args.eps_mem_dir,
            sem_mem_dir=args.sem_mem_dir,
            workers=workers,
            llm_args=llm_args,
            wm_args=wm_args if wm_args else "default"
        )

        # region run
        if args.mode == "pretrain":
            raise NotImplementedError
            # worker_name = args.input_prompt
            # response = orchestrator.pretrain_worker(
            #     worker_name=worker_name,
            #     cli_script=WORKER_CLI_SCRIPTS[worker_name],
            #     readme=WORKER_READMES[worker_name],
            #     max_iterations=args.max_iter,
            # )
        elif args.mode == "run":
            _action_log = os.path.join(args.log, "action_log.jsonl") if args.log else None
            response = orchestrator.run(
                prompt=args.input_prompt, max_iterations=args.max_iter,
                action_log_path=_action_log,
            )
        else:
            raise ValueError
        # endregion

        print("=====RESPONSE=====")
        print(response)
    finally:
        ## store data
        state = orchestrator.get_state()
        if args.log:
            # store orchestrator states
            state_path = os.path.join(args.log, "states.json")
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)

            # also write a flat interactions.csv extracted from states.json
            write_interactions_csv(state_path)

            # store environment, episodic and semantic memory
            import shutil
            shutil.make_archive(os.path.join(args.log, "environment"), "zip", args.env_dir)
            shutil.make_archive(os.path.join(args.log, "ltm_eps"), "zip", args.eps_mem_dir)
            shutil.make_archive(os.path.join(args.log, "ltm_sem"), "zip", args.sem_mem_dir)
