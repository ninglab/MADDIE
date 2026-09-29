import os
import sys
from pathlib import Path

# Resolve the checkout automatically regardless of where this is run from.
maddie_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(maddie_repo))

from lead_optimizer import (  # noqa: E402
    LeadOptimizerAgent,
    build_molopt_tools,
    build_oracle_registry,
)
from lead_optimizer.llms import OpenAICompatibleLLM
from lead_optimizer.planner import JSONPlanner
from lead_optimizer.tools.gellmo import GeLLMOTool
from lead_optimizer.tools import GeLLMOOptimizerTool

if __name__ == "__main__":
    import argparse
    import json

    # setup arguments
    parser = argparse.ArgumentParser(description="Run the MADDIE lead optimizer agent.")
    parser.add_argument("--input_prompt", help="Prompt to send to the agent.")
    parser.add_argument("--context_dir", nargs="?", help="Working environment directory (default: ./env/lead_optimizer).")
    parser.add_argument("--output", default=None, help="Path to write the response to (optional).")
    parser.add_argument("--path_to_metadata", default=None, help="Path to write the run metadata as JSON (optional).")
    parser.add_argument("--model", help="LLM used by the agent.")
    parser.add_argument("--num_outputs", type=int, default=5, help="Number of optimized variants to generate per input molecule (default: 5).")
    args = parser.parse_args()

    os.environ["LLM_MODELNAME"] = args.model

    # Set these via the environment (see .env.example) or edit the defaults.
    runtime = os.environ.get("MADDIE_OPTIMIZER_RUNTIME", "/path/to/lead_optimizer/runtime")
    gellmo_repo = os.environ.get("GELLMO_REPO", os.path.join(runtime, "GeLLMO-C"))
    molopt_repo = Path(os.environ.get("MOLOPT_REPO", os.path.join(runtime, "mol-opt")))
    benchmark_path = os.environ.get(
        "MOLOPT_BENCHMARK_PATH",
        os.path.join(runtime, "molopt_testing/agent_knowledge/molopt_benchmark_knowledge.json"),
    )

    base_model = os.environ.get("GELLMO_BASE_MODEL", "/path/to/models/llama31_8b_instruct")
    lora_weights = os.environ.get("GELLMO_LORA_WEIGHTS", "/path/to/models/gellmo_c_p10_llama_lora")

    # run agent
    planner = JSONPlanner(
        OpenAICompatibleLLM(max_new_tokens=1024, temperature=0),
        benchmark_path=benchmark_path,
    )

    try:
        oracle_registry = build_oracle_registry(
            molopt_repo=molopt_repo,
            evaluator_root=maddie_repo,
        )
        molopt_tools = build_molopt_tools(
            molopt_repo=molopt_repo,
            oracle="qed",
            oracle_registry=oracle_registry,
            skip_unavailable=True,
            output_root=Path(args.context_dir) / "output" / "molopt_work",
            freq_log=10,
            patience=5,
            n_jobs=1,
            verbose=False,
        )
    except ImportError as exc:
        print(f"MolOpt tools unavailable in this checkout: {exc}")
        print("Continuing with GeLLMO-C only.")
        molopt_tools = []

    gellmo = GeLLMOTool(
        base_model=base_model,
        lora_weights=lora_weights,
        #change to your gellmoc repo path 
        gellmo_repo=gellmo_repo,
    )

    agent = LeadOptimizerAgent(
        planner=planner,
        tools=molopt_tools + [GeLLMOOptimizerTool(gellmo)],
        num_outputs=args.num_outputs,
        use_llm_next_action=True,
        execute_next_actions=True,
        max_tool_actions=2,
    )

    response, metadata = agent.run(
        prompt=args.input_prompt,
        path_to_env=args.context_dir,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(response)
    else:
        print(response)

    if args.path_to_metadata:
        with open(args.path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=2, default=str)