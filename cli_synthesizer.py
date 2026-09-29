from synthesizer.tools.MEEA import MEEA
from synthesizer.tools.code import Code
from synthesizer.agent.agent import Agent
from synthesizer.llm.openai_compatible import OpenAILLM
import os

# Directory holding the MEEA/PrexSyn container images and their model assets.
ASSETS = os.environ.get("MADDIE_SYNTHESIZER_ASSETS", "/path/to/synthesizer-assets")

os.environ.setdefault("MEEA_IMAGE", os.path.join(ASSETS, "meea.sif"))
os.environ.setdefault("PREXSYN_IMAGE", os.path.join(ASSETS, "prexsyn.sif"))

largefiles = {
    name: os.path.join(ASSETS, name)
    for name in (
        "eMoleculesBB.txt",
        "EnamineREAL.txt",
        "policy_model.ckpt",
        "template_rules.dat",
        "value_pc.pt",
    )
}

if __name__ == "__main__":
    import argparse
    import json
    import sys, io

    # setup arguments
    parser = argparse.ArgumentParser(description="Run the MADDIE synthesizer agent.")
    parser.add_argument("--input_prompt", help="Prompt to send to the agent.")
    parser.add_argument("--context_dir", nargs="?", help="Working environment directory.")
    parser.add_argument("--output", default=None, help="Path to write the response to (optional).")
    parser.add_argument("--path_to_metadata", default=None, help="Path to write the run metadata as JSON (optional).")
    parser.add_argument("--model", help="LLM used by the agent.")
    args = parser.parse_args()

    # setup tools
    meea_tool = MEEA(largefiles=largefiles)
    meea_tool.setup_tool()

    code_tool = Code()
    code_tool.setup_tool()

    # setup llm
    os.environ["LLM_MODELNAME"] = args.model

    cli_llm = OpenAILLM()
    cli_agent = Agent([meea_tool, code_tool], llm=cli_llm)

    # run agent
    response, metadata = cli_agent(args.input_prompt, args.context_dir)

    if args.output:
        with open(args.output, "w") as f:
            f.write(response)
    else:
        print(response)

    if args.path_to_metadata:
        with open(args.path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
