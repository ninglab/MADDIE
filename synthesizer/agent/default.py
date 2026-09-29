from .agent import Agent
from synthesizer.tools.MEEA import MEEA
from synthesizer.tools.code import Code
from synthesizer.llm.openai_compatible import OpenAILLM
import os

class DefaultAgent(Agent):
    def __init__(self, tools = [], llm = None):
        os.environ["MEEA_IMAGE"] = os.environ.get("MEEA_IMAGE", "/path/to/synthesizer-assets/meea.sif")
        os.environ["PREXSYN_IMAGE"] = os.environ.get("PREXSYN_IMAGE", "/path/to/synthesizer-assets/prexsyn.sif")

        largefiles = {
            "eMoleculesBB.txt": "/path/to/synthesizer-assets/eMoleculesBB.txt",
            "EnamineREAL.txt": "/path/to/synthesizer-assets/EnamineREAL.txt",
            "policy_model.ckpt": "/path/to/synthesizer-assets/policy_model.ckpt",
            "template_rules.dat": "/path/to/synthesizer-assets/template_rules.dat",
            "value_pc.pt": "/path/to/synthesizer-assets/value_pc.pt",
        }

        if not tools:
            meea_tool = MEEA(largefiles=largefiles)
            meea_tool.setup_tool()
            code_tool = Code()
            code_tool.setup_tool()
            tools = [meea_tool, code_tool]

        if llm is None:
            llm = OpenAILLM({"max_tokens": 8192})
        super().__init__(tools, llm)

    