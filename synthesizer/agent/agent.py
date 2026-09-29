from typing import Any, Union, Dict, List, Tuple
from ..llm.base import LLM
from ..llm.openai_compatible import OpenAILLM
from ..tools.base import Tool
from ..tools.finish import Finish
from ..utils.linenum import get_code_location
from ..utils.timestamp import timestamp
import re, json, sys

system_intro = """You are an expert synthetic chemist who must complete the task given by the user using your computational tools.
You will be given a file containing molecules that require synthesis routes.

Your first and only action should be to call MEEA on the provided file. MEEA accepts both .smi and .csv files directly — do NOT preprocess, clean, or inspect the input file before calling MEEA.
MEEA internally handles PrexSyn as a fallback for non-synthesizable molecules. Do NOT call any other tool before or after MEEA unless MEEA itself fails to produce any output file.
If MEEA succeeds, finish immediately and report the output file path.
If MEEA produces no output file, you may use Code once to inspect the error message, then call Finish to report the failure — do not retry MEEA and do not use Code before MEEA has run.

Use your best judgment and follow the user's prompt.

Specify all of your responses using the following format:
Thought: (your reflection on the state)
Action: `ActionName(keyword=value, keyword=value, keyword=value)`
<PAUSE>

Your Action should be a backtick-wrapped function call using the syntax of one of the tools below:

"""

system_outro = "\nYou have {turns_remaining} actions remaining.  Make sure you can answer the user's request.\n{extra}"

class Agent(object):
    def __init__(self, tools: List[Tool], llm: LLM | None = None):
        self.llm = llm if llm is not None else OpenAILLM()
        finagent = Finish(agent=self)
        self.tools = tools + [finagent]
        for tool in self.tools:
            tool.setup_tool()
        self.tooldocs = "\n\n".join([t.documentation() for t in self.tools])
        self.finished = False

    def system_text(self, **kwargs):
        return system_intro + self.tooldocs + system_outro.format(**kwargs)
    
    def ahs(self):
        text = "Action History:"
        for _, hist in self.action_history:
            text += f"\n - {hist}"
        return text
    
    def format_molecule_set(self, mols : List[str]) -> str:
        return json.dumps(mols)

    def extract_action_call(self, resp: str) -> str | None:
        action_match = re.search(r"Action:\s*`", resp)
        if action_match is None:
            return None

        start = action_match.end()
        depth = 0
        in_string: str | None = None
        escaped = False

        for index in range(start, len(resp)):
            char = resp[index]

            if in_string is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == in_string:
                    in_string = None
                continue

            if char in {"'", '"'}:
                in_string = char
                continue
            if char == "(":
                depth += 1
                continue
            if char == ")":
                depth = max(0, depth - 1)
                continue
            if char == "`" and depth == 0:
                return resp[start:index]

        return None

    def __call__(self, prompt: str, environment_path: str, turn_limit=10) -> Tuple[str, dict]:
        turns = 0
        self.finished = False
        extra = ""
        self.action_history = []
        self.metadata = {"responses": []}
        while not self.finished and turns < turn_limit:
            systext = self.system_text(
                turns_remaining=turn_limit - turns,
                extra=extra
            )
            usertext = prompt + f"\nRelevant files are located in environment directory: {environment_path}.  Make sure to store all of your results in that same directory.\n" + self.ahs()
            resp, meta = self.llm.get_single_response(usertext, system=systext)
            self.metadata["responses"].append(meta)
            sys.stderr.write("\n".join([timestamp(), get_code_location(), "[DEBUG]", f"Starting turn {turns+1}"]) + "\n")
            sys.stderr.write("\n".join([timestamp(), get_code_location(), "[DEBUG]", systext]) + "\n")
            sys.stdout.write("\n".join([timestamp(), get_code_location(), "[INFO]", usertext]) + "\n")
            sys.stdout.write("\n".join([timestamp(), get_code_location(), "[INFO]", resp]) + "\n")
            if "Action: `" not in resp:
                extra = "Please ensure your response contains a function call wrapped in backticks following the format:\nAction: `...`"
            else:
                action_call = self.extract_action_call(resp)
                sys.stderr.write("\n".join([timestamp(), get_code_location(), "[DEBUG]", "Action: ", str(action_call)]) + "\n")
                if action_call is None:
                    extra = "Please ensure your Action line contains exactly one backtick-wrapped function call."
                else:
                    try:
                        for tool in self.tools:
                            tool.agent_output_dir = environment_path
                        tool_context = {tool.__class__.__name__: tool for tool in self.tools}
                        action_result, status = eval(action_call, {"__builtins__": {}}, tool_context)
                        self.action_history.append((action_result, action_call + " → " + status))
                        action_result_text = repr(action_result)
                        extra = "Full result of last action: " + action_result_text[:1000]
                        if len(action_result_text) > 1000:
                            extra += "..."
                    except Exception as e:
                        sys.stderr.write("\n".join([timestamp(), get_code_location(), "[ERROR]", "Action Error " + str(action_call) + " " + str(e)]) + "\n")
                        extra = "Please make sure you follow the Action format exactly:\nAction: `ActionName(keyword=value)`"
                    finally:
                        turns += 1
        if not self.finished:
            sys.stderr.write("\n".join([timestamp(), get_code_location(), "[ERROR]", "The agent reached the iteration limit while attempting to complete the task."]) + "\n")
            self.response = "The agent reached its iteration limit attempting to complete the task."
            finalprompt = f"Please summarize what tasks were performed and where the outputs are, since you were unable to complete the task within the allotted {turn_limit} iterations."
            finalprompt += f"\nOriginal user prompt: {prompt}\nDirectory for inputs/outputs: {environment_path}\n"
            finalprompt += self.ahs()
            finalprompt += "\n\nGenerate your response as a plaintext report, just summarizing what happened and where to find any intermediate results."
            summary, meta = self.llm.get_single_response(finalprompt)
            self.metadata["responses"].append(meta)
            self.response += summary
        return self.response, self.metadata
