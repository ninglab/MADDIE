from tool import HitIdentifierAgent

tool = HitIdentifierAgent(
    model="meta-llama/Llama-3.1-8B-Instruct",
)

print(tool.chat("What targets are configured right now?"))
print(tool.chat("What input files do you need before generating molecules?"))

result = tool.generate(
    "Find at least 10 potential hits for EGFR.pdb with Vina lower than -12 "
    "within 3 rounds. Use any available strategy, including SBDD, LBDD, or "
    "virtual screening."
)

print(result)
print(tool.last_result)