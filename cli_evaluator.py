from evaluator import EvaluatorAgent

if __name__ == "__main__":
    import argparse
    import json

    # setup arguments
    parser = argparse.ArgumentParser(description="Run the MADDIE evaluator agent.")
    parser.add_argument("--input_prompt", help="Prompt to send to the agent.")
    parser.add_argument("--context_dir", nargs="?", help="Working environment directory (default: ./env/evaluator).")
    parser.add_argument("--output", default=None, help="Path to write the response to (optional).")
    parser.add_argument("--path_to_metadata", default=None, help="Path to write the run metadata as JSON (optional).")
    parser.add_argument("--model", help="LLM used by the agent.")
    args = parser.parse_args()

    # run agent
    agent = EvaluatorAgent(model=args.model)
    response, metadata = agent.evaluate(
        prompt=args.input_prompt,
        path_to_env=args.context_dir,
        verbose=True
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(response)
    else:
        print(response)

    if args.path_to_metadata:
        with open(args.path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

