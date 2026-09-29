from hit_identifier.tool import HitIdentifierAgent

if __name__ == "__main__":
    import argparse
    import json
    import os

    # setup arguments
    parser = argparse.ArgumentParser(description="Run the MADDIE hit identifier agent.")
    parser.add_argument("--input_prompt", help="Prompt to send to the agent.")
    parser.add_argument("--context_dir", nargs="?", help="Working environment directory (default: ./env/hit_identifier).")
    parser.add_argument("--output", default=None, help="Path to write the response to (optional).")
    parser.add_argument("--path_to_metadata", default=None, help="Path to write the run metadata as JSON (optional).")
    parser.add_argument("--model", help="LLM used by the agent.")
    args = parser.parse_args()

    # run agent
    agent = HitIdentifierAgent(
        model=args.model,
        log_dir=os.path.join(args.context_dir, "log")
    )

    # response = agent.generate(
    #     args.input_prompt
    # )
    response, metadata = agent.run(
        args.input_prompt
    )


    if args.output:
        with open(args.output, "w") as f:
            f.write(response)
    else:
        print(response)

    if args.path_to_metadata:
        with open(args.path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=2, default=str)