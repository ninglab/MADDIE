import json
import os
import sys

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import DEFAULT_MAX_NEW_TOKENS
from gellmo_core import (
    build_prompt,
    decode_generated_sequences,
    extract_smiles_and_adjust,
    extract_tagged_smiles,
    normalize_base_model,
)
from prompter import Prompter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    request: str = "",
    load_in_8bit: bool = False,
    use_lora: bool = True,
    base_model: str = "Llama-3.1-8B-Instruct",
    lora_weights: str = "NingLab/GeLLMO-C-P10-Llama",
    opt_type: str = "thresh",
    instr_idx: int = 0,
    output_dir: str = "",
    num_return_sequences: int = 5,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 1.0,
):
    # Same prompter setup pattern as inference.py.
    response_split = "%%% Response:"
    prompter = Prompter(opt_type, response_split)

    smiles, adjust = extract_smiles_and_adjust(request)
    prompt = build_prompt(prompter, smiles, adjust, instr_idx)

    base_model = normalize_base_model(base_model)
    if "Llama-3.1-70B" in base_model:
        print("Using 70b model, loading in 8bit")
        load_in_8bit = True

    # Model/tokenizer loading follows inference.py.
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_lora:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if not load_in_8bit:
        model.bfloat16()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if not model.config.eos_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model.config.eos_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = "left"

    # Divergence from inference.py:
    # Use direct model.generate() and decode only continuation tokens.
    # This avoids delimiter-based split parsing issues in get_response().
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Keep generation args aligned with decoding mode.
    # This mirrors inference.py behavior: deterministic by default (do_sample=False).
    generate_kwargs = dict(
        **inputs,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        num_beams=1 if do_sample else num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature
    else:
        # Some checkpoints carry sampling defaults in generation_config
        # (e.g., temperature/top_p). Force deterministic defaults here to
        # avoid warnings when do_sample=False.
        generate_kwargs["temperature"] = 1.0
        generate_kwargs["top_p"] = 1.0

    generated = model.generate(**generate_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    raw_responses, responses = decode_generated_sequences(
        generated=generated,
        prompt_len=prompt_len,
        tokenizer=tokenizer,
        num_return_sequences=num_return_sequences,
    )

    results = [{"prompt": prompt, "response": responses, "raw_response": raw_responses}]
    output = {
        "request": request,
        "parsed": {"smiles": smiles, "adjust": adjust},
        "results": results,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "response.json"), "w") as f:
            json.dump(output, f, indent=2)
        with open(os.path.join(output_dir, "smiles.txt"), "w") as f:
            # Write cleaned SMILES-like lines.
            for r in responses:
                f.write(f"{extract_tagged_smiles(r)}\n")

    print("Prompt:")
    print(prompt)
    print("Model responses:")
    for i, r in enumerate(responses):
        print(f"[{i}] {r}")

    return output


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)
