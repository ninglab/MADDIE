from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .gellmo_core import (
    build_prompt,
    decode_generated_sequences,
    extract_smiles_and_adjust,
    extract_tagged_smiles,
    normalize_base_model,
)


@contextmanager
def gellmo_repo_context(gellmo_repo: Path | None):
    """Temporarily expose GeLLMO-C repo imports/templates when configured."""
    if gellmo_repo is None:
        yield
        return

    if str(gellmo_repo) not in sys.path:
        sys.path.insert(0, str(gellmo_repo))

    old_cwd = Path.cwd()
    os.chdir(gellmo_repo)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class GeLLMOTool:
    """
    GeLLMO-C optimization interface for single-request generation.
    """

    def __init__(
        self,
        base_model: str = "Llama-3.1-8B-Instruct",
        lora_weights: str = "NingLab/GeLLMO-C-P10-Llama",
        gellmo_repo: str | os.PathLike | None = None,
        use_lora: bool = True,
        load_in_8bit: bool = False,
        opt_type: str = "thresh",
    ):
        self.gellmo_repo = Path(gellmo_repo).resolve() if gellmo_repo else None
        self.base_model = normalize_base_model(base_model)
        self.lora_weights = lora_weights
        self.use_lora = use_lora
        self.load_in_8bit = load_in_8bit

        with gellmo_repo_context(self.gellmo_repo):
            try:
                from config import DEFAULT_MAX_NEW_TOKENS
                from prompter import Prompter
            except ImportError as exc:
                raise ImportError(
                    "Could not import GeLLMO-C config/prompter. Pass "
                    "gellmo_repo=/path/to/GeLLMO-C or add that repo to PYTHONPATH."
                ) from exc

            self.default_max_new_tokens = DEFAULT_MAX_NEW_TOKENS
            self.prompter = Prompter(opt_type, "%%% Response:")

        if "Llama-3.1-70B" in self.base_model:
            self.load_in_8bit = True

        self.tokenizer = None
        self.model = None
        self.last_result = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_8bit=self.load_in_8bit)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.use_lora:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_weights,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        if not self.load_in_8bit:
            self.model.bfloat16()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        if not self.model.config.eos_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer.pad_token_id = self.model.config.eos_token_id
            self.tokenizer.padding_side = "left"

    def optimize(
        self,
        request: str,
        instr_idx: int = 0,
        num_return_sequences: int = 5,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        output_dir: str = "",
        return_full: bool = False,
    ):
        self.load()
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        smiles, adjust = extract_smiles_and_adjust(request)
        prompt = build_prompt(self.prompter, smiles, adjust, instr_idx)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generate_kwargs = dict(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            num_beams=1 if do_sample else num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
        else:
            # Some checkpoints carry sampling defaults in generation_config
            # (e.g., temperature/top_p). Force deterministic defaults here to
            # avoid warnings when do_sample=False.
            generate_kwargs["temperature"] = 1.0
            generate_kwargs["top_p"] = 1.0

        generated = self.model.generate(**generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        raw_responses, responses = decode_generated_sequences(
            generated=generated,
            prompt_len=prompt_len,
            tokenizer=self.tokenizer,
            num_return_sequences=num_return_sequences,
        )

        result = {
            "request": request,
            "parsed": {"smiles": smiles, "adjust": adjust},
            "results": [{"prompt": prompt, "response": responses, "raw_response": raw_responses}],
        }
        self.last_result = result

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "response.json"), "w") as f:
                json.dump(result, f, indent=2)
            with open(os.path.join(output_dir, "smiles.txt"), "w") as f:
                for r in responses:
                    f.write(f"{extract_tagged_smiles(r)}\n")

        if return_full:
            return result
        if num_return_sequences == 1:
            return responses[0]
        return responses
