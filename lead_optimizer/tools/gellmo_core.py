import re
from typing import Any, List, Tuple


def normalize_base_model(base_model: str) -> str:
    """
    Keep the same short-name to HF namespace mapping used in inference.py.
    """
    if base_model in {"Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"}:
        return f"meta-llama/{base_model}"
    if base_model == "Mistral-7B-Instruct-v0.3":
        return f"mistralai/{base_model}"
    return base_model


def extract_smiles_and_adjust(request: str) -> Tuple[str, str]:
    """
    Extract one <SMILES>...</SMILES> block and treat the remaining text as
    the optimization instruction. (assumes user knows to use adjustment prompt similar to train/test data)
    """
    request = request.strip()
    if not request:
        raise ValueError("Empty request. Include <SMILES> ... </SMILES>.")

    match = re.search(r"<SMILES>\s*(.*?)\s*</SMILES>", request, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Could not find SMILES. Use <SMILES> ... </SMILES> in --request.")

    smiles = match.group(1).strip()
    adjust = (request[: match.start()] + " " + request[match.end() :]).strip()
    adjust = re.sub(r"\s+", " ", adjust)
    if not adjust:
        adjust = "Please optimize this molecule based on my request."
    return smiles, adjust


def build_prompt(prompter: Any, smiles: str, adjust: str, instr_idx: int) -> str:
    """
    Build the GeLLMO-C instruction-format prompt for a single request, using the same pattern as inference.py.
    """
    return (
        f"{prompter.instructions[instr_idx]}\n"
        f"%%% Input : <SMILES> {smiles} </SMILES>\n"
        f"%%% Adjust: {adjust}\n"
        f"%%% Response:\n"
    )


def extract_tagged_smiles(text: str) -> str:
    tagged = re.findall(r"<SMILES>\s*(.*?)\s*</SMILES>", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged[-1].strip()
    return text.strip()


def clean_generated_text(text: str) -> str:
    """
    Normalize common generation artifacts:
    - tagged <SMILES>...</SMILES> when present
    - Strip leading "%%%"
    - Strip leading "The final answer is:"
    - Strip trailing oprphan </SMILES>
    """
    cleaned = text.strip()
    tagged = re.findall(r"<SMILES>\s*(.*?)\s*</SMILES>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged[-1].strip()

    cleaned = re.sub(r"^\s*%%%\s*", "", cleaned)
    cleaned = re.sub(r"^\s*the\s+final\s+answer\s+is\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*</SMILES>\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def decode_generated_sequences(
    generated: Any,
    prompt_len: int,
    tokenizer: Any,
    num_return_sequences: int,
) -> Tuple[List[str], List[str]]:
    """
    Decode model continuations and return (raw_responses, cleaned_responses).
    """
    raw_responses: List[str] = []
    responses: List[str] = []
    for i in range(num_return_sequences):
        continuation = generated[i][prompt_len:]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        raw_responses.append(text)
        responses.append(clean_generated_text(text))
    return raw_responses, responses

