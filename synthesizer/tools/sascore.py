from __future__ import annotations

from pathlib import Path
from typing import Callable

from rdkit import Chem

from .base import Tool

docs = """`SAScore(molecules="targets.smi", output="targets.sascore.tsv")`
SAScore computes the synthetic accessibility (SA) score for each molecule in a SMILES file.
SA Scores range from 1-10, with 1 being the easiest to synthesize and 10 being the most difficult to synthesize.
This tool should only be used for internal prioritization of products, not to return results to the user.
For example, if the user requests the synthesis of many molecules within a strict time limit,
SAScore may be used to prioritize the products to plan first.

Specify molecules="filename.smi" with one molecule per line containing a single SMILES string.
Use output to choose the result file. By default, the tool writes a tab-separated file next to the input with columns:
smiles    sa_score
Lower SA scores are easier to synthesize. Invalid SMILES are kept in the output with an empty sa_score value.
"""


def _load_calculate_score() -> Callable[[Chem.Mol], float]:
    try:
        from rdkit.Contrib.SA_Score import sascorer

        return sascorer.calculateScore
    except ImportError:
        from sascorer import calculateScore

        return calculateScore


class SAScore(Tool):
    def documentation(self) -> str:
        return docs

    def __call__(self, **params):
        molecules = params.get("molecules")
        if molecules is None or not str(molecules).strip():
            return None, '⚠️ Failed; please specify molecules="filename.smi".'

        input_path = Path(str(molecules)).expanduser().resolve()
        if not input_path.exists():
            return None, f"⚠️ Failed with FileNotFoundError: molecules file not found: {input_path}"
        if not input_path.is_file():
            return None, f"⚠️ Failed with ValueError: molecules input must be a file: {input_path}"

        output_arg = params.get("output")
        output_path = (
            Path(str(output_arg)).expanduser().resolve()
            if output_arg is not None and str(output_arg).strip()
            else input_path.with_suffix(input_path.suffix + ".sascore.tsv")
        )

        try:
            calculate_score = _load_calculate_score()
        except ImportError as exc:
            return None, f"⚠️ Failed with ImportError: could not import SA score implementation: {exc}"

        total = 0
        valid = 0
        with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
            dst.write("smiles\tsa_score\n")
            for raw_line in src:
                line = raw_line.strip()
                if line.lower() in ["mol", "smiles", "molecule", "smi"]:
                    continue
                if not line:
                    continue

                total += 1
                smiles = line

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    dst.write(f"{smiles}\t\n")
                    continue

                valid += 1
                score = calculate_score(mol)
                dst.write(f"{smiles}\t{score:.4f}\n")

        invalid = total - valid
        invalid_stmt = ""
        if invalid:
            invalid_stmt = f" Additionally, for {invalid} invalid SMILES, no SA score could be computed."
        return None, f"✅ SAScore evaluated for {valid} valid molecule(s) to {output_path}." + invalid_stmt


__all__ = ["SAScore"]
