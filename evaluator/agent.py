"""
EvaluatorAgent – LLM-driven molecular evaluator using OpenAI tool-use.

The agent is given a natural-language prompt and a path to an environment
directory; it autonomously lists files, reads molecules, runs tools, saves
results, and returns a structured report.

Usage:
    agent = EvaluatorAgent()
    response, metadata = agent.evaluate(
        prompt="Evaluate the compounds in hits.csv (SMILES column: 'smiles')",
        path_to_env="/path/to/env",
    )
    print(response)
    print(metadata["tool_results"])
"""

import json
import logging
import os
import pandas as pd
from typing import Any, Optional

def _make_logger(name: str = "evaluator") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
            datefmt="%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

_logger = _make_logger()

from .tools.physchem import calculate_all_properties
from .tools.filters import get_filter_details
from .tools.autodockvina import run_autodockvina
from .tools.unidock import run_unidock
from .tools.admet import predict_toxicity, predict_toxicity_batch, predict_full_admet
from .tools.diversity import calculate_tanimoto_diversity
from .tools.ncircle import run_ncircle
from .tools.boltz import predict_structure_and_affinity


# ---------------------------------------------------------------------------
# Tool schema definitions (sent to Claude)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "calculate_physicochemical_properties",
        "description": (
            "Calculate key physicochemical properties of a molecule: "
            "QED (drug-likeness, 0–1 higher is better), SAScore (synthetic "
            "accessibility, 1–10 lower is easier), molecular weight, LogP, TPSA, "
            "HBD/HBA counts, rotatable bonds, and ring counts. "
            "Always call this first for any molecule evaluation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule."},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "apply_molecular_filters",
        "description": (
            "Apply medicinal chemistry filters to a molecule: PAINS, Brenk structural "
            "alerts, relaxed Lipinski Ro5, and Lilly MedChem Rules. "
            "Returns pass/fail for each filter plus a list of matched alerts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "predict_admet_properties",
        "description": (
            "Predict ADMET/toxicity properties using ADMET-AI. "
            "Key endpoints: hERG (cardiac ion channel toxicity), "
            "DILI (drug-induced liver injury), ClinTox (clinical toxicity), "
            "Mutagenicity (Ames test), Carcinogens. "
            "All scores are probabilities (0–1); interpret against thresholds specified in the user prompt, not fixed defaults. "
            "Set full_profile=true to get all ~100 ADMET properties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "full_profile": {
                    "type": "boolean",
                    "description": "Return full profile (slow) vs. toxicity-only (fast).",
                    "default": False,
                },
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "run_molecular_docking",
        "description": (
            "Dock a ligand into a protein target using AutoDock Vina. "
            "Returns binding affinity in kcal/mol (more negative = stronger binding). "
            "Requires pdb_path to the target protein."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "pdb_path": {"type": "string", "description": "Absolute path to the protein PDB file."},
                "grid_size": {"type": "integer", "description": "Docking grid size in Angstroms (default: 20)."},
                "n_poses": {"type": "integer", "description": "Number of docking poses to generate (default: 5)."},
                "exhaustiveness": {"type": "integer", "description": "Exhaustiveness of the docking search (default: 8)."},
                "prepare_receptor": {"type": "string", "enum": ["meeko", "mgltools"], "description": "Receptor preparation method (default: meeko)."},
                "prepare_ligand": {"type": "string", "enum": ["meeko", "obabel"], "description": "Ligand preparation method (default: meeko)."},
                "cache_dir": {"type": "string", "description": "Optional cache directory."},
            },
            "required": ["smiles", "pdb_path"],
        },
    },
    {
        "name": "predict_structure_affinity_boltz",
        "description": (
            "Use Boltz-2 deep learning to co-predict the protein-ligand complex "
            "structure and binding affinity. More accurate than docking but GPU-intensive. "
            "Use when high accuracy is needed or when no experimental structure exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "protein_pdb_path": {"type": "string"},
                "protein_sequence": {"type": "string"},
                "cache_dir": {"type": "string"},
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "calculate_ncircles",
        "description": (
            "Calculate the NCircles diversity metric for a set of molecules. "
            "Counts mutually dissimilar molecules whose pairwise Tanimoto distance exceeds a threshold. "
            "Higher count = more diverse set. Use threshold=0.0 to count unique molecules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Tanimoto distance threshold for circle separation (default: 0.75).",
                    "default": 0.75,
                },
            },
            "required": ["smiles_list"],
        },
    },
    {
        "name": "calculate_tanimoto_diversity",
        "description": (
            "Calculate Tanimoto diversity for a set of molecules using Morgan fingerprints. "
            "Returns mean, min, and max pairwise Tanimoto distance (1 - similarity). "
            "Higher mean distance = more diverse set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings.",
                },
            },
            "required": ["smiles_list"],
        },
    },
    # --- file-based tools ---
    {
        "name": "list_environment_files",
        "description": (
            "List all files in the environment directory. "
            "Call this first to discover available input files before doing any evaluation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path_to_env": {"type": "string", "description": "Path to the environment directory."},
            },
            "required": ["path_to_env"],
        },
    },
    {
        "name": "evaluate_molecules_from_file",
        "description": (
            "Read molecules from a CSV file, run all requested evaluations "
            "(physicochemical properties, filters, ADMET, optionally docking), "
            "and save the results to an output CSV in the same directory. "
            "Use this to evaluate an entire file rather than molecule-by-molecule."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the input CSV file."},
                "smiles_column": {"type": "string", "description": "Name of the column containing SMILES strings."},
                "output_path": {
                    "type": "string",
                    "description": "Path for the output CSV. Defaults to <input>_evaluated.csv in the same directory.",
                },
                "pdb_path": {
                    "type": "string",
                    "description": "Absolute path to a protein PDB file for docking (optional).",
                },
                "cache_dir": {"type": "string", "description": "Optional cache directory for docking."},
                "run_admet": {
                    "type": "boolean",
                    "description": "Whether to run ADMET prediction (default true).",
                    "default": True,
                },
            },
            "required": ["file_path", "smiles_column"],
        },
    },
]


# Convert _TOOLS (Anthropic format) → OpenAI function-calling format
_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in _TOOLS
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _execute_tool(name: str, tool_input: dict) -> Any:
    if name == "calculate_physicochemical_properties":
        return calculate_all_properties(tool_input["smiles"])

    if name == "apply_molecular_filters":
        return get_filter_details(tool_input["smiles"])

    if name == "predict_admet_properties":
        smiles = tool_input["smiles"]
        if tool_input.get("full_profile", False):
            return predict_full_admet(smiles)
        return predict_toxicity(smiles)

    if name == "run_molecular_docking":
        smiles = tool_input["smiles"]
        pdb_path = tool_input["pdb_path"]
        prepare_receptor = tool_input.get("prepare_receptor", "meeko")
        prepare_ligand = tool_input.get("prepare_ligand", "meeko")
        cache_dir = tool_input.get("cache_dir")
        _logger.info("running docking (Uni-Dock)...")
        try:
            score = run_unidock(smiles=smiles, pdb_path=pdb_path, cache_dir=cache_dir)
            _logger.info(f"docking completed (backend=unidock, score={score} kcal/mol)")
            return {"binding_affinity_kcal_mol": score, "backend": "quickvina2-cuda"}
        except Exception as cuda_err:
            _logger.warning(f"QuickVina2-CUDA unavailable ({cuda_err}), falling back to AutoDock Vina CPU.")
        _logger.info("running docking (AutoDock Vina CPU)...")
        try:
            score = run_autodockvina(
                smiles=smiles,
                pdb_path=pdb_path,
                grid_size=tool_input.get("grid_size", 20),
                n_poses=tool_input.get("n_poses", 5),
                exhaustiveness=tool_input.get("exhaustiveness", 8),
                prepare_receptor=prepare_receptor,
                prepare_ligand=prepare_ligand,
                cache_dir=cache_dir,
            )
            _logger.info(f"docking completed (backend=autodock-vina-cpu, score={score} kcal/mol)")
            return {"binding_affinity_kcal_mol": score, "backend": "autodock-vina-cpu"}
        except Exception as e:
            _logger.error(f"docking failed (autodock-vina-cpu): {e}")
            return {"binding_affinity_kcal_mol": None, "error": str(e)}

    if name == "predict_structure_affinity_boltz":
        return predict_structure_and_affinity(
            smiles=tool_input["smiles"],
            protein_pdb_path=tool_input.get("protein_pdb_path"),
            protein_sequence=tool_input.get("protein_sequence"),
            cache_dir=tool_input.get("cache_dir"),
        )

    if name == "calculate_ncircles":
        n, circs = run_ncircle(tool_input["smiles_list"], d=tool_input.get("threshold", 0.75))
        return {"n_circles": n, "representative_smiles": circs}

    if name == "calculate_tanimoto_diversity":
        return calculate_tanimoto_diversity(tool_input["smiles_list"])

    if name == "list_environment_files":
        return _list_environment_files(tool_input["path_to_env"])

    if name == "evaluate_molecules_from_file":
        return _evaluate_molecules_from_file(
            file_path=tool_input["file_path"],
            smiles_column=tool_input["smiles_column"],
            output_path=tool_input.get("output_path"),
            pdb_path=tool_input.get("pdb_path"),
            cache_dir=tool_input.get("cache_dir"),
            run_admet=tool_input.get("run_admet", True),
        )

    return {"error": f"Unknown tool: {name}"}


def _list_environment_files(path_to_env: str) -> dict:
    try:
        files = os.listdir(path_to_env)
        return {"path": path_to_env, "files": sorted(files), "count": len(files)}
    except Exception as e:
        return {"error": str(e)}


def _evaluate_molecules_from_file(
    file_path: str,
    smiles_column: str,
    output_path: Optional[str] = None,
    pdb_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    run_admet: bool = True,
) -> dict:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    actual_smiles_column = next(
        (col for col in df.columns if col.lower() == smiles_column.lower()),
        None,
    )
    if actual_smiles_column is None:
        return {"error": f"Column '{smiles_column}' not found. Available columns: {list(df.columns)}"}
    smiles_column = actual_smiles_column

    if pdb_path is not None:
        if not os.path.isfile(pdb_path):
            return {"error": f"pdb_path not found: '{pdb_path}'. Provide an absolute path to an existing .pdb file."}
        if not pdb_path.lower().endswith((".pdb", ".pdb.gz", ".ent", ".ent.gz")):
            return {"error": f"pdb_path must be a .pdb file, got: '{pdb_path}'."}

    records = []
    errors = []
    all_smiles = list(df[smiles_column].dropna())
    n_total = len(all_smiles)

    # Run ADMET in one batch pass before the per-molecule loop
    admet_batch: List[dict] = []
    if run_admet:
        try:
            admet_batch = predict_toxicity_batch(all_smiles)
        except Exception as e:
            errors.append(f"batch admet: {e}")
            admet_batch = [{} for _ in all_smiles]

    for idx, smi in enumerate(all_smiles):
        row: dict = {"smiles": smi}

        try:
            physchem = calculate_all_properties(smi)
            row.update({f"physchem_{k}": v for k, v in physchem.items()})
        except Exception as e:
            errors.append(f"row {idx} physchem: {e}")

        try:
            filt = get_filter_details(smi)
            row.update({f"filter_{k}": v for k, v in filt.items()})
        except Exception as e:
            errors.append(f"row {idx} filters: {e}")

        if run_admet and idx < len(admet_batch):
            row.update({f"admet_{k}": v for k, v in admet_batch[idx].items()})

        if pdb_path:
            try:
                score = run_unidock(smiles=smi, pdb_path=pdb_path, cache_dir=cache_dir)
                _logger.info(f"[{idx+1}/{n_total}] row {idx}: unidock {score} kcal/mol")
            except Exception:
                try:
                    score = run_autodockvina(smiles=smi, pdb_path=pdb_path, cache_dir=cache_dir)
                    _logger.info(f"[{idx+1}/{n_total}] row {idx}: autodock-vina-cpu {score} kcal/mol")
                except Exception as e:
                    _logger.error(f"[{idx+1}/{n_total}] row {idx}: docking failed — {e}")
                    errors.append(f"row {idx} docking: {e}")
                    score = None
            row["docking_kcal_mol"] = score

        records.append(row)

    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_evaluated{ext or '.csv'}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    out_df = pd.DataFrame(records)
    try:
        out_df.to_csv(output_path, index=False)
    except Exception as e:
        return {"error": f"Could not write output: {e}", "n_evaluated": len(records)}

    # Build summary from computed records
    summary: dict = {}

    qed_vals = out_df["physchem_QED"].dropna() if "physchem_QED" in out_df.columns else pd.Series([], dtype=float)
    if not qed_vals.empty:
        summary["QED"] = {
            "min": round(qed_vals.min(), 3),
            "max": round(qed_vals.max(), 3),
            "mean": round(qed_vals.mean(), 3),
        }

    sa_vals = out_df["physchem_SAScore"].dropna() if "physchem_SAScore" in out_df.columns else pd.Series([], dtype=float)
    if not sa_vals.empty:
        summary["SAScore"] = {
            "min": round(sa_vals.min(), 3),
            "max": round(sa_vals.max(), 3),
        }

    # Filter counts are intentionally excluded from the summary dict.
    # They are stored in the CSV (filter_PAINS, filter_Brenk, filter_RDKIT,
    # filter_LILLY) and can be counted by the caller via PROCESS if needed.
    # Including them here risks the LLM reporting them unprompted and causing
    # the caller to misinterpret them as docking/ADMET pass counts.

    for admet_col in ["admet_hERG", "admet_DILI", "admet_ClinTox",
                       "admet_Mutagenicity", "admet_Carcinogens"]:
        if admet_col in out_df.columns:
            vals = out_df[admet_col].dropna()
            if not vals.empty:
                summary[admet_col] = {
                    "mean": round(float(vals.mean()), 3),
                    "min":  round(float(vals.min()),  3),
                    "max":  round(float(vals.max()),  3),
                }

    if "docking_kcal_mol" in out_df.columns:
        dock_vals = out_df["docking_kcal_mol"].dropna()
        if not dock_vals.empty:
            summary["docking_kcal_mol"] = {
                "min": round(dock_vals.min(), 2),
                "max": round(dock_vals.max(), 2),
            }

    try:
        if len(all_smiles) >= 2:
            summary["tanimoto_diversity"] = calculate_tanimoto_diversity(all_smiles)
            n_circles, _ = run_ncircle(all_smiles, d=0.75)
            summary["n_circles"] = n_circles
    except Exception as e:
        errors.append(f"diversity metrics: {e}")

    result = {"output_file": output_path, "n_evaluated": len(records), "summary": summary}
    if errors:
        result["warnings"] = errors
    return result


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert computational medicinal chemist evaluating drug candidates.

You operate on files located in an environment directory. Your workflow:
1. Call list_environment_files to discover what files are available.
2. Identify the relevant input file(s) based on the user's prompt.
3. Call evaluate_molecules_from_file to run all evaluations and save results.
   - Always pass the smiles_column name (infer from the prompt or file columns).
   - Pass pdb_path if the user provides a protein structure path.
   - Let output_path default unless the user specifies a name.
4. Report the absolute path of the output file, how many molecules were evaluated,
   and a concise numerical summary using the EXACT column names from the output CSV:
   - ADMET columns are prefixed with "admet_" (e.g. admet_hERG, admet_DILI, admet_ClinTox,
     admet_Mutagenicity, admet_Carcinogens)
   - Docking score column is "docking_kcal_mol"
   - Physicochemical columns are prefixed with "physchem_" (e.g. physchem_QED, physchem_SAScore)
   Use the exact column name in the summary so the caller can reference it directly, e.g.:
     "admet_hERG: 0.003–0.962", "docking_kcal_mol: -8.0 to -5.8"
   Report numbers only — do NOT add qualitative labels such as "high risk", "low risk",
   "strong binding", "moderate", or any other interpretation.
   Threshold judgements are the caller's responsibility, not yours.
   NEVER report structural filter results (filter_PAINS, filter_Brenk, filter_RDKIT,
   filter_LILLY) in your summary unless the user's prompt explicitly asks for them.
   These are stored in the CSV but must be silently omitted from your text output.
   If the user does ask, report each filter separately using its exact column name, e.g.:
     "filter_PAINS: 72/89", "filter_LILLY: 20/89"
   Never aggregate them into a single combined count — each filter is independent.
   If the tool result contains a "warnings" field, always mention the errors explicitly
   (e.g. invalid SMILES, docking failures) so the user knows which molecules were skipped.
   If there is no "warnings" field in the tool result, do not invent a warnings line.
   Be specific and straight to the point. The less words you put there the better.
   Every word must be meaningful and contains valuable information.
   Always remember to be concise.

Use the single-molecule tools (calculate_physicochemical_properties, apply_molecular_filters, etc.)
only when asked to inspect a specific individual compound.
Only use Boltz-2 if explicitly requested."""


class EvaluatorAgent:
    """
    LLM-based molecular evaluator that uses OpenAI tool-use to
    orchestrate computational chemistry tools on files in an environment.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        max_iterations: int = 10,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError("Run: pip install openai") from e
        self.client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def evaluate(
        self,
        prompt: str,
        path_to_env: str,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> tuple[str, dict]:
        """
        Evaluate molecules in an environment directory.

        Args:
            prompt:      Natural-language instruction (e.g. which file, SMILES column, task).
            path_to_env: Path to the directory containing input files.
            cache_dir:   Cache directory for expensive computations.
            verbose:     Print tool calls as they happen.

        Returns:
            (response, metadata)
            response  – agent's final expert assessment (str)
            metadata  – {"tool_results": {tool_name: result, ...}, "iterations": int}
        """
        context = (
            f"Environment directory: {path_to_env}\n"
            f"Task: {prompt}"
        )
        if cache_dir:
            context += f"\nCache directory: {cache_dir}"

        _logger.info(f"EvaluatorAgent started (model={self.model}, env={path_to_env})")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        tool_results: dict = {}
        iterations = 0
        responses: list = []

        for iteration in range(self.max_iterations):
            iterations = iteration + 1
            _logger.info(f"iteration {iterations}/{self.max_iterations}")
            if self.model in ("gpt-5.4-mini"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=self.max_tokens,
                    tools=_TOOLS_OPENAI,
                    messages=messages,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    tools=_TOOLS_OPENAI,
                    messages=messages,
                )                
            responses.append(response)

            choice = response.choices[0]

            if choice.message.content:
                _logger.debug(f"model response:\n{choice.message.content}")
            _logger.debug(f"finish_reason={choice.finish_reason}")

            if choice.finish_reason == "stop":
                summary = choice.message.content or ""
                metadata = {"tool_results": tool_results, "iterations": iterations, "responses": responses}
                _logger.debug(f"tool results:\n{json.dumps(tool_results, indent=2, default=str)}")
                return summary, metadata

            if choice.finish_reason != "tool_calls":
                break

            # Append assistant turn with tool calls
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            # Execute every tool call in this turn
            for tc in choice.message.tool_calls:
                tool_input = json.loads(tc.function.arguments)
                _logger.debug(f"tool call:\n{tc.function.name}({json.dumps(tool_input, indent=2)})")

                result = _execute_tool(tc.function.name, tool_input)
                tool_results[tc.function.name] = result

                _logger.debug(f"tool output:\n{json.dumps(result, indent=2, default=str)}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        _logger.debug(f"tool results:\n{json.dumps(tool_results, indent=2, default=str)}")
        metadata = {"tool_results": tool_results, "iterations": iterations, "responses": responses}

        if responses:
            fin = responses[-1].choices[0].finish_reason
            partial = responses[-1].choices[0].message.content or ""
            if fin == "length":
                fallback = (
                    f"[Evaluator stopped: response truncated (max_tokens) after {iterations} iterations. "
                    f"Partial response: {partial}]"
                )
            else:
                fallback = (
                    f"[Evaluator stopped after {iterations} iterations (finish_reason={fin}). "
                    f"Partial response: {partial}]"
                )
        else:
            fallback = f"[Evaluator stopped: no LLM responses generated after {iterations} iterations.]"

        _logger.warning(
            f"evaluate() exiting without stop: "
            f"finish_reason={'N/A' if not responses else responses[-1].choices[0].finish_reason}, "
            f"iterations={iterations}"
        )
        return fallback, metadata