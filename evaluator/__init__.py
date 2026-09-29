"""
MADDIE Molecular Evaluator
==========================

EvaluatorAgent is the entry point: an LLM agent with tool-use that computes
every property MADDIE evaluates a molecule on. The individual tools are also
exported, so a property can be computed on its own without going through the
agent.

Quick start
-----------
    from evaluator import EvaluatorAgent

    agent = EvaluatorAgent()
    report = agent.evaluate(
        smiles="CC1=CC=C(C=C1)NC2=NC=CC(=N2)NCC3=CC=CO3",
        task_description="Evaluate as potential EGFR inhibitor.",
        pdb_path="dataset/pdbs/PSEN1.pdb",
        verbose=True,
    )
    print(report["summary"])
"""

from .agent import EvaluatorAgent
from .tools import (
    # physchem
    calculate_all_properties,
    calculate_qed,
    calculate_sascore,
    # filters
    apply_rdkit_filter,
    apply_lilly_filter,
    get_filter_details,
    # docking
    run_docking,
    # admet
    predict_toxicity,
    predict_full_admet,
    # boltz
    predict_structure_and_affinity,
)

__all__ = [
    "EvaluatorAgent",
    "calculate_all_properties",
    "calculate_qed",
    "calculate_sascore",
    "apply_rdkit_filter",
    "apply_lilly_filter",
    "get_filter_details",
    "run_docking",
    "predict_toxicity",
    "predict_full_admet",
    "predict_structure_and_affinity",
]
