"""
Evaluator usage examples.

Run from the MADDIE root directory:
    python -m evaluator.example
"""

import json

# Test molecule: Gefitinib (EGFR inhibitor, FDA-approved)
GEFITINIB = "COC1=C(OCCCN2CCOCC2)C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl"
PSEN1_PDB  = "dataset/pdbs/PSEN1.pdb"


# =============================================================================
# Example 1 – Individual tool calls (most lightweight)
# =============================================================================

def example_individual_tools():
    print("=" * 60)
    print("Example 1: Individual tool calls")
    print("=" * 60)

    from evaluator.tools import (
        calculate_all_properties,
        get_filter_details,
        predict_toxicity,
    )

    print(f"\nMolecule: Gefitinib\nSMILES: {GEFITINIB}\n")

    # --- Physicochemical properties ---
    props = calculate_all_properties(GEFITINIB)
    print("Physicochemical properties:")
    for k, v in props.items():
        print(f"  {k:20s}: {v:.3f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    # --- Molecular filters ---
    print("\nMolecular filters:")
    filters = get_filter_details(GEFITINIB)
    for k, v in filters.items():
        if not isinstance(v, list):
            print(f"  {k:20s}: {v}")
    if filters.get("PAINS_alerts"):
        print(f"  PAINS alerts      : {filters['PAINS_alerts']}")
    if filters.get("Brenk_alerts"):
        print(f"  Brenk alerts      : {filters['Brenk_alerts']}")

    # --- ADMET toxicity ---
    print("\nADMET toxicity (probability, lower = safer):")
    tox = predict_toxicity(GEFITINIB)
    for k, v in tox.items():
        flag = " ⚠" if v is not None and v > 0.5 else ""
        print(f"  {k:20s}: {v:.3f}{flag}" if v is not None else f"  {k:20s}: N/A")


# =============================================================================
# Example 3 – Docking only (requires vina/quickvina + receptor PDBQT)
# =============================================================================

def example_docking():
    print("\n" + "=" * 60)
    print("Example 3: Docking (AutoDock Vina / QuickVina-W)")
    print("=" * 60)

    from evaluator.tools import run_docking

    print(f"\nDocking {GEFITINIB[:40]}... into EGFR")
    score = run_docking(
        smiles=GEFITINIB,
        pdb_path=PSEN1_PDB,
        cache_dir="/tmp/maddie_docking_cache",
        exhaustiveness=8,
    )

    if score is None:
        print("  Docking failed – check that vina/quickvina-w is installed")
        print("  and that the PDB path exists.")
    else:
        print(f"  Binding affinity: {score:.2f} kcal/mol")
        print(f"  Threshold       : -6.53 kcal/mol")
        print(f"  Passes          : {score <= -6.53}")


# =============================================================================
# Example 4 – EvaluatorAgent (requires OPENAI_API_KEY with credits)
# =============================================================================

def example_agent():
    print("\n" + "=" * 60)
    print("Example 4: EvaluatorAgent (OpenAI GPT + tool-use)")
    print("=" * 60)

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("  Skipped – set OPENAI_API_KEY environment variable first:")
        print("  export OPENAI_API_KEY='sk-...'")
        return

    from evaluator import EvaluatorAgent

    agent = EvaluatorAgent(model="gpt-4o-mini")  # cheapest model
    summary, metadata = agent.evaluate(
        prompt=(
            "Evaluate the molecules in sample_molecules.csv (SMILES column: 'smiles') "
            "as potential EGFR kinase inhibitors. "
            "Check drug-likeness, toxicity flags, and overall suitability."
        ),
        path_to_env="evaluator/dataset/",
        verbose=True,
    )

    print("\n--- Tool results ---")
    for tool, result in metadata["tool_results"].items():
        print(f"\n[{tool}]")
        print(json.dumps(result, indent=2, default=str)[:400])

    print("\n--- Agent summary ---")
    print(summary)


# =============================================================================

if __name__ == "__main__":
    # example_individual_tools()
    # example_docking()        # comment out if vina not installed
    example_agent()
