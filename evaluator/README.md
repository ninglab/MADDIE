# Evaluator

Evaluator is the gate: nothing enters a campaign's accepted pool until it has
passed through here. Given a CSV of molecules and a target structure, it
computes physicochemical properties, structural filters, ADMET endpoints and a
docking score for every molecule, and writes them back as columns. It exposes
the shared agent interface:

```python
summary, metadata = agent.evaluate(prompt: str, path_to_env: str)
```

The orchestrator calls it through [`cli_evaluator.py`](../cli_evaluator.py).

---

## Installation

```bash
conda create -p /path/to/envs/maddie-evaluator python=3.10 -y
conda activate /path/to/envs/maddie-evaluator
conda install -c conda-forge rdkit          # conda is more reliable than pip here
pip install -r evaluator/requirements.txt
```

Docking runs on **Uni-Dock** when it is importable, with AutoDock Vina as the
fallback. Uni-Dock is installed separately.

---

## Configuration

Set in `.env`:

| Variable | Points at |
| --- | --- |
| `MADDIE_ENV_EVALUATOR` | evaluator conda prefix |
| `UNIDOCK_PATH` | Uni-Dock, if not already importable |

The worker inherits `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the
orchestrator.

---

## Running

```bash
conda activate /path/to/envs/maddie-evaluator
python cli_evaluator.py \
  --input_prompt "Evaluate ./env/evaluator/000_hits.csv using ./env/PSEN1.pdb" \
  --context_dir ./env/evaluator \
  --model "$QWEN_MODEL"
```

The prompt has to name the CSV and the PDB by **absolute path**: the agent sees
only its own workspace, so the orchestrator stages both files there first.

---

## Metrics

Every molecule gets all of the columns below. The names are exactly what
appears in the output CSV and what the orchestrator filters on.

| Column | Tool | Meaning |
| --- | --- | --- |
| `physchem_QED` | RDKit | drug-likeness, 0–1, higher is better |
| `physchem_SAScore` | RDKit | synthetic accessibility, 1–10, lower is easier |
| `physchem_MolecularWeight`, `_LogP`, `_TPSA`, `_HBD`, `_HBA`, `_RotatableBonds`, `_RingCount`, `_AromaticRings`, `_HeavyAtomCount` | RDKit | physicochemical descriptors |
| `filter_PAINS` | RDKit | passes the PAINS A/B/C alerts |
| `filter_Brenk` | RDKit | passes the Brenk structural alerts |
| `filter_RDKIT` | RDKit | PAINS + Brenk + a relaxed Ro5 (MW ≤ 600, LogP ≤ 6, HBD ≤ 5, HBA ≤ 10) |
| `filter_LILLY` | medchem | passes the Lilly MedChem Rules |
| `admet_hERG` | ADMET-AI | cardiac ion channel toxicity, probability, lower is safer |
| `admet_DILI` | ADMET-AI | drug-induced liver injury |
| `admet_ClinTox` | ADMET-AI | clinical toxicity |
| `admet_Mutagenicity` | ADMET-AI | Ames mutagenicity |
| `admet_Carcinogens` | ADMET-AI | carcinogenicity |
| `docking_kcal_mol` | Uni-Dock | binding affinity, more negative is stronger |

`filter_*` columns are True when the molecule passes; `filter_PAINS_alerts` and
`filter_Brenk_alerts` name the alerts that fired.

ADMET is predicted for the whole batch in one model call; the rest is computed
per molecule.

---

## Usage

### As a library

The tools can be called on their own, without the agent:

```python
from evaluator.tools import (
    calculate_all_properties,
    get_filter_details,
    predict_toxicity,
    run_unidock,
)

smiles = "COC1=C(OCCCN2CCOCC2)C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl"

calculate_all_properties(smiles)
# {'QED': 0.518, 'SAScore': 2.337, 'MolecularWeight': 446.15, ...}

get_filter_details(smiles)
# {'RDKIT': False, 'LILLY': False, 'Brenk_alerts': ['Aliphatic_long_chain'], ...}

predict_toxicity(smiles)
# {'hERG': 0.962, 'DILI': 0.924, 'ClinTox': 0.795, ...}

run_unidock(smiles, pdb_path="dataset/pdbs/PSEN1.pdb")
# -10.12
```

### As an agent

`EvaluatorAgent` decides which tools to call and writes the evaluated CSV
itself. It is built for batch evaluation: it discovers the input CSV and the
PDB in `path_to_env` and evaluates every row.

```python
from evaluator import EvaluatorAgent

agent = EvaluatorAgent(
    model="gpt-4o",     # default
    max_tokens=4096,    # default
    max_iterations=10,  # default — LLM turns, not molecules
    api_key=None,       # falls back to OPENAI_API_KEY
)
summary, metadata = agent.evaluate(
    prompt="Evaluate the molecules in sample_molecules.csv (SMILES column: 'smiles').",
    path_to_env="evaluator/dataset/",
    verbose=True,
)

print(summary)                   # the written assessment
print(metadata["tool_results"])  # raw output of each tool call
print(metadata["iterations"])    # LLM turns used
```

`evaluator/dataset/` holds a small sample input and its evaluated output.
[`example.py`](example.py) runs the tools, a docking call, and the agent in
turn:

```bash
python -m evaluator.example
```

---

## Structure

```text
evaluator/
├── agent.py          EvaluatorAgent: tool-use loop and batch evaluation
├── example.py        runnable examples
├── dataset/          sample input and its evaluated output
└── tools/
    ├── physchem.py       RDKit descriptors, QED, SAScore
    ├── filters.py        PAINS, Brenk, relaxed Ro5, Lilly MedChem Rules
    ├── admet.py          ADMET-AI, batched
    ├── unidock.py        Uni-Dock — the docking backend used
    ├── autodockvina.py   AutoDock Vina fallback
    ├── docking.py        AutoDock Vina / QuickVina-W command-line path
    ├── quickvina2cuda.py QuickVina2-CUDA
    ├── boltz.py          Boltz-2 structure prediction and affinity
    ├── diversity.py      set-level diversity
    └── ncircle.py        #Circles diversity metric
```
