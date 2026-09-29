# HitIdentifier

HitIdentifier is the entry point of a campaign: it generates candidate
molecules against a target pocket. It works in rounds, choosing a generation
strategy each round and scoring what comes back, until enough molecules meet
the requested criteria or the round budget runs out.

| Strategy | What it does |
| --- | --- |
| **SBDD** | Generates molecules from the target pocket (conDitar). |
| **LBDD** | Generates shape-similar molecules from a ligand seed (DiffSMol). |
| **Virtual screening** | Screens a SMILES drug sheet with Boltz-2. |

Every returned molecule is scored with Vina, QED and SAScore — lower Vina and
SAScore are better, higher QED is better. It exposes the shared agent
interface:

```python
response, metadata = agent.run(prompt: str)
```

The orchestrator calls it through
[`cli_hit_identifier.py`](../cli_hit_identifier.py).

---

## Installation

```bash
conda create -p /path/to/envs/maddie-hit-identifier --file hit_identifier/conda-requirements.txt
conda activate /path/to/envs/maddie-hit-identifier
pip install -r hit_identifier/pip-requirements.txt
```

`conda-requirements.txt` is a `linux-64` lock file. `AutoDockTools_py3` is
commented out in the pip file and has to come from git.

---

## Configuration

Set in `.env`:

| Variable | Points at |
| --- | --- |
| `MADDIE_ENV_HIT_IDENTIFIER` | hit identifier conda prefix |
| `UNIDOCK_PATH` | Uni-Dock, if not already importable |

The generators run as **subprocesses under their own interpreter**, not as
imports, so their settings live in
[`config/runtime_config.json`](config/runtime_config.json) rather than `.env`:

| Key | Ours | Meaning |
| --- | --- | --- |
| `lbdd/sbdd.python_bin` | — | interpreter that runs conDitar and DiffSMol |
| `lbdd/sbdd.config` | — | generator YAML holding the checkpoint paths |
| `lbdd/sbdd.sample_size` | 100 | molecules sampled per round |
| `virtual_screen.sample_size`, `.top_k` | 1, 1 | drugs screened, and hits kept, per VS round |
| `docking.gpu.*` | — | QuickVina2-CUDA build, used when Uni-Dock is unavailable |
| `tmp_root` | — | scratch space |

No per-target configuration is needed: all file paths are given in the prompt
at runtime.

---

## Running

```bash
conda activate /path/to/envs/maddie-hit-identifier
python cli_hit_identifier.py \
  --input_prompt "Find at least 100 potential hits for ./env/PSEN1.pdb within 5 rounds. Action list: SBDD, LBDD." \
  --context_dir ./env/hit_identifier \
  --model "$QWEN_MODEL"
```

---

## Usage

Everything is specified in the prompt: the `.pdb` path (required), an `.sdf`
reference ligand for LBDD (optional), a drug sheet for virtual screening
(`.csv`/`.tsv`/`.xlsx`), the number of molecules, the round budget, metric
thresholds, and which strategies are allowed.

```text
Find at least 10 potential hits for /path/to/target.pdb with Vina < -12 within 5 rounds.

Find at least 10 hits for /path/to/target.pdb with Vina lower than -10, QED higher than 0.5, and SA lower than 5 within 5 rounds.

Find at least 10 hits for /path/to/target.pdb using reference ligand /path/to/ref.sdf within 5 rounds.

Find hits for /path/to/target.pdb with drug sheet /path/to/drugs.csv within 3 rounds.

Find at least 10 hits for /path/to/target.pdb within 5 rounds. Use SBDD and LBDD only.
```

With no metric threshold, HitIdentifier keeps the best binders by Vina. The
drug sheet only needs a `SMILES` or `smiles` column.

`run()` returns `(response, metadata)`: a summary string, and
`metadata["responses"]` with the raw response of every LLM call. The structured
result is on `agent.last_result`:

```python
{
    "success": True,
    "target": "PSEN1",
    "run_dir": "log/<timestamp>_PSEN1",
    "run_state_dir": "log/<timestamp>_PSEN1/run_state",
    "successful_results_dir": "log/<timestamp>_PSEN1/successful_results",
}
```

### The round loop

Each round: build the legal actions, ask the LLM to choose one, ask it to
choose a seed if the action is LBDD, run the generator, score the molecules
with Vina/QED/SAScore, store them, and check the success criteria.

The LLM only ever chooses among **legal** actions:

| Strategy | Legal when | Notes |
| --- | --- | --- |
| SBDD | the target has a PDB and an SBDD runner | starts from the fixed pocket; no seed |
| LBDD | a reference ligand exists, or an eligible SBDD seed does | reference ligands are used directly |
| Virtual screening | `virtual_screen.enabled` and a drug sheet in the prompt | screens drugs not yet screened; never twice in a row |

An SBDD-generated molecule is eligible as an LBDD seed only if it has a Vina
score, has at most 40 heavy atoms, and has not been used as a seed before.

---

## Outputs

Each run writes a timestamped directory under `log/`:

```text
log/<timestamp>_<target>/
├── molecule_library/MOLxxx_ROWxxx/
│   ├── molecule.sdf         every evaluated molecule
│   ├── protein.pdb          when the molecule has a predicted protein pose
│   └── metadata.json        metrics and method-specific details
├── successful_results/
│   ├── successful_molecules.json      molecules meeting the criteria, or the
│   │                                  best binders when none was requested
│   └── successful_molecules_sdf/
├── sbdd_runs/*/generation_results.json
├── lbdd_runs/*/generation_results.json    includes seed_molecule
├── virtual_screen_runs/*/virtual_screen_results.json
└── run_state/<target>.json  human-readable run log
```

Progress goes to stdout as the run proceeds; subprocess output (sampling
progress, SMILES) goes to stderr.

```
[Run] target=PSEN1  pdb=/path/to/PSEN1.pdb
[Run] methods=SBDD_GENERATE, LBDD_GENERATE  criteria: min_pool=10 | Vina<-10.0
[Round 1/2] SBDD_GENERATE — generate new molecules from the target pocket
[Round 1/2] 10 molecules | best Vina this round=-11.82 | cumulative success 7/10
[Round 2/2] LBDD_GENERATE — 7 hits so far; shape-guided generation from best seed
[Round 2/2] 10 molecules | best Vina this round=-11.00 | cumulative success 11/10
[Done] Success — 11 molecules found (goal: 10)
```

---

## Structure

```text
hit_identifier/
├── tool.py           HitIdentifierAgent: prompt parsing and the round loop
├── agent/            LLM client
├── config/           runtime_config.json, prompts, resource resolution
├── core/             actions, environment, memory, planner, error diagnosis
├── chem/             docking and property evaluation
├── runners/          subprocess entry points for conDitar, DiffSMol, Boltz-2
└── vendor/           conDitar and DiffSMol, vendored
```
