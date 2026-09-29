# MADDIE: Multi-Agent Drug Discovery Engine

Given a target protein and a set of desired molecular properties, MADDIE
searches *in silico* for molecules that bind the target, satisfy every
property, and can be synthesized. It comprises a central agent, Orchestrator,
and four specialized worker agents, each carrying out one step of a drug
discovery campaign.

| Agent | Role |
| --- | --- |
| **Orchestrator** | Decides how the campaign proceeds: which worker agent to call next, and which candidate molecules to focus on. A memory module sustains a long-horizon campaign. |
| **HitIdentifier** | The entry point. The only agent that conditions generation on the target structure — conDitar (structure-based) or DiffSMol (ligand-based). |
| **LeadOptimizer** | The refinement stage. Advances existing candidates toward satisfying properties — GeLLMO-C, or one of 11 MolOpt algorithms. |
| **Synthesizer** | Bridges *in silico* design and practice. Proposes synthesis routes (MEEA\*), or synthesizable analogs (PrexSyn) where no route is found. |
| **Evaluator** | Quality control. Binding affinity, drug-likeness, ADMET and structural liabilities. |

Under Orchestrator's coordination the workers are composed into flexible
workflows rather than a fixed pipeline, so a campaign adapts as it unfolds. It
ends once the requested number of molecules both satisfy every property and are
synthesizable, or once the action budget is spent.

---

## Repository layout

| Path | What it is |
| --- | --- |
| `dataset/pdbs/` | the 50 benchmark targets |
| `orchestrator/` | planning loop, working memory, worker dispatch. `agent.py` holds Orchestrator's system prompt |
| `hit_identifier/` | generation agent; `vendor/` holds conDitar and DiffSMol |
| `lead_optimizer/` | planner plus the GeLLMO-C and MolOpt adapters |
| `synthesizer/` | ReAct agent wrapping MEEA\* and PrexSyn |
| `evaluator/` | property, filter, ADMET and docking tools |
| `base_env/`, `base_ltm/` | run templates; `base_ltm/semantic/workers/*/PROFILE.md` are the worker descriptions Orchestrator reads |
| `tests/` | the benchmark: `run_full.sh` runs all 50 targets as a SLURM array job |
| `cli.py` | orchestrator entry point — this is what you run |
| `cli_<agent>.py` | per-agent entry points, launched by the orchestrator |
| `vllm.sh` | serves the backbone LLM (SLURM) |

---

## Installation

Linux with conda, `uv`, Apptainer or Singularity, and GPUs — two A100s served
the 30B backbone, one more per campaign. SLURM only for [`vllm.sh`](vllm.sh) and
[`tests/run_full.sh`](tests/run_full.sh).

Every agent needs its own environment; put the four worker prefixes in
[`.env.example`](.env.example).

```bash
# orchestrator
conda create -n maddie python=3.10 -y && conda activate maddie
pip install -r requirements.txt

# HitIdentifier
conda create -p /path/to/envs/maddie-hit-identifier --file hit_identifier/conda-requirements.txt
conda activate /path/to/envs/maddie-hit-identifier && pip install -r hit_identifier/pip-requirements.txt

# LeadOptimizer
conda env create -f lead_optimizer/environment-lead-optimizer-all.yml -p /path/to/envs/maddie-lead-optimizer
conda activate /path/to/envs/maddie-lead-optimizer
python -m pip install --no-deps -r lead_optimizer/requirements-lead-optimizer-all.no-deps.txt

# Synthesizer
conda create -p /path/to/envs/maddie-synthesizer python=3.10 -y
conda activate /path/to/envs/maddie-synthesizer && pip install -e synthesizer/

# Evaluator
conda create -p /path/to/envs/maddie-evaluator python=3.10 -y
conda activate /path/to/envs/maddie-evaluator && pip install -r evaluator/requirements.txt
```

conDitar and DiffSMol run as subprocesses in one more environment, built from
upstream DiffSMol. Uni-Dock, the model weights and the `.sif` images are
obtained separately.

---

## Configuration

Copy [`.env.example`](.env.example) to `.env` — agent conda prefixes, GeLLMO-C
and mol-opt paths, Synthesizer assets, Uni-Dock. Unset variables fall back to a
`/path/to/...` placeholder that fails loudly.

### Command-line flags

| Flag | Ours | Meaning |
| --- | --- | --- |
| `--input_prompt` | *required* | the campaign request, control tags included |
| `--max_iter` | 100 | action budget; every delegation and processing step costs one |
| `--optimizer_num_outputs` | 10 | derivatives per input molecule |
| `--env_dir` | `./env` | campaign working directory |
| `--eps_mem_dir`, `--sem_mem_dir` | `./ltm/*` | episodic and semantic memory |
| `--log` | — | transcript, `states.json`, snapshots |
| `--llm` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | also `claude-sonnet-4-6`, `gpt-5.4-mini` |
| `--llm_url` | the vLLM endpoint | required unless `--llm` is hosted |
| `--working_memory_threshold` | 32000 | tokens before working memory is compacted |
| `--thinking`, `--cache` | off | Claude only |

### Control tags

Read out of `--input_prompt`:

| Tag | Effect |
| --- | --- |
| `<POOL=N>` | stop once N molecules pass every requirement and are synthesizable |
| `<FILE=name.csv>` | do not finish until that file exists and its contents check out |
| `<STOP=MAX_ITER>` | override Orchestrator's own stop; run until the budget or `<POOL=N>` |

---

## Running

**1. Serve the LLM.**

```bash
mkdir -p logs                                  # SLURM refuses to start without it
sbatch vllm.sh
export OPENAI_BASE_URL=http://<node>:8000/v1   # the node vllm.sh prints
export OPENAI_API_KEY=EMPTY                    # vLLM ignores the key
```

**2. Prepare a run directory.** `base_env/` ships PSEN1; for another target,
`rm env/*.pdb && cp dataset/pdbs/<TARGET>.pdb env/`.

```bash
cp -r base_env env && cp -r base_ltm ltm
```

**3. Run.** The reported configuration; a campaign takes a few hours.

```bash
python cli.py \
  --input_prompt "Find 100 diverse and novel molecules that bind to $PWD/env/PSEN1.pdb and satisfy all of these properties: physchem_QED > 0.36; docking_kcal_mol < -6.53; physchem_SAScore < 5.35; admet_hERG < 0.96; admet_ClinTox < 0.71; admet_Mutagenicity < 0.51; filter_LILLY == True. Each molecule must also have at least one Synthesis Path. Store the final set in FINAL.csv. <STOP=MAX_ITER> <FILE=FINAL.csv> <POOL=100>" \
  --max_iter 100 \
  --optimizer_num_outputs 10 \
  --env_dir ./env --eps_mem_dir ./ltm/episodic --sem_mem_dir ./ltm/semantic \
  --log ./logs/run1 \
  --llm "Qwen/Qwen3-30B-A3B-Instruct-2507" --llm_url "$OPENAI_BASE_URL"
```

**4. Read the result.** `env/FINAL.csv` holds the accepted molecules,
every property already a column.

### The benchmark

One task per target, same settings as above, at most ten at a time.

```bash
sbatch tests/run_full.sh              # all 50 targets
sbatch --array=20 tests/run_full.sh   # one target
```

---

## Third-party components

### Molecule generation

- **conDitar** — structure-based generation in HitIdentifier
  (`hit_identifier/vendor/conditar/`).\
  R. Gao, J. Peng, Z. Chen, F. N. Baker, et al. *Generating Developable 3D
  Molecules via Pocket-Conditioned Diffusion and Property-Aware Optimization.*
  arXiv:2607.12349, 2026.
- **DiffSMol** — ligand-based generation in HitIdentifier
  (`hit_identifier/vendor/diffsmol/`).\
  Z. Chen, B. Peng, T. Zhai, D. Adu-Ampratwum, X. Ning. *Generating 3D small
  binding molecules using shape-conditioned diffusion models with guidance.*
  Nature Machine Intelligence 7(5):758–770, 2025.

### Molecule optimization

- **GeLLMO-C** — multi-property optimizer in LeadOptimizer
  (`lead_optimizer/GeLLMO-C/`).\
  V. Dey, X. Hu, X. Ning. *Large language models for controllable multi-property
  multi-objective molecule optimization.* Findings of EMNLP 2025, 20996–21023.
- **MolOpt** — the 11 oracle-guided single-property optimizers in LeadOptimizer.
  Installed separately; see `MOLOPT_REPO` in [`.env.example`](.env.example).\
  W. Gao, T. Fu, J. Sun, C. Coley. *Sample efficiency matters: a benchmark for
  practical molecular optimization.* NeurIPS 35:21342–21357, 2022.

### Synthesis

- **MEEA\*** — retrosynthesis planning in Synthesizer
  (`synthesizer/tools/MEEA/`; bundles `rdchiral`, which keeps its own license).\
  D. Zhao, S. Tu, L. Xu. *Efficient retrosynthetic planning with MCTS
  exploration enhanced A\* search.* Communications Chemistry 7(1):52, 2024.
- **PrexSyn** — synthesizable analog generation in Synthesizer
  (`synthesizer/tools/PrexSyn/`).\
  S. Luo, C. W. Coley. *Efficient and programmable exploration of synthesizable
  chemical space.* TMLR, 2026.

### Evaluation

- **Uni-Dock** — binding affinity. Installed separately; see `UNIDOCK_PATH` in
  [`.env.example`](.env.example).\
  Y. Yu, C. Cai, J. Wang, Z. Bo, Z. Zhu, H. Zheng. *Uni-Dock: GPU-accelerated
  docking enables ultralarge virtual screening.* JCTC 19(11):3336–3345, 2023.
- **ADMET-AI** — the five ADMET endpoints.\
  K. Swanson, P. Walther, J. Leitz, et al. *ADMET-AI: a machine learning ADMET
  platform for evaluation of large-scale chemical libraries.* Bioinformatics
  40(7):416, 2024.
- **Lilly MedChem Rules** — structural liability filter, via the `medchem`
  package.\
  R. F. Bruns, I. A. Watson. *Rules for identifying potentially reactive or
  promiscuous compounds.* J. Med. Chem. 55(22):9763–9772, 2012.
- **QED** and **SAScore** — drug-likeness and synthetic accessibility, via
  RDKit.\
  G. R. Bickerton, G. V. Paolini, J. Besnard, S. Muresan, A. L. Hopkins.
  *Quantifying the chemical beauty of drugs.* Nature Chemistry 4(2):90–98, 2012.\
  P. Ertl, A. Schuffenhauer. *Estimation of synthetic accessibility score of
  drug-like molecules based on molecular complexity and fragment
  contributions.* J. Cheminformatics 1(1):8, 2009.

### Backbone LLM

- **Qwen3-30B-A3B-Instruct** — the model behind Orchestrator and all four
  worker agents.\
  A. Yang, A. Li, B. Yang, et al. *Qwen3 technical report.* arXiv:2505.09388,
  2025.
- **vLLM** — serves it locally; see [`vllm.sh`](vllm.sh).\
  W. Kwon, Z. Li, S. Zhuang, et al. *Efficient memory management for large
  language model serving with PagedAttention.* SOSP 2023.

---

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).
[`LICENSE`](LICENSE) governs; this is a summary.

- Use, modification and redistribution are permitted **for noncommercial
  purposes only**.
- Educational institutions and public research organizations qualify
  *regardless of funding source*; use on behalf of a company does not.
- Redistribution must pass these terms on.

Bundled components keep their own licenses:

| Component | License |
| --- | --- |
| `hit_identifier/vendor/diffsmol/` | PolyForm Noncommercial 1.0.0 |
| `lead_optimizer/GeLLMO-C/` | MIT |
| `synthesizer/tools/PrexSyn/prexsyn/` | MIT |
| `synthesizer/tools/MEEA/MEEA/rdchiral/` | MIT |

Separately installed components — MolOpt, Uni-Dock, ADMET-AI, `medchem`, RDKit,
and the MEEA and PrexSyn images — carry their own licenses.
