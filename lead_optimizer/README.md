# LeadOptimizer

LeadOptimizer improves molecules that bind but fail a requirement. It receives molecules from a CSV file and writes optimized SMILES to a new CSV. 

For each request, LeadOptimizer builds an optimization plan, selects
an optimizer, validates and de-duplicates variants, and may retry or
switch optimizers within its execution budget.
It exposes the shared agent interface:

```python
response, metadata = agent.run(prompt: str, path_to_env: str)
```
The orchestrator calls it through [`cli_lead_optimizer.py`](../cli_lead_optimizer.py).

--- 
## Installation

```bash
conda env create -f lead_optimizer/environment-lead-optimizer-all.yml -p /path/to/envs/maddie-lead-optimizer
conda activate /path/to/envs/maddie-lead-optimizer
python -m pip install --no-deps -r lead_optimizer/requirements-lead-optimizer-all.no-deps.txt
```
---

## Configuration

Set these in `.env`:

| Variable | Points at |
| --- | --- |
| `MADDIE_ENV_LEAD_OPTIMIZER` | lead optimizer conda prefix |
| `GELLMO_REPO` | GeLLMO-C checkout |
| `GELLMO_BASE_MODEL`, `GELLMO_LORA_WEIGHTS` | GeLLMO-C model weights |
| `MOLOPT_REPO`, `MOLOPT_BENCHMARK_PATH` | MolOpt checkout and benchmark records |

The worker inherits `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the
orchestrator.

---

## Running

```bash
conda activate /path/to/envs/maddie-lead-optimizer
python cli_lead_optimizer.py \
  --input_prompt "Optimize candidates.csv: increase QED, decrease hERG." \
  --context_dir ./env/lead_optimizer \
  --model "$QWEN_MODEL" \
  --num_outputs 10
```
`--num_outputs` is the number of optimized variants requested per input
molecule.

---

## Usage

The prompt should mention a CSV file:

```text
Optimize the molecules in sample.csv by increasing QED above 0.8.
Return 5 candidates.
```

Supported objectives:

```text
QED  SAScore  LogP  molecular weight  TPSA
HBD  HBA  rotatable bonds
hERG  DILI  ClinTox  Mutagenicity  Carcinogenicity
```

The CSV can live in either location:

```text
path_to_env/sample.csv
path_to_env/input/sample.csv
```

Common molecule column names are accepted, including `smiles`,
`canonical_smiles`, and `molecule`. Common ID columns such as `id`, `name`, and
`compound_id` are also accepted.

Example:

```csv
id,smiles
aspirin_like,CC(=O)Oc1ccccc1C(=O)O
drd2_seed,N#Cc1ccc(cc1)N(CCCl)CCCl
```

Output CSV columns:

```text
input_id,input_smiles,optimized_smiles,output_id,adjustment,tool,property,input_property_value,optimized_property_value
```

If the input is directly in `path_to_env`, the output is written next to it:

```text
path_to_env/smiles.csv
path_to_env/smiles_output.csv
```

If the input is in `path_to_env/input`, the output is written to:

```text
path_to_env/output/lead_optimizer/sample_optimized.csv
```
---

## Structure

```text
lead_optimizer/
├── agent.py          LeadOptimizerAgent and worker interface
├── core.py           CSV path resolution and read/write helpers
├── planner.py        request parsing and high-level tool choice
├── planning/         JSON planner prompt, schema, parser, and selection logic
├── strategy/         bounded retry and tool-switch decisions
├── execution/        CSV optimization runners
├── tools/            GeLLMO-C and MolOpt adapters
├── oracles/          objective parsing and scoring registry
├── response/         final response summary builder
├── config/           presets and benchmark loading
└── utils/            logging helpers
```

