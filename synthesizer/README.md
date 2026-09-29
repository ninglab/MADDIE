# Synthesizer

Synthesizer decides whether a molecule can actually be made, by trying to build
a synthesis route for it. Where it cannot find one, it generates synthesizable
analogs instead — molecules close to the original that do have a route. It
exposes the shared agent interface:

```python
response, metadata = agent.run(prompt: str, path_to_env: str)
```

The orchestrator calls it through
[`cli_synthesizer.py`](../cli_synthesizer.py).

---

## Installation

```bash
conda create -p /path/to/envs/maddie-synthesizer python=3.10 -y
conda activate /path/to/envs/maddie-synthesizer
pip install -e synthesizer/
```

Dependencies are declared in [`pyproject.toml`](pyproject.toml). You also need
to be able to run `apptainer`/`singularity` containers, which is what the two
synthesis tools use.

### Containers and large files

MEEA\* and PrexSyn run from `meea.sif` and `prexsyn.sif`, which can be
downloaded:

```bash
curl -O https://data.frazierbaker.com/meea.sif
curl -O https://data.frazierbaker.com/prexsyn.sif
```

MEEA\* needs further large files, from the links in the
[MEEA\* repository README](https://github.com/CMACH508/MEEA). Put both images
and those checkpoints in one directory:

```
meea.sif  prexsyn.sif
eMoleculesBB.txt  EnamineREAL.txt  policy_model.ckpt  template_rules.dat  value_pc.pt
```

---

## Configuration

Set in `.env`:

| Variable | Points at |
| --- | --- |
| `MADDIE_ENV_SYNTHESIZER` | synthesizer conda prefix |
| `MADDIE_SYNTHESIZER_ASSETS` | the directory holding the images and checkpoints |

The `/path/to/synthesizer-assets` in
[`cli_synthesizer.py`](../cli_synthesizer.py) is only the fallback used when
`MADDIE_SYNTHESIZER_ASSETS` is unset.

The worker inherits `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the
orchestrator and gets its model name from `--model`. `LLM_BASEURL`,
`LLM_APIKEY` and `LLM_MODELNAME` override those if you ever need the
Synthesizer on a different server.

---

## Running

```bash
conda activate /path/to/envs/maddie-synthesizer
python cli_synthesizer.py \
  --input_prompt "Find synthesis routes for the molecules in ./env/synthesizer/property_pool.csv" \
  --context_dir ./env/synthesizer \
  --model "$QWEN_MODEL"
```

The input must be a CSV with a `smiles` column, staged in Synthesizer's own
workspace — the orchestrator copies it there before delegating.

---

## Usage

Every molecule first goes to retrosynthesis planning. For those with no route,
five synthesizable analogs are generated instead. If one of those analogs is an
exact match for the original, the original counts as synthesizable; otherwise
the analogs are returned as candidates worth evaluating in their own right.

Output columns:

| Column | Meaning |
| --- | --- |
| `smiles` | the input molecule |
| `synthesizable_decision` | `YES` if a route was confirmed, `NO` otherwise |
| `source` | `MEEA` for a direct route, `PrexSyn` for one found via the analog path, empty for `NO` |
| `analog_smiles_1` … `_5` | analogs proposed for a `NO` molecule |

Synthesizer can also be used outside MADDIE as a standalone synthesizability
check: a molecule counts as synthesizable if MEEA\* finds a route or an exact
match analog comes back, and unsynthesizable otherwise.

---

## Structure

```text
synthesizer/
├── agent/            the ReAct loop: tool documentation, action history, dispatch
├── llm/              OpenAI-compatible client
├── tools/
│   ├── base.py       Tool base class
│   ├── finish.py     terminal action
│   ├── code.py       Python execution, for inspecting a failure
│   ├── sascore.py    synthetic accessibility score
│   ├── MEEA/         retrosynthesis planning, containerized
│   └── PrexSyn/      synthesizable analog generation, containerized
├── data/             Route data structures
├── serve/            optional HTTP front end
├── utils/            timing and logging helpers
└── tests/
```
