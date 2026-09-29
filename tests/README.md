# Benchmark run script

`run_full.sh` is the SLURM array job behind the reported results: one task per
target in `dataset/pdbs/`, ten at a time. It sources `_common.sh`, which
prepares a fresh environment for the target and invokes `cli.py`.

Submit it from the **project root**, not from `tests/`, with a vLLM job named
`vllm` already running (`sbatch vllm.sh`). Array tasks discover that job's nodes
through `squeue` and take an exclusive lock on one each, so a pool of servers is
shared without collisions; a task waits up to 30 minutes for a free node before
giving up.

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROJECT` | repo root | checkout to run from |
| `BASE_DIR` | `$PROJECT/runs/full` | where run outputs are written |
| `LOCK_DIR` | `$PROJECT/runs/node_locks` | vLLM node locks; share across scripts so they compete fairly |
| `PDB_DIR` | `$PROJECT/dataset/pdbs` | target structures |

Each task writes `$BASE_DIR/runs/<TARGET>-<timestamp>/`, holding the
orchestrator log, `states.json`, the zipped environment and memory, and
`FINAL.csv`.
