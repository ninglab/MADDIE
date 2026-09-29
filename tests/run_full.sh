#!/bin/bash
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --job-name=maddie_%a
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --array=0-49%10
#SBATCH --output=logs/sbatch-%A_%a.out
#SBATCH --error=logs/sbatch-%A_%a.error

# Runs MADDIE across the 50 benchmark targets, one array task per target, ten at
# a time. This is the configuration behind the reported results. Submit from the
# project root:
#
#   mkdir -p logs
#   sbatch tests/run_full.sh              # all 50 targets
#   sbatch --array=20 tests/run_full.sh   # one target

PROJECT=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE_DIR=${BASE_DIR:-$PROJECT/runs/full}
PDB_DIR=${PDB_DIR:-$PROJECT/dataset/pdbs}

LOCK_DIR="${LOCK_DIR:-$PROJECT/runs/node_locks}"
mkdir -p "$LOCK_DIR"

VLLM_NODE=""
for attempt in $(seq 1 60); do
    mapfile -t AVAILABLE < <(
        squeue --me --name=vllm --states=R -h -o "%N" | tr ',' '\n' | sort -u
    )
    for node in $(printf '%s\n' "${AVAILABLE[@]}" | shuf); do
        if mkdir "${LOCK_DIR}/${node}.lock" 2>/dev/null; then
            VLLM_NODE="$node"
            trap "rmdir '${LOCK_DIR}/${node}.lock' 2>/dev/null" EXIT INT TERM
            break 2
        fi
    done
    echo "All nodes busy, retrying in 30s (attempt ${attempt}/60)..."
    sleep 30
done

if [[ -z "$VLLM_NODE" ]]; then
    echo "ERROR: no free vLLM node after 60 attempts." >&2
    exit 1
fi

TARGETS=(
  ABCC8 ACHE ADAM10 ADRB1 BCHE CDK5 CYP3A5 DRD2 FLT3 GRIN2A_NDMAR2A
  HRP2 HTR2A IL1B KIT MET_HGFR P2RY12_P2Y12 PSEN1 PTGS1_COX1 RET TLR7
  ABCD1_ALD ABO_BGAT ACE ADAMTS5 AKR1B10 AKT1 BACE2 BAZ2A CD38 CDK6
  DPP7 DYRK2 FKBP1A GRK4 GSTP1 HDAC8 IDH2 IPMK MAP3K14 MME
  NOS2 NQO1 NR1H4 PAK4 PTGIS QPCT SIRT3 TBK1 TNKS1 TNKS2
)

TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

PDB_FILE="${PDB_DIR}/${TARGET}.pdb"

TS=$(date +%Y-%m-%d_%H-%M-%S)

cd "$PROJECT"
source tests/_common.sh

BACKEND=qwen
MAX_ITER=100
SOURCE_DIR=base_env
SOURCE_LTM=base_ltm
REMOVE_SMILES=1
OPTIMIZER_NUM_OUTPUTS=10

RUN_BASE="${BASE_DIR}/runs/${TARGET}-${TS}"
mkdir -p "${RUN_BASE}"

exec > "${RUN_BASE}/slurm.out" 2>&1
echo "target=$TARGET  array_task=$SLURM_ARRAY_TASK_ID  vllm_node=$VLLM_NODE  pdb=$PDB_FILE  optimizer_num_outputs=$OPTIMIZER_NUM_OUTPUTS"

ENV_DIR="${RUN_BASE}/env"
LTM_DIR="${RUN_BASE}/ltm"
LOG_DIR="${RUN_BASE}/log"

PROMPT="Find 100 diverse and novel molecules that bind to ${ENV_DIR}/${TARGET}.pdb and satisfy all of these properties: physchem_QED > 0.36; docking_kcal_mol < -6.53; physchem_SAScore < 5.35; admet_hERG < 0.96; admet_ClinTox < 0.71; admet_Mutagenicity < 0.51; filter_LILLY == True. Each molecule must also have at least one Synthesis Path. Store the final set in FINAL.csv. <STOP=MAX_ITER> <FILE=FINAL.csv> <POOL=100>"

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL
run_maddie
