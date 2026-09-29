#!/bin/bash
# Shared logic for the MADDIE test scripts.
#
# Usage: source this from a wrapper that MUST set all of: PROMPT, MAX_ITER,
# BACKEND ("openai" | "qwen"), ENV_DIR, LTM_DIR, SOURCE_DIR, SOURCE_LTM, LOG_DIR;
# then call run_maddie. Any of these left unset makes run_maddie error out (no
# silent defaults). ENV_DIR/LTM_DIR are reset fresh each run from the template
# dirs SOURCE_DIR and SOURCE_LTM. LOG_DIR must be unique per run (concurrent
# array tasks sharing one LOG_DIR would clobber each other's logs).
# Optionally set PDB_FILE to a .pdb path; it is copied into the reset ENV_DIR
# in addition to whatever SOURCE_DIR carried (used by the job-array submitter).
#
# IMPORTANT: run from the project root (the repository root), e.g.
#   sbatch tests/run_full.sh
# All paths below (cli.py, base_env/, ...) are relative to the project root,
# and this file is sourced as tests/_common.sh from that same directory.

# OPENAI_KEY_FILE="/path/to/my-openai-key.txt"   # only used when BACKEND != qwen

run_maddie() {
    : "${PDB_FILE:=}"   # optional: a target .pdb to drop into ENV_DIR

    ## every run must explicitly set these (no silent defaults)
    local var
    for var in PROMPT MAX_ITER BACKEND ENV_DIR LTM_DIR SOURCE_DIR SOURCE_LTM LOG_DIR; do
        if [[ -z "${!var:-}" ]]; then
            echo "ERROR: $var is not set by the wrapper." >&2
            return 1
        fi
    done

    cleanup() {
        kill $PID $TAIL_PID 2>/dev/null
        exit 1
    }
    trap cleanup INT TERM

    ## select model backend
    local -a LLM_ARGS
    if [[ "$BACKEND" == "qwen" ]]; then
        export OPENAI_API_KEY="EMPTY"
        # Discover the running vLLM job's node (works for both sbatch and an
        # interactive shell, as long as vllm.sh is running under your account).
        if [[ -z "${VLLM_NODE:-}" ]]; then
            VLLM_NODE=$(squeue --me --name=vllm --states=R -h -o "%N" | head -1)
            if [[ -z "$VLLM_NODE" ]]; then
                echo "ERROR: no running 'vllm' job found (squeue empty). Start vllm.sh first." >&2
                return 1
            fi
        fi
        export OPENAI_BASE_URL="http://${VLLM_NODE}:8000/v1"
        LLM_ARGS=(--llm_url "$OPENAI_BASE_URL" --llm "${QWEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}")
    else
        export OPENAI_API_KEY=$(cat "$OPENAI_KEY_FILE")
        unset OPENAI_BASE_URL
        LLM_ARGS=(--llm "gpt-5.4-mini")
    fi

    ## run-time paths (copied fresh from the SOURCE_DIR/SOURCE_LTM templates each run)
    # strip any trailing slash so the directory names are well-formed
    local RUN_ENV="${ENV_DIR%/}"
    local RUN_LTM="${LTM_DIR%/}"
    local RUN_LTM_EPS="${RUN_LTM}/episodic/"
    local RUN_LTM_SEM="${RUN_LTM}/semantic/"
    local SRC_ENV="${SOURCE_DIR%/}"
    local SRC_LTM="${SOURCE_LTM%/}"

    ## reset environment and memory
    # ensure the (possibly nested) run dirs exist; cp won't create parent dirs
    mkdir -p "${RUN_ENV}" "${RUN_LTM}"
    rm -rf "${RUN_ENV}"/*
    cp -r "${SRC_ENV}/." "${RUN_ENV}/"
    rm -rf "${RUN_LTM}"/*
    cp -r "${SRC_LTM}/." "${RUN_LTM}/"

    ## optionally remove seed files that would distract the Orchestrator
    if [[ -n "${REMOVE_SMILES:-}" ]]; then
        rm -f "${RUN_ENV}/smiles.csv"
    fi

    ## optionally add a specific target PDB alongside whatever the template carried
    # Remove any PDBs that came from SOURCE_DIR first, then copy in the real target.
    if [[ -n "$PDB_FILE" ]]; then
        if [[ ! -f "$PDB_FILE" ]]; then
            echo "ERROR: PDB_FILE '$PDB_FILE' does not exist." >&2
            return 1
        fi
        rm -f "${RUN_ENV}"/*.pdb
        cp "$PDB_FILE" "${RUN_ENV}/"
    fi

    ## logging (LOG_DIR is supplied by the wrapper; must be unique per run)
    LOG_DIR="${LOG_DIR%/}"   # strip any trailing slash
    mkdir -p "${LOG_DIR}"
    local LOG_FILE="${LOG_DIR}/$(basename "${LOG_DIR}").log"
    touch "$LOG_FILE"

    ## run MADDIE (stdout+stderr -> log file; tail mirrors it live)
    local -a EXTRA_ARGS=()
    if [[ -n "${OPTIMIZER_NUM_OUTPUTS:-}" ]]; then
        EXTRA_ARGS+=(--optimizer_num_outputs "$OPTIMIZER_NUM_OUTPUTS")
    fi

    "${MADDIE_PYTHON:-python}" cli.py \
        --input_prompt "$PROMPT" \
        --max_iter "$MAX_ITER" \
        --mode "run" \
        --log "${LOG_DIR}" \
        --env_dir "${RUN_ENV}" \
        --eps_mem_dir "${RUN_LTM_EPS}" \
        --sem_mem_dir "${RUN_LTM_SEM}" \
        "${LLM_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" &> "$LOG_FILE" & PID=$!

    tail -f "$LOG_FILE" & TAIL_PID=$!
    wait $PID
    sleep 1
    kill $TAIL_PID 2>/dev/null
}
