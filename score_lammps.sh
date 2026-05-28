#!/usr/bin/env bash
# Score a LAMMPS experiment run with batch_lammps_evaluate.py.
#
# Usage:
#   bash score_lammps.sh <run_name> [--no-llm-judge]
#
# Example:
#   bash score_lammps.sh run5
#   bash score_lammps.sh run6 --no-llm-judge
#
# Evaluates all agents from run5.sh against the LAMMPS ground truth.
# Results are written to data/eval/lammps_scores/<agent>/<run_name>/.

set -euo pipefail

RUN_NAME="${1:?Usage: score_lammps.sh <run_name> [--no-llm-judge]}"
EXTRA_ARGS="${2:-}"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

RESULTS_ROOT="${REPO_ROOT}/data/eval/run${RUN_NAME#run}_results"
# Accept run name with or without the "run" prefix
if [[ "$RUN_NAME" == run* ]]; then
    RESULTS_ROOT="${REPO_ROOT}/data/eval/${RUN_NAME}_results"
else
    RESULTS_ROOT="${REPO_ROOT}/data/eval/run${RUN_NAME}_results"
    RUN_NAME="run${RUN_NAME}"
fi

GT_DIR="${REPO_ROOT}/data/eval/experiments_lammps_gt"
EXPERIMENTS_DIR="${REPO_ROOT}/data/eval/experiments_lammps"
SCORES_ROOT="${REPO_ROOT}/data/eval/lammps_scores"

AGENTS=(
    lammps_vanilla
    lammps_plugin
    lammps_plugin_validate
    lammps_plugin_no_hook
    lammps_plugin_no_rag
    lammps_plugin_validate_no_rag
    lammps_deepseek_vanilla
    lammps_deepseek_plugin
    lammps_deepseek_plugin_validate
    lammps_deepseek_no_hook
    lammps_deepseek_no_rag
    lammps_deepseek_validate_no_rag
)

echo "Scoring run: ${RUN_NAME}"
echo "Results root: ${RESULTS_ROOT}"
echo ""

for AGENT in "${AGENTS[@]}"; do
    AGENT_RUN_DIR="${RESULTS_ROOT}/${AGENT}/${RUN_NAME}"
    RESULTS_DIR="${SCORES_ROOT}/${AGENT}/${RUN_NAME}"

    if [[ ! -d "$AGENT_RUN_DIR" ]]; then
        echo "  SKIP  ${AGENT}  (${AGENT_RUN_DIR} not found)"
        continue
    fi

    echo "--- ${AGENT} ---"
    mkdir -p "$RESULTS_DIR"

    uv run python scripts/eval/batch_lammps_evaluate.py \
        --agent-run-dir "$AGENT_RUN_DIR" \
        --ground-truth-dir "$GT_DIR" \
        --experiments-dir "$EXPERIMENTS_DIR" \
        --results-dir "$RESULTS_DIR" \
        ${EXTRA_ARGS}

    echo ""
done

echo "Done. Scores written to ${SCORES_ROOT}/*/${RUN_NAME}/"
