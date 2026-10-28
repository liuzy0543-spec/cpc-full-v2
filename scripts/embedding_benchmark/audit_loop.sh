#!/usr/bin/env bash
# The audit loop the task asks for: run the full pipeline, audit it, and stop only
# when the audit passes.  Each round writes `docs/audit/round_NN_findings.md`; a
# failing round is meant to be read, fixed, and re-run -- this script does not
# fix anything itself, because a loop that repairs its own input can pass by
# weakening the gate.
#
# Usage:
#   bash scripts/embedding_benchmark/audit_loop.sh            # up to 4 rounds
#   bash scripts/embedding_benchmark/audit_loop.sh 6          # up to 6 rounds
#   SKIP_PIPELINE=1 bash .../audit_loop.sh 2                  # audit only
#
# Exit codes: 0 = the audit passed (stop), 1 = rounds used up without passing.
set -u

REPO="/d/项目/CellPainting-Claw-v2"
PY="D:/ep_bench/venv/Scripts/python.exe"
MAX_ROUNDS="${1:-4}"
CELLS="E:/bbbc022/cells"

export PATH="/c/Users/DELL/miniconda3/envs/torch_gpu/Library/bin:$PATH"
export PYTHONPATH="$REPO/src"
export CPC_WEIGHTS_ROOT="E:/cpc_weights"
export PYTHONIOENCODING="utf-8"

cd "$REPO" || exit 1

round=1
while [ "$round" -le "$MAX_ROUNDS" ]; do
    echo ""
    echo "==================== ROUND $round / $MAX_ROUNDS ===================="

    # Step 1: regenerate everything.  The audit must judge the artefacts this
    # round produced, never a leftover from a previous fix.
    if [ "${SKIP_PIPELINE:-0}" != "1" ]; then
        echo "[1/2] pipeline"
        if ! bash scripts/embedding_benchmark/run_e2e.sh; then
            echo "  pipeline FAILED -- see $CELLS/_e2e_run.log"
        fi
    else
        echo "[1/2] pipeline skipped (SKIP_PIPELINE=1)"
    fi

    # Step 2: audit it.
    echo "[2/2] audit"
    if "$PY" scripts/embedding_benchmark/audit_round.py --round "$round"; then
        echo ""
        echo "STOPPED: round $round passed. Findings: docs/audit/round_$(printf '%02d' "$round")_findings.md"
        exit 0
    fi

    echo ""
    echo "round $round did NOT pass. Read docs/audit/round_$(printf '%02d' "$round")_findings.md,"
    echo "fix the listed items, then run this script again for the next round."
    round=$((round + 1))
    # Without an intervening fix the next round audits identical code and will
    # fail identically; that is intentional, it makes "no fix" visible as a
    # repeated round rather than as progress.
done

echo ""
echo "EXIT: used $MAX_ROUNDS rounds without passing."
exit 1