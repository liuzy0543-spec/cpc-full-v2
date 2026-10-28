#!/usr/bin/env bash
# End-to-end run: regenerate both profiles from scratch, then score and verify.
# Written as one script so the whole chain shares a single log and a single
# timestamp, and so a failure in step N is visible as a stopped log rather than
# as a downstream script reading stale artefacts.
set -u

REPO="/d/项目/CellPainting-Claw-v2"
# Dataset directory is overridable so the same chain can run on a differently
# sized build (e.g. the 11.6k-cell plate 20585 set) without editing the script.
# Every artefact and the log land next to the inputs they came from.
CELLS="${CELLS_DIR:-E:/bbbc022/cells}"
PY="${PYTHON_BIN:-D:/ep_bench/venv/Scripts/python.exe}"
LOG="$CELLS/_e2e_run.log"

export PATH="/c/Users/DELL/miniconda3/envs/torch_gpu/Library/bin:$PATH"
export PYTHONPATH="$REPO/src"
export CPC_WEIGHTS_ROOT="E:/cpc_weights"
export PYTHONIOENCODING="utf-8"

cd "$REPO" || exit 1
: > "$LOG"

step() {
    echo "" | tee -a "$LOG"
    echo "########## $* ##########" | tee -a "$LOG"
    date '+  started %Y-%m-%d %H:%M:%S' | tee -a "$LOG"
}

run() {
    "$@" >> "$LOG" 2>&1
    local rc=$?
    echo "  exit=$rc  $(date '+%H:%M:%S')" | tee -a "$LOG"
    return $rc
}

step "1/6  v2 embeddings  (embed_dataset --force, 4 official methods)"
run "$PY" scripts/embedding_benchmark/embed_dataset.py --cells-dir "$CELLS" --force
V2_RC=$?

step "2/6  v3 embeddings  (build_v3_surrogates --force, 3 surrogate methods)"
run "$PY" scripts/embedding_benchmark/build_v3_surrogates.py --cells-dir "$CELLS" --force
V3_RC=$?

step "3/6  IO uniformity  v2 (7-method script, v2 profile, limit 64)"
run "$PY" scripts/embedding_benchmark/verify_io_uniformity.py \
    --cells-dir "$CELLS" --profile v2 --limit 64 \
    --output "$CELLS/io_uniformity_v2_e2e.json"
IO2_RC=$?

step "4/6  IO uniformity  v3 (all seven, limit 64)"
run "$PY" scripts/embedding_benchmark/verify_io_uniformity.py \
    --cells-dir "$CELLS" --profile v3 --limit 64 \
    --output "$CELLS/io_uniformity_v3_e2e.json"
IO3_RC=$?

step "5/6  scoring  v2 + v3"
run "$PY" scripts/embedding_benchmark/score_benchmark.py --cells-dir "$CELLS" --profile v2 \
    --output "$CELLS/benchmark_scores_v2_e2e.json"
S2_RC=$?
run "$PY" scripts/embedding_benchmark/score_benchmark.py --cells-dir "$CELLS" --profile v3 \
    --output "$CELLS/benchmark_scores_v3_e2e.json"
S3_RC=$?

step "6/6  stability  v3 (4 folds over wells)"
run "$PY" scripts/embedding_benchmark/stability_check.py --cells-dir "$CELLS" \
    --profile v3 --folds 4 --output "$CELLS/stability_v3_e2e.json"
ST_RC=$?

echo "" | tee -a "$LOG"
echo "########## SUMMARY ##########" | tee -a "$LOG"
printf '  embed_dataset       v2 : %s\n' "$V2_RC" | tee -a "$LOG"
printf '  build_v3_surrogates v3 : %s\n' "$V3_RC" | tee -a "$LOG"
printf '  io_uniformity       v2 : %s\n' "$IO2_RC" | tee -a "$LOG"
printf '  io_uniformity       v3 : %s\n' "$IO3_RC" | tee -a "$LOG"
printf '  score               v2 : %s\n' "$S2_RC" | tee -a "$LOG"
printf '  score               v3 : %s\n' "$S3_RC" | tee -a "$LOG"
printf '  stability           v3 : %s\n' "$ST_RC" | tee -a "$LOG"
echo "  log: $LOG" | tee -a "$LOG"