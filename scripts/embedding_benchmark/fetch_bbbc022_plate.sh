#!/usr/bin/env bash
# Fetch the five channel-sharded archives of BBBC022 plate 20585 in full.
#
# The copies in E:/bbbc022 are truncated (0.57-0.78 GB each against ~1.38 GB), so
# only 72 fields end up with all five channels present, which caps the built
# dataset at ~3.9k cells.  These go to a separate directory so the truncated
# originals stay untouched until a fresh build has been verified.
set -u

DEST="E:/bbbc022/_full"
BASE="https://data.broadinstitute.org/bbbc/BBBC022"
mkdir -p "$DEST"

for n in 1 2 3 4 5; do
    name="BBBC022_v1_images_20585w${n}.zip"
    target="$DEST/$name"
    echo "=== $name ==="
    # -C - resumes a partial file, so an interrupted run is not wasted.
    curl -L -C - --retry 5 --retry-delay 5 --max-time 21600 \
         -o "$target" "$BASE/$name"
    rc=$?
    size=$(stat -c%s "$target" 2>/dev/null || echo 0)
    echo "  exit=$rc  size=$size"
done

echo "=== verify ==="
for n in 1 2 3 4 5; do
    name="BBBC022_v1_images_20585w${n}.zip"
    python - "$DEST/$name" <<'PY'
import sys, zipfile
path = sys.argv[1]
try:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        print(f'  {path}: OK entries={len(zf.namelist())} first_bad={bad}')
except Exception as exc:
    print(f'  {path}: BAD {type(exc).__name__}: {exc}')
PY
done