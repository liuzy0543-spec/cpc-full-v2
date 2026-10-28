#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$DIST_DIR"

export ROOT DIST_DIR
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ['ROOT'])
dist = Path(os.environ['DIST_DIR'])
pyproject = root / 'pyproject.toml'

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

with pyproject.open('rb') as fh:
    project = tomllib.load(fh)
version = project['project']['version']
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
bundle_name = f'cellpainting-claw-{version}-release-{stamp}'
staging = dist / bundle_name
archive = dist / f'{bundle_name}.tar.gz'
checksum = dist / f'{bundle_name}.sha256'
manifest = dist / f'{bundle_name}.manifest.json'

if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

include_paths = [
    'README.md',
    'CHANGELOG.md',
    'RELEASE_NOTES.md',
    'pyproject.toml',
    'docs',
    'configs',
    'environment',
    'src',
    'tests',
    'scripts',
    'release',
    'integrations/openclaw',
]

exclude_dir_names = {
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    'state',
}
exclude_file_suffixes = {'.pyc', '.pyo'}
exclude_file_names = {
    'provider.env',
    '.env',
}

copied = []
for rel in include_paths:
    src = root / rel
    dst = staging / rel
    if not src.exists():
        continue
    if src.is_dir():
        for path in src.rglob('*'):
            rel_path = path.relative_to(root)
            if any(part in exclude_dir_names for part in rel_path.parts):
                continue
            if path.is_dir():
                continue
            if path.name in exclude_file_names:
                continue
            if any(str(path).endswith(suffix) for suffix in exclude_file_suffixes):
                continue
            dst_path = staging / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst_path)
            copied.append(str(rel_path))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

release_readme = staging / 'release' / 'README.md'
release_readme.parent.mkdir(parents=True, exist_ok=True)
if not release_readme.exists():
    release_readme.write_text(
        '# Release Staging Area\n\nGenerated release bundles are created from the source tree by `scripts/build_release_bundle.sh`.\n',
        encoding='utf-8',
    )

if archive.exists():
    archive.unlink()
with tarfile.open(archive, 'w:gz') as tf:
    tf.add(staging, arcname=bundle_name)

sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
checksum.write_text(f'{sha256}  {archive.name}\n', encoding='utf-8')
manifest.write_text(json.dumps({
    'bundle_name': bundle_name,
    'version': version,
    'created_at_utc': stamp,
    'archive': str(archive),
    'checksum_file': str(checksum),
    'staging_dir': str(staging),
    'file_count': len(copied),
    'included_roots': include_paths,
}, indent=2) + '\n', encoding='utf-8')

print(json.dumps({
    'bundle_name': bundle_name,
    'archive': str(archive),
    'checksum_file': str(checksum),
    'manifest': str(manifest),
    'staging_dir': str(staging),
    'file_count': len(copied),
}, indent=2))
PY
