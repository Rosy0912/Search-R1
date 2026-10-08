#!/bin/bash
#SBATCH --job-name=g1-known-tools-cpu
#SBATCH --partition=gpu
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-known-tools-cpu-%j.log
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
: "${KNOWN_TOOLS_ROOT:?actual isolated source directory}"
: "${KNOWN_TOOLS_MANIFEST_SHA256:?actual source manifest SHA}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TASK_ROOT/test-tools:$KNOWN_TOOLS_ROOT"
OUT="$TASK_ROOT/logs/g1-known-tools-native-$SLURM_JOB_ID"
mkdir -p "$OUT"
"$TASK_ROOT/env/bin/python" - "$KNOWN_TOOLS_ROOT" "$KNOWN_TOOLS_MANIFEST_SHA256" <<'PY'
import pathlib,sys,json,hashlib
root=pathlib.Path(sys.argv[1]);manifest=root/'files.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==sys.argv[2]
records=json.loads(manifest.read_text())['files'];actual=set()
for p in root.rglob('*'):
 assert not p.is_symlink()
 if p.is_file():actual.add(str(p.relative_to(root)))
assert actual==set(records)|{'files.json'}
for name,r in records.items():
 p=root/name;assert p.stat().st_size==r['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
print('EXACT_SOURCE_MANIFEST',manifest,sys.argv[2])
PY
"$TASK_ROOT/env/bin/python" -m pytest -q "$KNOWN_TOOLS_ROOT/test_known_record.py" "$KNOWN_TOOLS_ROOT/test_native_run.py" --basetemp "$OUT/pytest-temp" -o "cache_dir=$OUT/pytest-cache"
