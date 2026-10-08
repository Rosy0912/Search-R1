#!/bin/bash
#SBATCH --job-name=g1-initial-static-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-initial-static-cpu-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$TASK_ROOT/test-tools:$TASK_ROOT/tools/g1-production-pipeline-v3-r2:$TASK_ROOT/tools/g1-native-recipe-v2"
RUN="$TASK_ROOT/logs/g1-initial-static-cpu-$SLURM_JOB_ID"
mkdir -m 700 "$RUN"
test "$#" -eq 3
"$TASK_ROOT/env/bin/python" - "$TASK_ROOT" "$1" "$2" "$3" <<'PYVERIFY'
import sys,pathlib,json,hashlib
root=pathlib.Path(sys.argv[1])
for name,expected in zip(('g1-staged-calibration-producer-v3','g1-production-pipeline-v3-r2','g1-native-recipe-v2'),sys.argv[2:]):
 directory=root/'tools'/name;manifest=directory/'files.json'
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected
 records=json.loads(manifest.read_text())['files']
 actual=set()
 for p in directory.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():actual.add(str(p.relative_to(directory)))
 assert actual==set(records)|{'files.json'}
 for n,r in records.items():
  p=directory/n;assert p.stat().st_size==r['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
print('ALL_THREE_EXACT_FROZEN_NATIVE_CPU_PACKAGES')
PYVERIFY
"$TASK_ROOT/env/bin/python" -m pytest -q "$TASK_ROOT/tools/g1-production-pipeline-v3-r2/test_pipeline.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r2/test_guardian.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r2/test_retrieval_gate.py" "$TASK_ROOT/tools/g1-staged-calibration-producer-v3/test_producer.py" "$TASK_ROOT/tools/g1-staged-calibration-producer-v3/test_phase_control.py" "$TASK_ROOT/tools/g1-native-recipe-v2/test_assemble.py" --basetemp="$RUN/pytest-temp" -o cache_dir="$RUN/pytest-cache"
"$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/g1-native-recipe-v2/assemble.py" --output-directory "$RUN/recipe"
