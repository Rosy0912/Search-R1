#!/bin/bash
#SBATCH --job-name=g1-initial-r4-rebind-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-initial-r4-rebind-cpu-%j.log
set -euo pipefail
TASK_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$TASK_ROOT/test-tools:$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4:$TASK_ROOT/tools/g1-native-recipe-v4-r4"
test "$#" -eq 5
RUN="$TASK_ROOT/logs/g1-initial-r4-rebind-cpu-$SLURM_JOB_ID"
mkdir -m 700 "$RUN"
"$TASK_ROOT/env/bin/python" - "$TASK_ROOT" "$1" "$2" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
for name,expected in zip(('g1-production-pipeline-v3-r4-r4','g1-native-recipe-v4-r4'),sys.argv[2:]):
 d=root/'tools'/name;p=d/'files.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 files=json.loads(p.read_text())['files'];actual=set()
 for f in d.rglob('*'):
  assert not f.is_symlink()
  if f.is_file():actual.add(str(f.relative_to(d)))
 assert actual==set(files)|{'files.json'}
 for n,r in files.items():
  f=d/n;assert f.stat().st_size==r['size'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
print('EXACT_R4_REBIND_FROZEN_SOURCE_BYTES')
VERIFY
"$TASK_ROOT/env/bin/python" -m pytest -q "$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4/test_health.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4/test_rpc.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4/test_retrieval_gate.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4/test_retrieval_http_audit.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4/test_guardian.py" "$TASK_ROOT/tools/g1-production-pipeline-v3-r4-r4/test_controlled_exit.py" "$TASK_ROOT/tools/g1-native-recipe-v4-r4/test_rebind_initial.py" --basetemp="$RUN/pytest-temp" -o cache_dir="$RUN/pytest-cache"
"$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/g1-native-recipe-v4-r4/rebind_initial.py" --original-config "$3" --assembly-receipt "$4" --output "$5" --pipeline-manifest-sha256 "$1"
