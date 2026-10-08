#!/bin/bash
#SBATCH --job-name=g1-32b-candidate-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/g1-32b-candidate-cpu-%j.log
set -euo pipefail
case "$(hostname)" in h10011|h100-11) ;; *) exit 70;; esac
: "${SLURM_JOB_ID:?Native CPU Slurm only}"
ROOT=/dataset/shaoshitong/search-r1-rl/topd
CANDIDATE="$ROOT/tools/g1-32b-fallback-candidate"
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/test-tools:$CANDIDATE"
cd "$CANDIDATE"
"$ROOT/env/bin/python" - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path.cwd();manifest=json.loads((root/'reproduction/conditional-teacher100/files.json').read_text())
for name,rec in manifest['files'].items():
 p=root/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=rec['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:raise SystemExit('candidate bytes mismatch: '+name)
print(json.dumps({'scope':'G1_CONDITIONAL_CPU_CONTRACTS_NO_TRAINING','source_base':manifest['source_base'],'payload_files':len(manifest['files'])}))
PY
"$ROOT/env/bin/python" -m pytest -q --tb=short -p no:cacheprovider reproduction/conditional-teacher100/test_conditional_candidate.py
