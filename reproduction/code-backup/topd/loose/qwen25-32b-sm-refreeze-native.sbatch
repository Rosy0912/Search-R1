#!/bin/bash
#SBATCH --job-name=topd32b-sm-refreeze-cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/qwen25-32b-sm-refreeze-native-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$AGENT_ROOT/test-tools:$AGENT_ROOT/repo"
cd "$AGENT_ROOT/tools/qwen25-32b-eval64"
"$AGENT_ROOT/env/bin/python" - <<'CHECK'
import pathlib,hashlib,tarfile
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools');archive=root/'qwen25-32b-eval-sm-refreeze-v1.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='635c9032aecc6c032035c2228ed83f2c38d3c1b7125bdab01f5c811d03777c48'
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert m.isfile() and '..' not in pathlib.Path(m.name).parts
  b=(root/m.name).read_bytes();assert b==t.extractfile(m).read();print(hashlib.sha256(b).hexdigest(),m.name)
CHECK
"$AGENT_ROOT/env/bin/python" -m pytest test_eval64.py -q
"$AGENT_ROOT/env/bin/python" -m pytest test_build_binding.py -q
