#!/bin/bash
#SBATCH --job-name=topd32b-native-download
#SBATCH --partition=gpu
#SBATCH --nodelist=h100-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/dataset/shaoshitong/search-r1-rl/topd/logs/topd32b-native-download-%j.log
set -euo pipefail
export AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/topd
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$AGENT_ROOT/test-tools"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$AGENT_ROOT/tools/topd-official-h16proxy-download-v2"
"$AGENT_ROOT/env/bin/python" - <<'PY'
import hashlib, json
from pathlib import Path
p=Path('files.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='be40e8e0c30d1c2d1c133c8e4053729be59f77589ad67c2c8fe6dd3640ae39a6'
for name, rec in json.loads(p.read_text()).items():
    f=Path(name)
    assert not f.is_symlink() and f.is_file()
    b=f.read_bytes()
    assert len(b)==rec['size'] and hashlib.sha256(b).hexdigest()==rec['sha256'], name
print('EXACT_FROZEN_TRANSPORT_FILES_VERIFIED_NO_GPU', flush=True)
PY
"$AGENT_ROOT/env/bin/python" -m pytest -q test_download.py
"$AGENT_ROOT/env/bin/python" -u download.py \
  --metadata "$AGENT_ROOT/logs/qwen2.5-32b-official-metadata.json" \
  --handoff "$AGENT_ROOT/logs/qwen25-32b-v6-to-native-handoff.json" \
  --state "$AGENT_ROOT/logs/qwen25-32b-download-h16proxy-state.json"
