#!/bin/bash
set -euo pipefail
export AGENT_ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
ROOT="$AGENT_ROOT"
test -n "$SLURM_JOB_ID"
test "$SLURM_JOB_NODELIST" = h100-8
test -f "$ROOT/logs/environment.ready"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 SEARCH_R1_ASSETS="$ROOT/assets-root"
export TMPDIR="$ROOT/tmp" HF_DATASETS_CACHE="$ROOT/cache/datasets"
export PYTHONPATH="$ROOT/repo:$ROOT/tools/test-lib"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cd "$ROOT/repo"
"$ROOT/env/bin/python" "$ROOT/tools/cluster_verify.py"
"$ROOT/env/bin/python" reproduction/h20/check_environment.py
"$ROOT/env/bin/python" reproduction/h20/prepare_paper_data.py
SEARCH_R1_BASELINE_OUTPUT="$ROOT" "$ROOT/env/bin/python" - <<'PY'
import os,json,hashlib
from pathlib import Path
from hydra import initialize_config_dir,compose
from search_r1.baselines.holdout import prepare_common_holdout
root=Path(os.environ['AGENT_ROOT'])
with initialize_config_dir(config_dir=str(root/'repo/verl/trainer/config'),version_base=None):
 config=compose(config_name='ppo_baselines',overrides=['baselines.method=grpo','baselines.holdout.output_dir='+str(root/'common-data')])
record=prepare_common_holdout(config)
assert record['holdout_rows']==3393,record
assert record['content_identity']['sha256']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524',record
(root/'logs/prepared-data-paths.json').write_text(json.dumps({'train':config.data.train_files, 'manifest':str(Path(config.data.train_files).parent/'manifest.json'), 'identity':record['content_identity']['sha256'], 'train_sha256':record['sha256']['train.parquet']}))
print('COMMON_SPLIT_IDENTITY_MATCHES_ALL_AGENTS',record['content_identity']['sha256'],flush=True)
PY
"$ROOT/env/bin/python" -m pytest tests/baselines tests/reproduction -q
"$ROOT/env/bin/python" reproduction/h20/check_environment.py > "$ROOT/logs/environment-post-cpu.json"
echo NATIVE_CPU_AND_ASSET_GATE_PASS
printf '0\n' > "$ROOT/logs/native-cpu-$SLURM_JOB_ID.exit"
