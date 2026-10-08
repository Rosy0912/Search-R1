#!/bin/bash
set -euo pipefail
root=/dataset/shaoshitong/search-r1-rl/bapo
export PATH="$root/env/bin:$root/tools:$PATH" SEARCH_R1_ASSETS="$root/assets-root" SEARCH_R1_REPO="$root/repo"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$root/logs" "$root/assets-root/logs"
cd "$root/repo"
test -f "$root/logs/environment.ready"
python "$root/repo/reproduction/h20/verify_assets.py" > "$root/logs/asset-verification.json"
test -f "$root/logs/cpu-gate.ready"
test "$(cat "$root/logs/cpu-gate-source.txt")" = "$(git rev-parse HEAD)"
python "$root/repo/reproduction/h20/prepare_paper_data.py" > "$root/logs/paper-data-preparation.json"
python - <<'PYCODE'
import json,os
from hydra import initialize_config_dir,compose
from search_r1.baselines.holdout import prepare_common_holdout
root='/dataset/shaoshitong/search-r1-rl/bapo'
os.environ['SEARCH_R1_BASELINE_OUTPUT']=root+'/prepared'
with initialize_config_dir(config_dir=root+'/repo/verl/trainer/config',version_base=None):
 config=compose(config_name='ppo_baselines',overrides=['baselines.method=bapo','actor_rollout_ref.actor.ppo_micro_batch_size=8'])
manifest=prepare_common_holdout(config)
open(root+'/logs/common-split-manifest.json','w').write(json.dumps(manifest,indent=2))
PYCODE
wiki="$root/assets-root/assets/wiki"
if [ ! -f "$wiki/e5_Flat.index" ]; then
 cat "$wiki/part_aa" "$wiki/part_ab" > "$wiki/e5_Flat.index.partial"
 expected=$(stat -c %s "$wiki/part_aa"); second=$(stat -c %s "$wiki/part_ab")
 test "$(stat -c %s "$wiki/e5_Flat.index.partial")" = "$((expected+second))"
 mv "$wiki/e5_Flat.index.partial" "$wiki/e5_Flat.index"
fi
if [ ! -f "$wiki/wiki-18.jsonl" ]; then
 gzip -dc "$wiki/wiki-18.jsonl.gz" > "$wiki/wiki-18.jsonl.partial"
 mv "$wiki/wiki-18.jsonl.partial" "$wiki/wiki-18.jsonl"
fi
sha256sum "$wiki/e5_Flat.index" "$wiki/wiki-18.jsonl" > "$root/logs/derived-assets.sha256"
touch "$root/logs/cpu-assets-prepared.ready"
