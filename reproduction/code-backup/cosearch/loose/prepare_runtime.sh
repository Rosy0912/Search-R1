#!/usr/bin/env bash
set -euo pipefail
AGENT_ROOT=/dataset/shaoshitong/search-r1-rl/cosearch
export PATH="$AGENT_ROOT/env/bin:$PATH"
export SEARCH_R1_ASSETS="$AGENT_ROOT/assets-root" SEARCH_R1_REPO="$AGENT_ROOT/repo"
export SEARCH_R1_BASELINE_OUTPUT="$AGENT_ROOT/runs/preparation/output"
export TMPDIR="$AGENT_ROOT/tmp" HF_HOME="$AGENT_ROOT/cache/hf"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$AGENT_ROOT/repo"
test -f "$AGENT_ROOT/logs/environment.ready"
python reproduction/h20/verify_assets.py > "$AGENT_ROOT/logs/asset-verification.json"
echo VERIFIED_ALL_IMMUTABLE_ASSETS
if [[ ! -f "$SEARCH_R1_ASSETS/data/paper-grpo/train.parquet" ]]; then
  python reproduction/h20/prepare_paper_data.py > "$AGENT_ROOT/logs/prepare-data.log" 2>&1
fi
echo PREPARED_ORIGINAL_DATA
if [[ ! -f "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" ]]; then
  cat "$SEARCH_R1_ASSETS/assets/wiki/part_aa" "$SEARCH_R1_ASSETS/assets/wiki/part_ab" \
    > "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index.agent6.partial"
  mv "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index.agent6.partial" "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index"
fi
echo CONCATENATED_VERIFIED_INDEX_SHARDS
if [[ ! -f "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" ]]; then
  gzip -dc "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl.gz" \
    > "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl.agent6.partial"
  mv "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl.agent6.partial" "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl"
fi
echo DECOMPRESSED_VERIFIED_CORPUS
python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from search_r1.baselines.holdout import prepare_common_holdout
root = Path('/dataset/shaoshitong/search-r1-rl/cosearch')
with initialize_config_dir(config_dir=str(root/'repo/verl/trainer/config'), version_base=None):
    config = compose(config_name='ppo_baselines', overrides=['baselines.method=cosearch',
        'actor_rollout_ref.actor.ppo_micro_batch_size=8',
        'baselines.ranker.model_path='+str(root/'assets-root/assets/qwen2.5-3b')])
OmegaConf.resolve(config)
manifest = prepare_common_holdout(config)
(root/'logs/split-content-identity.json').write_text(json.dumps(manifest, indent=2)+'\n')
(root/'logs/resolved-prepared.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
print('SPLIT_IDENTITY', manifest['content_identity']['sha256'], flush=True)
print('PREPARATION_COMMIT', subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
PY
touch "$AGENT_ROOT/logs/runtime-assets.ready"
