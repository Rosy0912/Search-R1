set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097
export PYTHONDONTWRITEBYTECODE=1
export PATH="$R/env/bin:$PATH" SEARCH_R1_ASSETS="$R/assets-root" SEARCH_R1_BASELINE_OUTPUT="$R/prepared" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$R/repo"
python - <<'PY' > "$R/logs/stage64-b097-proof.json"
from pathlib import Path
import json,hashlib
import pandas as pd
from hydra import compose,initialize_config_dir
from search_r1.baselines.holdout import prepare_common_holdout,question_key,file_hash
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')

with initialize_config_dir(config_dir=str(root/'repo/verl/trainer/config'),version_base=None):
 c=compose(config_name='ppo_baselines',overrides=['baselines.method=bapo','actor_rollout_ref.actor.ppo_micro_batch_size=64'])
new=prepare_common_holdout(c); assert new['content_identity']['sha256']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
(root/'logs/common-split-manifest.json').write_text(json.dumps(new,indent=2)+'\n')
p=Path(c.baselines.bapo.stage_validation_files)
h=pd.read_parquet(p);train=pd.read_parquet(c.data.train_files)
keys=lambda f:set(f['prompt'].map(question_key))
assert len(h)==len(keys(h))==64 and not keys(h)&keys(train)
for f in [c.data.val_files,c.data.final_val_files]:
 if f:assert not keys(h)&keys(pd.read_parquet(f))
record=json.loads(p.with_name('manifest.json').read_text())
assert len(record['question_identities'])==64
assert file_hash(p)=='82a5ed0d04b261031d4ee03532f0c35fc688dd2fb11c7cdaf1cc87ad0e46c34e'
assert len(train)==166222 and file_hash(Path(c.data.train_files))=='e5e774fd1f5777a8be91103facaccb711cfd9593d7563e9d5afe04e6b1a52bea'
print('BAPO_STAGE64_DERIVATION_PROOF_JSON='+json.dumps(dict(stage_file=str(p),stage_sha256=file_hash(p),manifest=record,train_rows=len(train),train_sha256=file_hash(Path(c.data.train_files)),parent_content_identity=new['content_identity']['sha256'],disjoint_training_periodic_final=True),sort_keys=True))
PY
tail -n 1 "$R/logs/stage64-b097-proof.json"
