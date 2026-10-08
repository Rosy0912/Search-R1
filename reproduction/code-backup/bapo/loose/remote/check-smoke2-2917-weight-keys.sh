set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
CUDA_VISIBLE_DEVICES='' "$R/env/bin/python" - <<'PY'
import json,pathlib,hashlib
from safetensors import safe_open
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
base=r/'assets-root/assets/qwen2.5-3b'
ck=r/'runs/bapo-smoke2-2917/output/checkpoints/bapo-smoke2-2917/actor/global_step_2'
def scan(root):
 d={}
 for p in sorted(root.glob('*.safetensors')):
  with safe_open(str(p),framework='pt',device='cpu')as f:
   for k in f.keys():d[k]={'file':p.name,'shape':f.get_slice(k).get_shape(),'dtype':f.get_slice(k).get_dtype()}
 return d
b,c=scan(base),scan(ck)
print(json.dumps(dict(base_path=str(base),checkpoint_path=str(ck),base_only={k:b[k]for k in b.keys()-c.keys()},checkpoint_only={k:c[k]for k in c.keys()-b.keys()},base_config=json.load(open(base/'config.json')),checkpoint_config=json.load(open(ck/'config.json')),base_tensor_count=len(b),checkpoint_tensor_count=len(c)),indent=2))
PY
