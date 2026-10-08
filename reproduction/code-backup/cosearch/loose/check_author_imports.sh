set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$A/repo:$A/repo/verl"
"$A/env/bin/python" - <<'PY'
import importlib,json
for n in ['torch','vllm','flash_attn','main_co_search_ppo','verl.tools.co_search_tool']:
 try:importlib.import_module(n);print(json.dumps({'module':n,'ok':True}),flush=True)
 except Exception as e:print(json.dumps({'module':n,'ok':False,'error':repr(e)}),flush=True)
PY
