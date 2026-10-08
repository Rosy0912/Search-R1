set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
cat > "$A/tools/install-env.sh" <<'ENV'
set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
export UV_CACHE_DIR="$A/cache/uv"
export UV_HTTP_TIMEOUT=60
export MAX_JOBS=16
UV=$A/bootstrap/bin/uv
"$UV" pip install --python "$A/env/bin/python" 'vllm==0.11.0' 'transformers==4.57.6' 'tokenizers==0.22.2' 'numpy<2' 'tensordict>=0.8,<=0.10,!=0.9' 'trl==0.9.6' -r "$A/repo/verl/requirements.txt" codetiming mathruler pylatexenc qwen_vl_utils nvtx ujson openai omegaconf pytrec-eval
"$UV" pip install --python "$A/env/bin/python" --no-deps -e "$A/repo/verl"
"$A/env/bin/python" -c 'import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ["torch","vllm","transformers","ray","tensordict","verl"]}))'
ENV
python3 - "$A" <<'PY'
import subprocess,sys,json,time,os
from pathlib import Path
r=Path(sys.argv[1]);state=r/'install-process.json'
if state.exists():
 d=json.loads(state.read_text())
 try:os.kill(d['pid'],0);print('EXISTING',d);raise SystemExit
 except ProcessLookupError:pass
log=(r/'logs/install-env.log').open('a')
p=subprocess.Popen(['bash',str(r/'tools/install-env.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d={'pid':p.pid,'started_at':time.time(),'log':str(r/'logs/install-env.log')};state.write_text(json.dumps(d));print(json.dumps(d))
PY
