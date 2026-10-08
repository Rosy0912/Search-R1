set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
cat > "$A/tools/build-flash.sh" <<'BUILD'
set -euo pipefail
A=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
python3 - "$A" <<'PY'
import json,sys,subprocess,hashlib
from pathlib import Path
r=Path(sys.argv[1]);packages=json.loads((r/'tools/cuda-debs.json').read_text());selected={}
for x in packages:selected[x['Package']]=x
out=r/'cuda-extract';out.mkdir(exist_ok=True)
for name,d in selected.items():
 p=r/'cache'/Path(d['Filename']).name
 if not p.exists():subprocess.run(['curl','-fsSL','--connect-timeout','15','--max-time','600','https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/'+d['Filename'].removeprefix('./'),'-o',str(p)],check=True)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==d['SHA256'],name
 subprocess.run(['dpkg-deb','-x',str(p),str(out)],check=True);print('VERIFIED_CUDA_DEB',name,d['Version'],flush=True)
PY
export CUDA_HOME="$A/cuda-extract/usr/local/cuda-12.8"
export PATH="$CUDA_HOME/bin:$A/env/bin:$PATH"
export UV_CACHE_DIR="$A/cache/uv"
export MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=''
export TORCH_CUDA_ARCH_LIST=9.0
export FLASH_ATTN_CUDA_ARCHS=90
"$A/bootstrap/bin/uv" pip install --python "$A/env/bin/python" packaging ninja setuptools wheel
"$A/bootstrap/bin/uv" pip install --python "$A/env/bin/python" --no-build-isolation flash_attn==2.8.1
"$A/env/bin/python" -c 'import flash_attn;print("FLASH_ATTN_INSTALLED",flash_attn.__version__)'
BUILD
python3 - "$A" <<'PY'
from pathlib import Path
import subprocess,sys,json,time
r=Path(sys.argv[1]);p=subprocess.Popen(['bash',str(r/'tools/build-flash.sh')],stdout=(r/'logs/flash-build.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True);d={'pid':p.pid,'started_at':time.time()};(r/'flash-process.json').write_text(json.dumps(d));print(json.dumps(d))
PY
