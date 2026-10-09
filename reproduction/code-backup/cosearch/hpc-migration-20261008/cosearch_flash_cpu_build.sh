#!/bin/bash
set -euo pipefail
BASE=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
R="$BASE/runtime/flash-attn-281-hpc-build"
mkdir -p "$R"
exec 9>"$R/build.lock"
flock -n 9
set +u
source /etc/profile
set -u
module load gcc/11.5 cuda/12.8
PY="$BASE/source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
export PYTHONPATH="$BASE/source/author-native/env/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES=''
export CUDA_HOME=/usr/local/cuda-12.8
export CC=gcc CXX=g++ MAX_JOBS=2 NVCC_THREADS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0
export TMPDIR="$R/tmp" XDG_CACHE_HOME="$R/cache" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$R/bin"
"$PY" - "$BASE" "$R" <<'PY'
from pathlib import Path
import shutil,sys,json,time,torch
b,r=map(Path,sys.argv[1:]);assert not torch.cuda.is_initialized()
s=b/'runtime/flash-attn-281-source';d=r/'src'
if not d.exists():shutil.copytree(s,d)
(r/'build-start.json').write_text(json.dumps({'epoch':time.time(),'cuda_initialized':False,'gpu_requested':False,'source':str(s),'build_copy':str(d)}))
PY
printf '#!/bin/bash\nexec "%s" -m ninja "$@"\n' "$PY" > "$R/bin/ninja"
chmod 700 "$R/bin/ninja"
export PATH="$R/bin:$PATH"
cd "$R/src"
"$PY" setup.py build_ext --build-lib "$R/lib" --build-temp "$R/objects"
export PYTHONPATH="$R/lib:$R/src:$PYTHONPATH"
"$PY" - "$R" <<'PY'
import flash_attn,torch,json,time,sys
from pathlib import Path
assert flash_attn.__version__=='2.8.1' and not torch.cuda.is_initialized()
Path(sys.argv[1],'build-complete.json').write_text(json.dumps({'epoch':time.time(),'status':'FLASH_ATTN_HPC_BUILD_IMPORT_PASS','version':flash_attn.__version__,'cuda_initialized':False,'gpu_parity_pending':True}))
PY
