#!/bin/bash
# CPU-only, separate compatibility build of unchanged pytrec_eval 0.5.
set -euo pipefail
BASE=/data/user/jxu957/experiments/cosearch-continue300-x203-20261008
R="$BASE/runtime/pytrec-eval-05-hpc-build"
exec 9>"$R/build.lock"
flock -n 9
set +u
source /etc/profile
set -u
module load gcc/11.5
PY="$BASE/source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
export PYTHONPATH="$BASE/source/author-native/env/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES='' CC=gcc CXX=g++
cd "$R/pytrec_eval-0.5"
"$PY" setup.py build --build-base "$R/build"
export PYTHONPATH="$(find "$R/build" -maxdepth 1 -type d -name 'lib*' -print -quit):$PYTHONPATH"
"$PY" - "$R" <<'PY'
import pytrec_eval,pytrec_eval_ext,json,hashlib,time,sys,math
from pathlib import Path
q={'q':{'a':1,'b':0,'c':1}}
run={'q':{'b':3.0,'a':2.0,'c':1.0}}
result=pytrec_eval.RelevanceEvaluator(q,{'recip_rank','recall.3','ndcg_cut.3'}).evaluate(run)['q']
assert result['recip_rank']==0.5 and result['recall_3']==1.0
expected=(1/math.log2(3)+1/math.log2(4))/(1+1/math.log2(3))
assert abs(result['ndcg_cut_3']-expected)<1e-12
p=Path(pytrec_eval_ext.__file__)
Path(sys.argv[1],'build-complete.json').write_text(json.dumps({'status':'PYTREC_05_NATIVE_IMPORT_METRICS_PASS','epoch':time.time(),'extension':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metrics':result,'gpu_requested':False}))
print('PYTREC_05_NATIVE_IMPORT_METRICS_PASS',result,flush=True)
PY
