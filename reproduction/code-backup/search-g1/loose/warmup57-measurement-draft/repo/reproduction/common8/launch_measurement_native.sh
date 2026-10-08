#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?explicit new measurement namespace required}"
PARENT="${SEARCH_G1_PRODUCTION_BASE:?actual distinct fresh0 formal intent required}"
METHOD="${SEARCH_G1_MEASUREMENT_METHOD:?prospective new warmup57 method file required}"
COMMIT="${SEARCH_G1_SOURCE_COMMIT:?committed measured source required}"
: "${SLURM_JOB_ID:?actual native allocation required}"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
export SEARCH_G1_PROFILE=common8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" RAY_TMPDIR="$BASE/ray" XDG_CACHE_HOME="$BASE/cache" HF_HOME="$BASE/cache/hf"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 NCCL_NVLS_ENABLE=0
export SEARCH_G1_MEASUREMENT_METHOD="$METHOD" SEARCH_G1_PRODUCTION_BASE="$PARENT"
unset PYTHONHOME RAY_ADDRESS
mkdir -p "$BASE/logs" "$BASE/tmp" "$BASE/ray" "$BASE/artifacts" "$BASE/cache/triton" "$BASE/cache/inductor" "$BASE/cache/hf"
cd "$BASE/repo"
# No model/data reads before the actual full native allocation is confirmed.
python - <<'PY'
import json,os,socket,subprocess
from pathlib import Path
from reproduction.common8.native_allocation import fields
from reproduction.common8.stage_retrieval import memory_bytes
from reproduction.common8.prepare_run import prepare,verify_model
from reproduction.common8.source_files import runtime_files
from reproduction.common8.verify_data import verify,digest
from reproduction.common8.formal_contract import assert_formal_run
from reproduction.common8.warmup_measurement_plan import scientific_identity,verify_method
from reproduction.common8.freeze_calibration_plan import _unstarted
job=fields(subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True))
assert job['JobState']=='RUNNING' and job['NodeList']==socket.gethostname().split('.')[0]=='h100-8'
assert memory_bytes(job['TRES'])>=512*2**30 and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
base=Path(os.environ['SEARCH_G1_BASE']);parent=Path(os.environ['SEARCH_G1_PRODUCTION_BASE'])
method=Path(os.environ['SEARCH_G1_MEASUREMENT_METHOD']);commit=os.environ['SEARCH_G1_SOURCE_COMMIT']
assert base.is_absolute() and base.resolve()==base and parent.is_absolute() and parent.resolve()==parent and base!=parent
assert str(base).startswith('/mnt/data/shaoshitong/') and str(parent).startswith('/mnt/data/shaoshitong/')
assert method.is_absolute() and method.resolve()==method and method.is_file() and not method.is_symlink()
assert not (base/'logs/run-manifest.json').exists(), 'Never overwrite an attempted measurement namespace'
model='bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725'
official='preparation/official-model-manifest-original.json'
parent_manifest=parent/'logs/run-manifest.json'
if parent_manifest.exists():
    assert parent_manifest.is_file() and not parent_manifest.is_symlink()
    parent_run=json.loads(parent_manifest.read_text());assert_formal_run(parent_run);_unstarted(parent,parent_run)
    assert parent_run['source_commit']==commit
    assert parent_run['initial_model']==str((parent/model).resolve())
    assert verify_model(parent/model,parent/official)==parent_run['initial_model_sha256']
    verify(parent/'data')
    assert all(digest(parent/'data'/name)==value for name,value in parent_run['input_sha256'].items())
    actual_source={str(path.relative_to(parent/'repo')):digest(path) for path in runtime_files(parent/'repo')}
    assert actual_source==parent_run['source_sha256'], 'Actual fresh0 parent source changed'
    scientific_identity(parent_run)
else:
    parent_run=prepare(parent,parent/model,parent/official,commit)
# prepare() checks prospective method, complete science and same source/data/model
# against this real archived parent, then writes the separate child manifest once.
child=prepare(base,base/model,base/official,commit,execution_role='warmup57_measurement',
              measurement_parent=parent_manifest,measurement_method=method)
assert child['measurement_parent_run_id']==parent_run['run_id']
PY
exec python -m reproduction.common8.run_pipeline --base "$BASE" --export-port "${SEARCH_G1_EXPORT_PORT:-19874}"
