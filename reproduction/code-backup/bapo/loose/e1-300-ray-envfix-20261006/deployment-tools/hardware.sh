#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006
RUN=$1
python - <<'PYTD' > "$RUN/tensordict-device-api.json"
import inspect,hashlib,json,importlib.metadata,os,torch
from tensordict import TensorDict,TensorDictBase
from pathlib import Path
names={'TensorDict.cuda':TensorDict.cuda,'TensorDictBase.cuda':TensorDictBase.cuda}
records={}
for name,fn in names.items():
 p=Path(inspect.getsourcefile(fn)).resolve()
 records[name]={'signature':str(inspect.signature(fn)),'source':inspect.getsource(fn),'file':str(p),'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
actor=Path(os.environ['SEARCH_R1_REPO'])/'verl/workers/actor/dp_actor.py'
print(json.dumps({'status':'ACTUAL_TARGET_TENSORDICT_API_SOURCE_ONLY_NOT_GPU_ACCEPTANCE','source_commit':os.environ['SOURCE'],'torch_version':torch.__version__,'tensordict_version':importlib.metadata.version('tensordict'),'actor_path':str(actor),'actor_sha256':hashlib.sha256(actor.read_bytes()).hexdigest(),'cuda_methods':records},indent=2))
PYTD
port=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/repo/reproduction/h20/cuda_preflight.py"
export E1_EXPECTED_SOURCE="$SOURCE" E1_SOURCE_COMMIT="$SOURCE"
export E1_COUNT_FIXTURE_OUTPUT="$RUN/native-count"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/native-count-fixture.py"
export E1_STATE_FIXTURE_OUTPUT="$RUN/native-state"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/native-state-fixture.py"
export E1_ACTOR_FIXTURE_OUTPUT="$RUN/native-actor"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/native-actor-fixture.py" --repo "$ROOT/repo" --output "$RUN/native-actor"
export NUMERIC_FIXTURE_OUTPUT="$RUN/numeric-fixture"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/numerics-gpu-fixture.py"
export E1_STABILITY_FIXTURE_OUTPUT="$RUN/native-stability"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/native-stability-fixture.py"
export E1_FLASH_OUTPUT="$RUN/flash-native"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/flash_preflight.py"
first_gpu=${CUDA_VISIBLE_DEVICES%%,*}
CUDA_VISIBLE_DEVICES="$first_gpu" python - <<'PY'
import os
from vllm import LLM, SamplingParams
model=LLM(model=os.environ['BAPO_E1_MODEL'],tensor_parallel_size=1,gpu_memory_utilization=.12,max_model_len=1024,enforce_eager=True)
out=model.generate(['The capital of France is'],SamplingParams(temperature=0,max_tokens=16))
assert out[0].outputs[0].token_ids
import json,pathlib
record={'status':'REAL_OFFICIAL_INSTRUCT_VLLM_CUDA_OK','model_path':os.environ['BAPO_E1_MODEL'],'token_ids':out[0].outputs[0].token_ids,'text':out[0].outputs[0].text,'source_commit':os.environ['SOURCE']}
(pathlib.Path(os.environ['BAPO_RUN_ID'])/'model-gpu.json').write_text(json.dumps(record))
print('REAL_MODEL_VLLM_CUDA_OK',repr(out[0].outputs[0].text),flush=True)
PY
sleep 3
args=(--repo "$ROOT/repo" --work-root "$RUN" --index "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" --corpus "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" --model "$SEARCH_R1_ASSETS/assets/e5-base-v2" --asset-manifest "$SEARCH_R1_ASSETS/assets.json" --hash-large-assets --threads 8 --repeats 2)
PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" /usr/bin/time -v python "$ROOT/tools/bapo-retrieval-parity.py" --mode gpu "${args[@]}" --embeddings-out "$RUN/gpu-embeddings.npz" --output "$RUN/retrieval-gpu.json" > "$RUN/parity-gpu.log" 2> "$RUN/parity-gpu-time.log"
/usr/bin/time -v python "$ROOT/tools/bapo-retrieval-parity.py" --mode cpu "${args[@]}" --output "$RUN/retrieval-cpu.json" > "$RUN/parity-cpu.log" 2> "$RUN/parity-cpu-time.log"
python "$ROOT/tools/bapo-retrieval-parity.py" --mode compare --left "$RUN/retrieval-cpu.json" --right "$RUN/retrieval-gpu.json" --output "$RUN/retrieval-comparison.json"
# Force one certified OWN 310-process freeze with live local NCCL8.
# This is a scalability fixture, never evidence of an unsafe production sample.
python "$ROOT/tools/freeze_scalability_gpu_fixture.py" --run "$RUN" --source "$SOURCE"
# Reuse the allocation/lease/resource guardian for a real production-model
# eight-rank recovery. verify-only emits witnesses without generation/update.
bash "$ROOT/tools/stage-gpu-inner.sh" hardware
touch "$RUN/HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY"
# Measurement alone never signs the independent acceptance receipt.
