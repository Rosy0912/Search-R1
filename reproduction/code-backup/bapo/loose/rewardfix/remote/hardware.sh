#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097
RUN=$1
port=$(python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/repo/reproduction/h20/cuda_preflight.py"
OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/flash_preflight.py"
first_gpu=${CUDA_VISIBLE_DEVICES%%,*}
CUDA_VISIBLE_DEVICES="$first_gpu" python - <<'PY'
import os
from vllm import LLM, SamplingParams
model=LLM(model=os.environ['SEARCH_R1_ASSETS']+'/assets/qwen2.5-3b',tensor_parallel_size=1,gpu_memory_utilization=.12,max_model_len=1024,enforce_eager=True)
out=model.generate(['The capital of France is'],SamplingParams(temperature=0,max_tokens=16))
assert out[0].outputs[0].token_ids
print('REAL_MODEL_VLLM_CUDA_OK',repr(out[0].outputs[0].text),flush=True)
PY
sleep 3
args=(--repo "$ROOT/repo" --work-root "$RUN" --index "$SEARCH_R1_ASSETS/assets/wiki/e5_Flat.index" --corpus "$SEARCH_R1_ASSETS/assets/wiki/wiki-18.jsonl" --model "$SEARCH_R1_ASSETS/assets/e5-base-v2" --asset-manifest "$SEARCH_R1_ASSETS/assets.json" --hash-large-assets --threads 8 --repeats 2)
PYTHONPATH="$ROOT/retriever-gpu-lib:$ROOT/repo" /usr/bin/time -v python "$ROOT/tools/bapo-retrieval-parity.py" --mode gpu "${args[@]}" --embeddings-out "$RUN/gpu-embeddings.npz" --output "$RUN/retrieval-gpu.json" > "$RUN/parity-gpu.log" 2> "$RUN/parity-gpu-time.log"
/usr/bin/time -v python "$ROOT/tools/bapo-retrieval-parity.py" --mode cpu "${args[@]}" --output "$RUN/retrieval-cpu.json" > "$RUN/parity-cpu.log" 2> "$RUN/parity-cpu-time.log"
python "$ROOT/tools/bapo-retrieval-parity.py" --mode compare --left "$RUN/retrieval-cpu.json" --right "$RUN/retrieval-gpu.json" --output "$RUN/retrieval-comparison.json"
touch "$RUN/HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY"
# Measurement alone never signs the independent acceptance receipt.
