#!/bin/bash
set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
RUN=$R/runs/bapo-fp32-cpu-parity-20261003
mkdir "$RUN"
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
cd "$R/repo"
/usr/bin/time -v "$R/env/bin/python" "$R/tools/bapo-retrieval-fp32-parity.py" --mode cpu --encoder-fp32 --repo "$R/repo" --work-root "$RUN" --index "$R/assets-root/assets/wiki/e5_Flat.index" --corpus "$R/assets-root/assets/wiki/wiki-18.jsonl" --model "$R/assets-root/assets/e5-base-v2" --threads 8 --batch-size 512 --repeats 2 --hash-large-assets --asset-manifest "$R/assets-root/assets.json" --output "$RUN/parity-cpu-fp32.json" > "$RUN/parity.log" 2> "$RUN/time.log"
"$R/env/bin/python" "$R/tools/bapo-retrieval-fp32-parity.py" --mode compare --left "$R/runs/bapo-hardware-2855/parity-gpu.json" --right "$RUN/parity-cpu-fp32.json" --output "$RUN/comparison-original-gpu-fp32.json"
touch "$RUN/MEASUREMENTS_DONE"
