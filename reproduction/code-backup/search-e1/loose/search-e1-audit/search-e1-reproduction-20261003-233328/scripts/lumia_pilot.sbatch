#!/bin/bash
#SBATCH --job-name=search-e1-pilot
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/jmzhang/Workspace/Search-E1-20261001
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
set -euo pipefail
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
export RUN="$PWD/runs/pilot-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4
mkdir -p "$RUN/data"
"$PYTHON" scripts/gpu_smoke.py --flash
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/distributed_smoke.py
"$PYTHON" - "$RUN/data" <<'PY'
import sys
from datasets import load_dataset
from search_e1.io import read_jsonl,write_jsonl
for split,n in [('train',8),('validation',4)]:
    ds=load_dataset('parquet',data_files=f'data/{split}.parquet',split='train').select(range(n))
    ds.to_parquet(f'{sys.argv[1]}/{split}.parquet')
    write_jsonl(f'{sys.argv[1]}/{split}.jsonl',read_jsonl(f'data/{split}.jsonl')[:n])
PY
source scripts/start_retriever.sh
MODEL="$PWD/models/Qwen2.5-3B-Instruct" OUTPUT="$RUN/grpo" DATA_DIR="$RUN/data" GPUS=4 STEPS=2 BATCH=8 MINIBATCH=8 \
 bash scripts/grpo.sh trainer.save_freq=2 trainer.test_freq=-1 data.val_data_num=4 data.val_batch_size=4 2>&1 | tee "$RUN/grpo.log"
CHECKPOINT="$RUN/grpo/actor/global_step_2"
"$PYTHON" -m search_e1.rollout --model "$CHECKPOINT" --data "$RUN/data/train.jsonl" --output "$RUN/rollouts.jsonl" --k 5 --backend vllm --batch-size 8 --retriever-url "$RETRIEVER_URL"
"$PYTHON" -m search_e1.mine --rollouts "$RUN/rollouts.jsonl" --output "$RUN/pairs.jsonl"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m search_e1.opsd --model "$CHECKPOINT" --pairs "$RUN/pairs.jsonl" --validation "$RUN/data/validation.jsonl" --output "$RUN/opsd" --retriever-url "$RETRIEVER_URL" --accumulation 1
"$PYTHON" -m search_e1.rollout --model "$RUN/opsd/merged" --data "$RUN/data/validation.jsonl" --output "$RUN/merged_validation.jsonl" --k 1 --temperature 0 --retriever-url "$RETRIEVER_URL"
touch "$RUN/pilot.done"
