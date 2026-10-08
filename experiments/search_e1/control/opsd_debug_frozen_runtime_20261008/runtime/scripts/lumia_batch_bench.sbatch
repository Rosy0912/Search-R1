#!/bin/bash
#SBATCH --job-name=search-e1-batch-bench
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/jmzhang/Workspace/Search-E1-20261001
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
set -euo pipefail
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/upstream/Search-R1:$PWD"
export RUN="$PWD/runs/batch-bench-$SLURM_JOB_ID"
export OMP_NUM_THREADS=4
export RETRIEVER_BACKEND=cpu
mkdir -p "$RUN/data"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" - "$RUN/data" <<'PY'
import sys
from datasets import load_dataset
for split,n in [('train',16),('validation',4)]:
    ds=load_dataset('parquet',data_files=f'data/{split}.parquet',split='train').select(range(n))
    ds.to_parquet(f'{sys.argv[1]}/{split}.parquet')
PY
source scripts/start_retriever.sh
MODEL="$PWD/models/Qwen2.5-3B-Instruct" OUTPUT="$RUN/grpo" DATA_DIR="$RUN/data" GPUS=2 STEPS=2 BATCH=16 MINIBATCH=16 \
 GRPO_MICRO_PER_GPU=2 GRPO_SCORE_PER_GPU=4 bash scripts/grpo.sh \
 trainer.save_freq=2 trainer.test_freq=-1 data.val_data_num=4 data.val_batch_size=4 2>&1 | tee "$RUN/grpo.log"
test -f "$RUN/grpo/actor/global_step_2/config.json"
touch "$RUN/batch-bench.done"
