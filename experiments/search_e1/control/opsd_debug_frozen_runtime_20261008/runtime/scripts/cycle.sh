#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-$ROOT/.venv/bin/python}
MODEL=${MODEL:-$ROOT/models/Qwen2.5-3B-Instruct}
RUN=${RUN:-$ROOT/runs/paper-3b}
export PYTHON PYTHONPATH="$ROOT/upstream/Search-R1:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN"
for round in 1 2; do
  DIR="$RUN/round-$round"
  mkdir -p "$DIR"
  if [[ ! -f "$DIR/grpo.done" ]]; then
    MODEL="$MODEL" OUTPUT="$DIR/grpo" bash scripts/grpo.sh 2>&1 | tee "$DIR/grpo.log"
    touch "$DIR/grpo.done"
  fi
  CHECKPOINT=$("$PYTHON" - "$DIR/grpo" <<'PY'
import sys
from pathlib import Path
paths = list(Path(sys.argv[1]).glob('actor/global_step_*'))
if not paths:
    raise SystemExit('GRPO produced no HF actor checkpoint')
print(max(paths, key=lambda p: int(p.name.rsplit('_', 1)[1])))
PY
)
  if [[ ! -f "$DIR/rollouts.done" ]]; then
    "$PYTHON" -m search_e1.rollout --model "$CHECKPOINT" --data data/train.jsonl --output "$DIR/rollouts.jsonl" --k 5 --backend vllm --retriever-url "${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
    touch "$DIR/rollouts.done"
  fi
  "$PYTHON" -m search_e1.mine --rollouts "$DIR/rollouts.jsonl" --output "$DIR/pairs.jsonl"
  if [[ ! -f "$DIR/opsd.done" ]]; then
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${GPUS:-4}" -m search_e1.opsd --model "$CHECKPOINT" --pairs "$DIR/pairs.jsonl" --validation data/validation.jsonl --output "$DIR/opsd" --retriever-url "${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}" 2>&1 | tee "$DIR/opsd.log"
    touch "$DIR/opsd.done"
  fi
  MODEL="$DIR/opsd/merged"
done
"$PYTHON" -m search_e1.rollout --model "$MODEL" --data data/test.jsonl --output "$RUN/test.jsonl" --k 1 --backend vllm --temperature 0 --retriever-url "${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
