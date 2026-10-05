#!/bin/bash
set -euo pipefail
BASE=${E1_BASE:?};PROFILE=${E1_PROFILE:?archive or r1};STAGE=${E1_STAGE:-smoke}
RUNTIME="$BASE/runtime/$PROFILE"
continuing=0
if [[ "$PROFILE" == archive && -f "$BASE/receipts/archive-live-jit-optimizer.json" ]]; then
 [[ "${E1_GPUS:-4}" == 4 ]] || exit 2
 RUNTIME="$BASE/runtime/archive-jit-20261005-env-v2"
 export E1_JIT_OPTIMIZER=1 E1_PHYSICAL_SCORE=1
 "$BASE/env/bin/python" - "$BASE" "$RUNTIME" <<'VERIFY_ARCHIVE_JIT'
import sys,json,hashlib
from pathlib import Path
base,runtime=map(Path,sys.argv[1:])
manifest=json.loads((base/'receipts/archive-jit-env-v2-runtime.sha256.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==digest,name
VERIFY_ARCHIVE_JIT
fi
if [[ "$PROFILE" == r1 && "$STAGE" == round-1-grpo && -f "$BASE/receipts/continue-current-eight.json" ]]; then
 [[ "${E1_GPUS:-4}" == 8 ]] || exit 2
 RUNTIME="$BASE/runtime/r1-full-resume-20261004"
 export E1_FULL_RESUME_STATE="$BASE/resume-state/8d708a8556084021ae3069b93335109d"
 test -f "$E1_FULL_RESUME_STATE/full-complete.json"
 "$BASE/env/bin/python" - "$BASE" "$RUNTIME" <<'VERIFY'
import sys,json,hashlib
from pathlib import Path
base,runtime=map(Path,sys.argv[1:])
manifest=json.loads((base/'receipts/r1-full-resume-runtime.sha256.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==digest,name
VERIFY
 continuing=1
fi

export PYTHON="$BASE/env/bin/python"
export PYTHONPATH="$RUNTIME/upstream/Search-R1:$RUNTIME"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export GPUS=${E1_GPUS:-4}
if [[ "$GPUS" != 4 && "$GPUS" != 8 ]]; then exit 2; fi
if [[ "$GPUS" == 8 && "$PROFILE" != r1 ]]; then exit 2; fi
export MODEL=${E1_MODEL:-$BASE/assets/model}
export OUTPUT="$BASE/runs/$PROFILE/$STAGE"
export DATA_DIR="$BASE/data"
export RETRIEVER_URL=${E1_RETRIEVER_URL:-http://10.120.17.101:18001/retrieve}
export GRPO_SCORE_PER_GPU=4
if [[ "$PROFILE" == archive ]]; then
 export STEPS=200 MINIBATCH=128 GRPO_MICRO_PER_GPU=2
 entropy=0.0
elif [[ "$PROFILE" == r1 ]]; then
 export STEPS=500 MINIBATCH=256 GRPO_MICRO_PER_GPU=16 GRPO_SCORE_PER_GPU=32
 if [[ "$GPUS" == 8 ]]; then
  export GRPO_MICRO_PER_GPU=8 GRPO_SCORE_PER_GPU=16 E1_VIRTUAL_DP_WORLD=4
 fi
 entropy=0.001
else
 exit 2
fi
if [[ -n "${E1_ALIGNED_STEPS:-}" ]]; then
 [[ "$PROFILE" == r1 && "$GPUS" == 8 && "$continuing" == 0 ]] || exit 2
 [[ "$E1_ALIGNED_STEPS" == 200 || "$E1_ALIGNED_STEPS" == 300 ]] || exit 2
 export STEPS="$E1_ALIGNED_STEPS" MINIBATCH=128
fi
# Only A activates the archive's per-trajectory reduction. B preserves micro64.
if [[ "$PROFILE" == r1 ]]; then
 export E1_DISABLE_TRAJECTORY_BATCHING=1
fi
mkdir -p "$OUTPUT"
util=${E1_VLLM_UTIL:-0.4}
options=(actor_rollout_ref.actor.entropy_coeff="$entropy" actor_rollout_ref.rollout.gpu_memory_utilization="$util" ++trainer.val_before_train=true trainer.save_freq=50 trainer.test_freq=50)
if [[ "$STAGE" == smoke* ]]; then
 options+=(+trainer.stop_after=2 trainer.save_freq=1 ++trainer.val_before_train=false)
fi
if [[ "$continuing" == 1 ]]; then
 options+=(++trainer.val_before_train=false)
fi
if [[ "${E1_CONFIG_ONLY:-0}" == 1 ]]; then options+=(--cfg job); fi
cd "$RUNTIME"
if [[ "$continuing" == 1 ]]; then
 "$PYTHON" - "$BASE" "$OUTPUT/train.log" <<'LOG'
from pathlib import Path
import sys,re
base=Path(sys.argv[1]);source=base/'runs/r1/smoke-025/train.log'
rows=[s for s in source.read_text().splitlines() if re.search(r'\bstep:[12] - ',s) and 'actor/' in s]
assert len(rows)==2
Path(sys.argv[2]).write_text('CONTINUATION_SOURCE_PRESERVED_NATIVE_METRICS='+str(source)+'\n'+'\n'.join(rows)+'\n')
LOG
 bash scripts/grpo.sh "${options[@]}" >> "$OUTPUT/train.log" 2>&1
else
 bash scripts/grpo.sh "${options[@]}" > "$OUTPUT/train.log" 2>&1
fi
