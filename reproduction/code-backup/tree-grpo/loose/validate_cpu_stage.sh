#!/bin/bash
set -euo pipefail
set -o noclobber
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
STAGE=${1:?two|five|formal}
case "$STAGE" in two|five|formal) ;; *) exit 2 ;; esac
RUN=$(cat "$ROOT/logs/$STAGE-stage-completed.run")
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH="$ROOT/repo" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$ROOT/env/bin/python" - "$ROOT" "$STAGE" "$RUN" <<'CHECK'
import pathlib,sys,json,re
root,stage,run=sys.argv[1:];assert re.fullmatch(re.escape(root)+r'/runs/tree-'+stage+r'-[1-9][0-9]*',run)
assert json.loads(pathlib.Path(run,'cost.json').read_text())['exit_code']==0
CHECK
"$ROOT/env/bin/python" "$ROOT/tools/validate_tree_stage.py" --run "$RUN" --stage "$STAGE" --base "$ROOT/assets-root/assets/qwen2.5-3b" --output "$RUN/stage-validation-$SLURM_JOB_ID.json" --require-tree-audit > "$RUN/stage-validation-$SLURM_JOB_ID.log" 2>&1
