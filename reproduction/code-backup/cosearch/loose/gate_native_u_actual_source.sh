set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/cosearch/author-native
STEP=$(python3 -c 'import json; print(json.load(open("/dataset/shaoshitong/search-r1-rl/cosearch/author-native/continue-300-tools-u/source-checkpoint-verification.json"))["source_step"])')
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 COSEARCH_NATIVE_RUN_ID=author-native-continue-300-u COSEARCH_NATIVE_SOURCE_RUN_ID=author-native-continue-300-t COSEARCH_NATIVE_RESUME_STEP="$STEP" COSEARCH_NATIVE_CONTINUE_TO=300 PYTHONPATH="$ROOT/continue-300-tools-u:$ROOT/repo:$ROOT/repo/verl" "$ROOT/env/bin/python" "$ROOT/continue-300-tools-u/cpu_continue_gate.py" > "$ROOT/continue-300-tools-u/cpu-continue-gate.log" 2>&1
cat "$ROOT/continue-300-tools-u/cpu-continue-gate.json"
