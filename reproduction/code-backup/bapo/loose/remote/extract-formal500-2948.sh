set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$R/env/bin/python" - "$R/tools/bapo-formal500-evidence-gpu64-tied.py" <<'PY'
import hashlib,sys,pathlib
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()=='329037a3b1bc6b77a774c1964fa123e30d0da3481f2d8b19ee9db815e31f1166'
PY
"$R/env/bin/python" "$R/tools/test_bapo_formal500_evidence.py" > "$R/logs/formal500-2948-extractor-native-tests.log" 2>&1
RUN="$R/runs/bapo-formal500-2948"
"$R/env/bin/python" "$R/tools/bapo-formal500-evidence-gpu64-tied.py" --run-root "$RUN" --log "$RUN/train.log" --base-model "$R/assets-root/assets/qwen2.5-3b" --compare-weights --hf-load --hash-checkpoint --slurm-record "$R/logs/formal500-2948-terminal-slurm-watch.log" --output "$R/logs/formal500-2948-gpu64-tied-extraction.json" > "$R/logs/formal500-2948-gpu64-tied-extractor.log" 2>&1
