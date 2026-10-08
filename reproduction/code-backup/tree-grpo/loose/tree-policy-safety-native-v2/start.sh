#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
CAND=$ROOT/candidates/tree-policy-safety-native-0980ec3-v2
export PYTHONPATH="$CAND/repo:$ROOT/tools" PYTHONDONTWRITEBYTECODE=1
export PATH="$ROOT/env/bin:$PATH"
cd "$CAND/repo"
"$ROOT/env/bin/python" - "$CAND" <<'CHECK'
import hashlib,json,pathlib,sys,os
p=pathlib.Path(sys.argv[1]);m=p/'manifest.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()=='b9c2e07714a745e462d3678328ad4d25a3a83958c1683787cf63e1562768137c'
r=json.loads(m.read_text())
for file,wanted in r['files'].items():assert hashlib.sha256((p/file).read_bytes()).hexdigest()==wanted,file
for file,wanted in r['external_helpers'].items():assert hashlib.sha256((pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/tools')/file).read_bytes()).hexdigest()==wanted,file
assert os.environ['SLURM_JOB_NODELIST']=='h100-9' and os.environ['SLURM_JOB_GPUS']
CHECK
exec srun --ntasks=1 "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$CAND/native-run.py" "$CAND"
