#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
TOOLS=$ROOT/candidates/tree-fixed-formal-296217a
CAND=$ROOT/candidates/tree-native-numeric-296217a
export PYTHONPATH="$ROOT/tools:$CAND/repo" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PATH="$ROOT/env/bin:$PATH"
cd "$CAND/repo"
"$ROOT/env/bin/python" - "$TOOLS" <<'CHECK'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);manifest=p/'manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='f66afebc025db70c2cd9391442fae6834722de95a6dcafc030416d60dee4e911'
m=json.loads(manifest.read_text())
for file,expected in m['files'].items():assert hashlib.sha256((p/file).read_bytes()).hexdigest()==expected,file
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
for file,expected in m['external_helpers'].items():assert hashlib.sha256((root/'tools'/file).read_bytes()).hexdigest()==expected,file
assert hashlib.sha256((root/'repo/reproduction/h20/retriever_local.py').read_bytes()).hexdigest()==m['retriever_script_sha256']
CHECK
exec "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$TOOLS/supervisor.py" --stage formal --previous-review "$TOOLS/NATIVE3063_ACTUAL_RECEIPT.json" --previous-review-sha bbde9887931e5d8e09bdaf554459a1c86712b92ada141fc0543e51a075e18675 --actual-two-review "$TOOLS/ACTUAL3095_RECEIPT.json" --actual-two-review-sha e79c8cf1f580b1aae3eb5119667aff7c31247c45ec053de085a32710d966c6c7 --actual-five-review "$TOOLS/ACTUAL3103_RECEIPT.json" --actual-five-review-sha 4dac9d6c2b71ec57751e1f6b6df886671b08bc11fe1947b334bbcdbdf59ca59f
