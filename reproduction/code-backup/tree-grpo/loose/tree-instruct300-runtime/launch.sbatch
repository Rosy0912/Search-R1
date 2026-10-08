#!/usr/bin/env bash
#SBATCH --job-name=tree-instruct300
#SBATCH --nodes=1
#SBATCH --nodelist=h100-9
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
set -euo pipefail
: "${TREE_INSTRUCT_TOOLS:?Provide reviewed immutable package path}"
: "${TREE_INSTRUCT_PACKAGE_SHA256:?Provide independent reviewed manifest hash}"
: "${TREE_INSTRUCT_REVIEW_SHA256:?Provide independent pre-dispatch receipt hash}"
export PYTHONPATH="$TREE_INSTRUCT_TOOLS:/dataset/shaoshitong/search-r1-rl/tree-grpo/tools:$TREE_INSTRUCT_TOOLS/source/repo"
cd "$TREE_INSTRUCT_TOOLS/source/repo"
/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python - <<'PY'
import os,pathlib,hashlib,json
root=pathlib.Path(os.environ['TREE_INSTRUCT_TOOLS'])
manifest=root/'manifest.json';receipt=root/'PRE_DISPATCH_RECEIPT.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==os.environ['TREE_INSTRUCT_PACKAGE_SHA256']
assert hashlib.sha256(receipt.read_bytes()).hexdigest()==os.environ['TREE_INSTRUCT_REVIEW_SHA256']
r=json.loads(receipt.read_bytes())
assert r['status']=='READY_FOR_SAME_JOB_NATIVE_GATE_THEN_INSTRUCT300'
assert r['manifest_sha256']==os.environ['TREE_INSTRUCT_PACKAGE_SHA256']
assert r['requires_native8_fullstate_before3b'] is True and r['requires_independent_formal5_before6'] is True
m=json.loads(manifest.read_bytes())
for relative,expected in m['files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
PY
exec /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -m search_r1.baselines.preflight -- /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python "$TREE_INSTRUCT_TOOLS/supervisor.py"
