#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
cd "$ROOT/repo"
"$ROOT/env/bin/python" - <<'CHECK'
import hashlib,pathlib
root=pathlib.Path("/dataset/shaoshitong/search-r1-rl/tree-grpo/tools")
expected={'tree_shared_supervisor_fixed.py': 'be4e9f057c4e28a6cd25f4172ec8fd62f789549e3a9b8faeff3102028b993378', 'tree_shared_owned_runtime_fixed.py': '4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0', 'probe_tree_shared_retrieval.py': '335a3410bdc19fac01d5d29aa7eea3832c031fae5093f1f9f2d65f76374157b5', 'resource_guard.py': '790c349e117c2020778c606115e8c94abf425008d96db8f5539ceaaa5df74381'}
for name,wanted in expected.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted, "Unreviewed helper: "+name
CHECK
export PYTHONPATH="$ROOT/tools:$ROOT/repo"
export PYTHONUNBUFFERED=1
exec "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/tree_shared_supervisor_fixed.py" "$@"
