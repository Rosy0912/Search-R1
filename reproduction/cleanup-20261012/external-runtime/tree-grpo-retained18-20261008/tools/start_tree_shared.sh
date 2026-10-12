#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
cd "$ROOT/repo"
"$ROOT/env/bin/python" - <<'CHECK'
import hashlib,pathlib
root=pathlib.Path("/dataset/shaoshitong/search-r1-rl/tree-grpo/tools")
expected={'tree_shared_supervisor.py': '7b6be0bea308da63150c68aeff27876d3dcb7f8edbc4d7680b762b6b38e83dcc', 'tree_shared_owned_runtime.py': '31d8e50b08df5d39cb28efc799863f99edfe81e02d86baa594508120b3ca676f', 'probe_tree_shared_retrieval.py': '335a3410bdc19fac01d5d29aa7eea3832c031fae5093f1f9f2d65f76374157b5', 'resource_guard.py': '790c349e117c2020778c606115e8c94abf425008d96db8f5539ceaaa5df74381'}
for name,wanted in expected.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted, "Unreviewed helper: "+name
CHECK
export PYTHONPATH="$ROOT/tools:$ROOT/repo"
export PYTHONUNBUFFERED=1
exec "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/tree_shared_supervisor.py" "$@"
