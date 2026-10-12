#!/bin/bash
set -euo pipefail
ROOT=/dataset/shaoshitong/search-r1-rl/tree-grpo
cd "$ROOT/repo"
"$ROOT/env/bin/python" - <<'CHECK'
import hashlib,pathlib
root=pathlib.Path("/dataset/shaoshitong/search-r1-rl/tree-grpo/tools")
expected={'tree_shared_supervisor_formal_cost_identity.py': '21a50c834a61ced93ae3616f9f87d5d40ad7078c9e6b5a387550b3eeb28fd73a', 'tree_shared_owned_runtime_fixed.py': '4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0', 'probe_tree_shared_retrieval.py': '335a3410bdc19fac01d5d29aa7eea3832c031fae5093f1f9f2d65f76374157b5', 'resource_guard_identity.py': '284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88'}
expected.update({'validation_cost_entry.py': '9d36c4df859ae152975515ae105b1ad2df10bae68064265d09cdb5fcdcc68fe8', 'validation_cost_hooks.py': '63fafc8555d23e78c474e311dec96526eaf1a9e8bb5b57fc7599e5fcdc323e8c', 'run_tree_with_validation_cost_shared.sh': 'd867a3205cfb03e03fad59a1110bab908785dfa8e02f7b323012f5b52a24b74d'})
for name,wanted in expected.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==wanted, "Unreviewed helper: "+name
CHECK
export PYTHONPATH="$ROOT/tools:$ROOT/repo"
export PYTHONUNBUFFERED=1
exec "$ROOT/env/bin/python" -m search_r1.baselines.preflight -- "$ROOT/env/bin/python" "$ROOT/tools/tree_shared_supervisor_formal_cost_identity.py" --stage formal "$@"
