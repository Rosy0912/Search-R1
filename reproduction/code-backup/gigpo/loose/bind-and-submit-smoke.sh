set -euo pipefail
ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
test "$(cat "$ROOT/logs/native-cpu-2907.exit")" = 0
test "$(cat "$ROOT/logs/hardware-2903.exit")" = 0
tar -xzf "$ROOT/gigpo-reviewed-tools.tar.gz" -C "$ROOT/tools"
tar -xzf "$ROOT/gigpo-launch-final.tar.gz" -C "$ROOT/tools"
cd "$ROOT/repo"
test "$(git rev-parse HEAD)" = 27c989b9047421523c5ea400a2f67d897d35513f
test -z "$(git status --porcelain)"
export AGENT_ROOT="$ROOT"
BINDING_SHA=$(/usr/bin/python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
r=Path(os.environ['AGENT_ROOT'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reference=r/'logs/cpu-gpu-retrieval-comparison.json'
canonical=hashlib.sha256(json.dumps(json.loads(reference.read_text()),indent=2).encode()).hexdigest()
assert canonical=='18808aa9d2a763380d5b00cf028ce97b7c9d38b56c30958298cac97ffd93f94f',canonical
native=r/'logs/native-cpu-2907.log';assert '172 passed' in native.read_text() and 'NATIVE_CPU_AND_ASSET_GATE_PASS' in native.read_text()
items=[native,r/'logs/native-cpu-2907.exit',reference,r/'logs/hardware-2903.log',r/'logs/hardware-2903.exit',r/'runs/hardware-2903/resource-samples.jsonl',r/'runs/hardware-2903/actor-entropy-compile.json',r/'logs/cluster-copy-verification.json',r/'logs/environment-post-cpu.json',r/'logs/prepared-data-paths.json',r/'logs/deployment-independent-review.md']
items += [r/'tools'/name for name in ['gigpo-short-run.py','owned_runtime.py','gigpo-short.sbatch','resource_guard.py','check_http_retrieval.py','verify-gigpo-checkpoint.py','checkpoint_semantics.py']]
record=dict(status='REVIEWED',code_commit='27c989b9047421523c5ea400a2f67d897d35513f',public_commit='4bda3078b255b5a4d9cf90bf9a443ba3102c1e35',evidence=[dict(path=str(p),sha256=sha(p)) for p in items],reviewed_reference_normalized_json_sha256=canonical,native_cpu_pass=172,reviewer='review_common',scope='deployment and executor only; actual micro64 coexistence/eight-actor updates remain smoke gates')
p=r/'logs/deployment-binding-27c989b.json'
with p.open('x') as f:json.dump(record,f,indent=2)
print(sha(p))
PY
)
printf 'BINDING_SHA=%s\n' "$BINDING_SHA"
sbatch "$ROOT/tools/gigpo-short.sbatch" --stage smoke2 --binding "$ROOT/logs/deployment-binding-27c989b.json" --binding-sha "$BINDING_SHA"
