set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = 438af3529b40663aae7b8293080e09fda57f6a87
test -z "$(git -C repo status --porcelain)"
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('.');base=root/'tools/afunix-scorer-probe'
for name,expected in json.loads((base/'files.json').read_text())['files'].items():
    assert hashlib.sha256((base/name).read_bytes()).hexdigest()==expected,name
reviewed=json.loads((base/'reviewed-sources.json').read_text())
assert reviewed['code_commit']=='438af3529b40663aae7b8293080e09fda57f6a87'
assert reviewed['source_fingerprint']=='101eb46fdd5cc493d8be3be11cb9d63313b3e94819bd9d2034b37a1f94e3a9a6'
for name,expected in reviewed['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print('REPAIRED_HARDWARE_V2_SOURCE_BYTES_VERIFIED_NO_TRAINING')
PY
record=logs/afunix-scorer-hardware-v2-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable --dependency=afterok:2966:2967 tools/afunix-scorer-probe/probe.sbatch > "$record"; cat "$record"; fi
