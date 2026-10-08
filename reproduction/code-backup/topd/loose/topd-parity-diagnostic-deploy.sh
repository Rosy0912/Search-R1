set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = 438af3529b40663aae7b8293080e09fda57f6a87
tar -xzf topd-parity-diagnostic-v1.tar.gz -C tools
python3 - <<'PY'
from pathlib import Path
import hashlib,json
assert hashlib.sha256(Path('topd-parity-diagnostic-v1.tar.gz').read_bytes()).hexdigest()=='27355f032de1d727b95a0ba40e7e4d82cf17d90d7284a23455eab2642d8fb886'
for name,expected in json.loads(Path('tools/parity-diagnostic-files.json').read_text())['files'].items():
    assert hashlib.sha256(Path('tools',name).read_bytes()).hexdigest()==expected,name
print('FAILED_ORIGINAL_ROW_DIAGNOSTIC_FROZEN_BYTES_VERIFIED_NO_TRAINING')
PY
record=logs/parity-diagnostic-v1-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/topd-parity-diagnostic.sbatch > "$record"; cat "$record"; fi
