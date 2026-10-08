set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = 438af3529b40663aae7b8293080e09fda57f6a87
test -z "$(git -C repo status --porcelain)"
tar -xzf topd-repaired-tools-v2.tar.gz -C tools
tar -xzf afunix-scorer-hardware-probe-v2.tar.gz -C tools
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('.')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(root/'topd-repaired-tools-v2.tar.gz')=='bed8d801deba9be03c1e0acdb820e0863165c3ae59db590ba883dc411c804647'
assert sha(root/'afunix-scorer-hardware-probe-v2.tar.gz')=='563fadff3db770de79c95b9209a520e2b99a62a4c18de27dd7b5886fd5a4bc7a'
tools=json.loads((root/'tools/repair-tools-files.json').read_text())
assert tools['native_counts']==dict(source=205,tools=123)
for name,expected in tools['files'].items():assert sha(root/'tools'/name)==expected,name
base=root/'tools/afunix-scorer-probe'
for name,expected in json.loads((base/'files.json').read_text())['files'].items():assert sha(base/name)==expected,name
for name,expected in json.loads((base/'reviewed-sources.json').read_text())['files'].items():assert sha(root/name)==expected,name
print('REPAIRED_TOOLS_AND_HARDWARE_V2_EXACT_BYTES_PASS_NO_TRAINING')
PY
tool_record=logs/scorer-cleanup-repair-tools-v2-submission.txt
if test -f "$tool_record"; then cat "$tool_record"; else sbatch --parsable tools/execution-proof-native.sbatch > "$tool_record"; cat "$tool_record"; fi
