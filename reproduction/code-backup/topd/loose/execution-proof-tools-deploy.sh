set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = a3103a2082c9df77cee6bb7505d2666ee9df4eb2
test -z "$(git -C repo status --porcelain)"
tar -xzf topd-execution-proof-tools.tar.gz -C tools
python3 - <<'PY'
from pathlib import Path
import hashlib,json
for n,h in json.loads(Path('tools/execution-proof-tools-files.json').read_text())['files'].items():
    assert hashlib.sha256(Path('tools',n).read_bytes()).hexdigest()==h,n
parent=Path('logs/parallel-gpu-native-2940.log');raw=parent.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='c096ddda92ac3481017cd31bbc5d303a09ad8df48fa093520ceb4d2de62b68f6'
marker=b'205 passed, 3 warnings in 20.26s\n';end=raw.index(marker)+len(marker)
assert end==1975
out=Path('logs/parallel-gpu-source-suite-2940.log')
if out.exists():assert out.read_bytes()==raw[:end]
else:out.write_bytes(raw[:end])
assert hashlib.sha256(out.read_bytes()).hexdigest()=='689ff93afcb035390c873b03c51ef3daf66b5199919df3a6c6b2a5137635ce15'
print('FINAL_TOOL_BYTES_AND_EXACT_RAW_SOURCE_LOG_SLICE_PASS')
PY
record=logs/execution-proof-native-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/execution-proof-native.sbatch > "$record"; cat "$record"; fi
