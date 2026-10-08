set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd/repo
test -z "$(git status --porcelain)"
git bundle verify ../topd-parallel-gpu-source.bundle
git fetch ../topd-parallel-gpu-source.bundle reproduce/topd
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = a3103a2082c9df77cee6bb7505d2666ee9df4eb2
cd ..
tar -xzf topd-parallel-gpu-tools.tar.gz -C tools
python3 - <<'PY'
from pathlib import Path
import hashlib,json
for n,h in json.loads(Path('tools/parallel-gpu-tools-files.json').read_text())['files'].items():
    assert hashlib.sha256(Path('tools',n).read_bytes()).hexdigest()==h,n
print('PARALLEL_GPU_TOOL_BYTE_GATE_PASS')
PY
record=logs/parallel-gpu-source-native-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/parallel-gpu-native.sbatch > "$record"; cat "$record"; fi
