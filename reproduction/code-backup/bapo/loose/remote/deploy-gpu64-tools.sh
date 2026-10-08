set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
export PATH="$R/env/bin:$PATH"
exec 8> "$R/gpu-job.lock"
flock -n 8
test ! -e "$R/tools/stage-b6ecbeb.sh" || cmp "$R/tools/stage-b6ecbeb.sh" "$R/tools/stage.sh"
cp "$R/tools/stage.sh" "$R/tools/stage-b6ecbeb.sh"
cp "$R/tools/resource_watch.py" "$R/tools/resource_watch-pre-gpu64.py"
tar -xf "$R/tools/gpu64-tools.tar" -C "$R/tools"
python - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('/dataset/shaoshitong/search-r1-rl/bapo/tools')
records=json.loads((r/'gpu64-tools-hashes.json').read_text())
for name,h in records.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
print('GPU64_TOOLS_ALL_HASHES_VERIFIED',json.dumps(records,sort_keys=True))
PY
