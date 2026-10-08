set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
tar -xzf topd-gpu-lease-tools-20261003.tar.gz -C tools
python3 - <<'PY'
import json,hashlib
from pathlib import Path
for n,h in json.loads(Path('tools/topd-launcher-tool-hashes.json').read_text()).items():assert hashlib.sha256((Path('tools')/n).read_bytes()).hexdigest()==h,n
print('GPU_LEASE_TOOLS_BYTE_GATE_PASS')
PY
record=logs/gpu-lease-native-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/launcher-native.sbatch > "$record"; cat "$record"; fi
