set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
tar -xzf gpu-coleasing-probe-numeric-v2.tar.gz -C tools
python3 - <<'PY'
from pathlib import Path
import json,hashlib
base=Path('tools/gpu-coleasing-probe')
for n,h in json.loads((base/'files.json').read_text())['files'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n
print('GPU_COEXIST_NUMERIC_V2_TOOL_SHA_PASS_NO_TRAINING')
PY
record=logs/gpu-coexist-numeric-v2-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/gpu-coleasing-probe/probe.sbatch > "$record"; cat "$record"; fi
