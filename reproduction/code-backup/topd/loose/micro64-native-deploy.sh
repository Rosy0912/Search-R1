set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
mkdir -p tools
 tar -xzf topd-micro64-tools-20261003.tar.gz -C tools
python3 - <<'PY'
import json,hashlib
from pathlib import Path
for n,h in json.loads(Path('tools/topd-launcher-tool-hashes.json').read_text()).items():
 assert hashlib.sha256((Path('tools')/n).read_bytes()).hexdigest()==h,n
print('MICRO64_ALL_TOOL_BYTES_VERIFIED')
PY
record=logs/micro64-native-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/launcher-native.sbatch > "$record"; cat "$record"; fi
