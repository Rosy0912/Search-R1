set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = a3103a2082c9df77cee6bb7505d2666ee9df4eb2
test -z "$(git -C repo status --porcelain)"
tar -xzf afunix-scorer-hardware-probe-v1.tar.gz -C tools
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('.');base=root/'tools/afunix-scorer-probe'
manifest=json.loads((base/'files.json').read_text())
for n,h in manifest['files'].items():
    assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n
reviewed=json.loads((base/'reviewed-sources.json').read_text())
for n,h in reviewed['files'].items():
    assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert reviewed['files']['tools/topd-short-run.py']=='c7e0b530753f2cc5bb71b1d8dca0c539d697eed2fea82759df19a83812656638'
print('FINAL_AFUNIX_HARDWARE_ONLY_PROBE_BYTES_PASS')
PY
record=logs/afunix-scorer-hardware-v1-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable --dependency=afterok:2954 tools/afunix-scorer-probe/probe.sbatch > "$record"; cat "$record"; fi
