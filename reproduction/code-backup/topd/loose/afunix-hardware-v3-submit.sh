set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = 438af3529b40663aae7b8293080e09fda57f6a87
test -z "$(git -C repo status --porcelain)"
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('.');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'topd-gpu-oracle-tools-v3.tar.gz')=='66fa728dc7d11167179185bd08bbb84016f1de6155d65bded18c51814ae0256a'
assert sha(root/'afunix-scorer-hardware-probe-v3.tar.gz')=='f666ed23aa2d283bc8209abbdf5b9a55d1bfa1c5502c96bcd7fac83d4987c8d6'
for name,h in json.loads((root/'tools/repair-tools-files.json').read_text())['files'].items():assert sha(root/'tools'/name)==h,name
base=root/'tools/afunix-scorer-probe'
for name,h in json.loads((base/'files.json').read_text())['files'].items():assert sha(base/name)==h,name
review=json.loads((base/'reviewed-sources.json').read_text())
assert review['code_commit']=='438af3529b40663aae7b8293080e09fda57f6a87'
assert review['source_fingerprint']=='101eb46fdd5cc493d8be3be11cb9d63313b3e94819bd9d2034b37a1f94e3a9a6'
for name,h in review['files'].items():assert sha(root/name)==h,name
assert (root/'logs/gpu-oracle-v3-tools-submission.txt').read_text().strip()=='2981'
assert (root/'logs/gpu-oracle-v3-normalizer-submission.txt').read_text().strip()=='2982'
print('PROSPECTIVE_GPU_ORACLE_V3_HARDWARE_SOURCE_VERIFIED_NO_TRAINING')
PY
record=logs/afunix-scorer-hardware-v3-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable --dependency=afterok:2966:2980:2981:2982 tools/afunix-scorer-probe/probe.sbatch > "$record"; cat "$record"; fi
