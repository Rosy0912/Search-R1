set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = 438af3529b40663aae7b8293080e09fda57f6a87
test -z "$(git -C repo status --porcelain)"
python3 - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
archives={
 'topd-gpu-oracle-tools-v3.tar.gz':('66fa728dc7d11167179185bd08bbb84016f1de6155d65bded18c51814ae0256a',16,'repair-tools-files.json',''),
 'afunix-scorer-hardware-probe-v3.tar.gz':('f666ed23aa2d283bc8209abbdf5b9a55d1bfa1c5502c96bcd7fac83d4987c8d6',11,'afunix-scorer-probe/files.json','afunix-scorer-probe/')}
for name,(expected,count,manifest,prefix) in archives.items():
 p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 with tarfile.open(p) as t:
  members=t.getmembers();m=json.load(t.extractfile(manifest))
  names={prefix+n for n in m['files']}|{manifest}
  assert len(members)==count==len(names) and {v.name for v in members}==names
  assert all(v.isfile() and not Path(v.name).is_absolute() and '..' not in Path(v.name).parts for v in members)
  t.extractall('tools')
root=Path('.');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tools=json.loads((root/'tools/repair-tools-files.json').read_text())
assert tools['native_counts']==dict(source=205,tools=128,normalizer=32,oracle=24)
for name,h in tools['files'].items():assert sha(root/'tools'/name)==h,name
base=root/'tools/afunix-scorer-probe'
for name,h in json.loads((base/'files.json').read_text())['files'].items():assert sha(base/name)==h,name
review=json.loads((base/'reviewed-sources.json').read_text())
assert review['code_commit']=='438af3529b40663aae7b8293080e09fda57f6a87'
assert review['source_fingerprint']=='101eb46fdd5cc493d8be3be11cb9d63313b3e94819bd9d2034b37a1f94e3a9a6'
for name,h in review['files'].items():assert sha(root/name)==h,name
print('PROSPECTIVE_GPU_ORACLE_V3_EXACT_BYTES_VERIFIED_NO_TRAINING')
PY
record=logs/gpu-oracle-v3-tools-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/execution-proof-native.sbatch > "$record"; cat "$record"; fi
record=logs/gpu-oracle-v3-normalizer-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/repair-normalizer-native.sbatch > "$record"; cat "$record"; fi
