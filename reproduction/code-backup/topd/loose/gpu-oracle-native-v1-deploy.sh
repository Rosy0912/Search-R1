set -euo pipefail
cd /dataset/shaoshitong/search-r1-rl/topd
test "$(git -C repo rev-parse HEAD)" = 438af3529b40663aae7b8293080e09fda57f6a87
test -z "$(git -C repo status --porcelain)"
python3 - <<'PY'
from pathlib import Path
import hashlib,tarfile
p=Path('gpu-oracle-native-v1.tar.gz')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e0d1ddad8a838b33abf3edf14959f19f99c65ae3e6e2a708cd366132bba79c76'
expected={'afunix-scorer-probe/'+n for n in ('original_gpu_oracle.py','test_original_gpu_oracle.py','probe_round_checks.py','test_probe_round_checks.py')}|{'gpu-oracle-native.sbatch','gpu-oracle-native-files.json'}
with tarfile.open(p) as t:
 members=t.getmembers()
 assert len(members)==len(expected) and {m.name for m in members}==expected
 assert all(m.isfile() for m in members)
 t.extractall('tools')
PY
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('tools');m=json.loads((root/'gpu-oracle-native-files.json').read_text())
assert m['native_expected_count']==24 and m['scope']=='CPU_FIXTURES_ONLY_NO_TRAINING'
for n,h in m['files'].items():
 p=root/n if n.endswith('.sbatch') else root/'afunix-scorer-probe'/n
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n
print('GPU_ORACLE_CPU_NATIVE_EXACT_BYTES_VERIFIED')
PY
record=logs/gpu-oracle-native-v1-submission.txt
if test -f "$record"; then cat "$record"; else sbatch --parsable tools/gpu-oracle-native.sbatch > "$record"; cat "$record"; fi
