set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/bapo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$R/env/bin/python" - <<'PY'
import pathlib,json,hashlib,shutil
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo/logs')
p=r/'bapo-hardware-2910-acceptance.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='eb1353e4b6b0dd264cfe856765820c1fa24d4f16f2c79ebdc3614a49fb0613a5'
x=json.loads(p.read_text());assert len(x['evidence'])==45
active=r/'hardware-acceptance.json'
old=r/'hardware-acceptance-b6ecbeb-2880.json'
assert not old.exists()
assert json.loads(active.read_text())['slurm_job_id']=='2880'
shutil.copy2(active,old);shutil.copy2(p,active)
PY
"$R/env/bin/python" "$R/tools/verify_hardware_receipt.py"
