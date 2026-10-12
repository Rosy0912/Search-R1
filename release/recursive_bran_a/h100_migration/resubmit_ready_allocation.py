"""Submit the explicit prepared launch when the previous allocation has expired."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

b = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert not subprocess.check_output(['squeue','-h','-j','3387','-o','%i'], text=True).strip()
assert (b/'control/launch-ready.sh').exists()
assert not (b/'receipts/train-start.json').exists()
active = subprocess.check_output(['squeue','-h','-o','%j'], text=True).splitlines()
assert not any(name == 'rba-a-h100-run200' for name in active)
script = (b/'source-71d4e69/release/recursive_bran_a/h100_migration/allocation.sh').read_text()
result = subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=8','h100-5',
    'sbatch --parsable --job-name=rba-a-h100-run200 --nodelist=h100-11 --gres=gpu:h100:8 --exclusive --cpus-per-task=96 --mem=768G --time=7-00:00:00 --output=/dataset/shaoshitong/recursive-bran-a-h100-20261012/logs/slurm-%j.log'],
    input=script,capture_output=True,text=True,timeout=30)
proof = dict(time=time.time(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
    allocation_script_sha256=hashlib.sha256(script.encode()).hexdigest(),previous_job='3387',
    training_run=json.loads((b/'receipts/configuration.json').read_text())['run_id'])
(b/'receipts/accounting-restart-new-allocation.json').write_text(json.dumps(proof))
print(json.dumps(proof),flush=True)
raise SystemExit(result.returncode)
