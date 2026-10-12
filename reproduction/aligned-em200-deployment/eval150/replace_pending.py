import json,subprocess,time,uuid
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');h=r/'scheduler-head';old=json.loads((h/'eval150-submit-receipt.json').read_text());assert old['job']=='3373'
j=subprocess.check_output(['scontrol','show','job','3373'],text=True);assert 'JobState=PENDING' in j and 'BAPO_EVAL150:'+old['token'] in j
assert not Path(old['run']).exists()
assert not(h/'eval150-cohost-submit-intent.json').exists()
subprocess.run(['scancel','3373'],check=True)
j=subprocess.check_output(['scontrol','show','job','3373'],text=True);assert 'JobState=CANCELLED' in j
(h/'eval150-3373-pending-retirement.json').write_text(json.dumps(dict(time=time.time(),actual=j,reason='zero-start pending allocation replaced by explicit cohost CPU allocation')))
(r/'tools/eval150/eval150.sbatch').write_text('#!/bin/bash\n#SBATCH --job-name=bapo-eval150-final51713\n#SBATCH --partition=gpu\n#SBATCH --nodelist=h100-11\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=256G\n#SBATCH --time=08:00:00\n#SBATCH --no-requeue\n#SBATCH --export=NONE\n#SBATCH --chdir=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009\n#SBATCH --output=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/logs/slurm-eval150-%j.out\n#SBATCH --error=/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/logs/slurm-eval150-%j.err\nset -euo pipefail\nexport SLURM_EXPORT_ENV=ALL\nsrun --export=ALL --kill-on-bad-exit=1 bash /dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009/tools/eval150/worker.sh "${1:?token}"\n')
token=uuid.uuid4().hex;(h/'eval150-cohost-submit-intent.json').write_text(json.dumps(dict(time=time.time(),token=token,source='0a68dc4')))
p=subprocess.run(['sbatch','--parsable','--comment=BAPO_EVAL150:'+token,str(r/'tools/eval150/eval150.sbatch'),token],capture_output=True,text=True,timeout=45)
(h/'eval150-cohost-submit-response.json').write_text(json.dumps(dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)));assert p.returncode==0,p.stderr
job=p.stdout.strip().split(';')[0];assert job.isdigit()
x=dict(job=job,run=str(r/'runs'/('eval150-final51713-'+token+'-'+job)),token=token,source='0a68dc4',batch_size=512,gpus=8,physical_cohost=True,retired_pending_job='3373')
(h/'eval150-submit-receipt-original3373.json').write_text(json.dumps(old));(h/'eval150-submit-receipt.json').write_text(json.dumps(x));print(json.dumps(x))
