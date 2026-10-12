import json,subprocess,os,sys,time,uuid
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');h=r/'scheduler-head';old=json.loads((h/'eval150-submit-receipt.json').read_text());assert old['job']=='3374'
s=subprocess.check_output(['scontrol','show','job','3374'],text=True);assert 'JobState=CANCELLED' in s
os.environ['SLURM_JOB_ID']='3374';sys.path.insert(0,str(r/'tools'));import resource_watch
code=resource_watch.cleanup(Path(old['run']));assert not code['remaining'] and not code['errors']
assert not(h/'eval150-fixed-submit-intent.json').exists()
token=uuid.uuid4().hex;(h/'eval150-fixed-submit-intent.json').write_text(json.dumps(dict(token=token,time=time.time(),source='299f829',oldjob=s)))
p=subprocess.run(['sbatch','--parsable','--comment=BAPO_EVAL150:'+token,str(r/'tools/eval150/eval150.sbatch'),token],capture_output=True,text=True,timeout=45)
(h/'eval150-fixed-submit-response.json').write_text(json.dumps(dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)));assert p.returncode==0,p.stderr
job=p.stdout.strip().split(';')[0];assert job.isdigit()
x=dict(job=job,run=str(r/'runs'/('eval150-final51713-'+token+'-'+job)),token=token,source='299f829',batch_size=512,gpus=8,physical_cohost=True,retired_startup_job='3374')
(h/'eval150-submit-receipt-startup3374.json').write_text(json.dumps(old));(h/'eval150-submit-receipt.json').write_text(json.dumps(x));print(json.dumps(x))
