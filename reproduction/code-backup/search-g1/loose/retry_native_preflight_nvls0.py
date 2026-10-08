import sys,subprocess,shlex,json,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
code=r"""import subprocess,json,os,uuid,hashlib,time;from pathlib import Path
base=Path('/dataset/shaoshitong/Search-G1-common8');prep=base/'preparation'
assert 'JobState=FAILED' in subprocess.check_output(['scontrol','show','job','3235','-o'],text=True)
full=subprocess.check_output(['scontrol','show','job','3238','-o'],text=True)
assert 'JobState=PENDING' in full and 'DependencyNeverSatisfied' in full
assert not (base/'logs/run-manifest.json').exists()
claim=prep/'nvls0-retry-submit-claim.json'
with claim.open('x') as f:json.dump(dict(previous_preflight='3235',full_job='3238',unix=time.time()),f);f.flush();os.fsync(f.fileno())
old=prep/'eight-rank-mechanics-reviewed-5c2d521448bd4c06a14ffe0c613a6933';new=prep/('eight-rank-mechanics-nvls0-'+uuid.uuid4().hex);new.mkdir()
files=['reproduction/common8/gpu_preflight.py','reproduction/common8/scheduler.py','search_g1/numerics.py','search_g1/gradient_clipping.py'];hashes={}
for n in files:
 p=new/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((old/'source'/n).read_bytes());hashes[n]=hashlib.sha256(p.read_bytes()).hexdigest()
script=(old/'job.sh').read_text().replace(str(old),str(new)).replace('export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1','export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1\nexport NCCL_NVLS_ENABLE=0 NCCL_DEBUG=INFO')
script=script.replace('#SBATCH --job-name=search-g1-eight-rank-check','#SBATCH --job-name=search-g1-eight-rank-nvls0\n#SBATCH --output='+str(new/'slurm-%j.log'))
(new/'job.sh').write_text(script)
r=subprocess.run(['sbatch','--parsable',str(new/'job.sh')],capture_output=True,text=True);assert r.returncode==0,r.stderr
job=r.stdout.strip().split(';')[0];assert job.isdigit()
proof=dict(job_id=job,previous_failed_job='3235',preflight_dir=str(new),source_sha256=hashes,nccl_nvls_enable=0,slurm_stdout=str(new/('slurm-'+job+'.log')),submitted_unix=time.time(),scope='Same eight-rank mechanics, new native job; only task-local NVLS offload disabled, no service modification')
with (prep/'nvls0-retry-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
r=subprocess.run(['scontrol','update','JobId=3238','Dependency=afterok:'+job],capture_output=True,text=True)
proof['dependency_update_exit_code']=r.returncode;proof['dependency_update_stdout']=r.stdout;proof['dependency_update_stderr']=r.stderr
(prep/'nvls0-retry-dependency-update.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof));assert r.returncode==0,r.stderr
"""
compile(code,'native-retry','exec')
r=subprocess.run(control.route('h100-8','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
p=json.loads(r.stdout);Path('/Users/shaoshitong/Documents/Search-G1/preparation/nvls0-retry-launch.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))
