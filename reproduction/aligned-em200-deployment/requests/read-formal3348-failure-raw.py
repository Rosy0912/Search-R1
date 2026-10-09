"""Bound read-only startup-failure archive; no asset/model read or mutation."""
from pathlib import Path
import base64,json,hashlib,os,stat,subprocess,datetime
R=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');RUN=R/'runs/formal-aligned-em200-66cb18840a404586ab3333f8c393b8b7-3348'
if os.uname().nodename!='h100-5' or RUN.resolve()!=RUN:raise RuntimeError('wrong node/run')
files=sorted(p for p in RUN.rglob('*') if not p.is_dir())
files += [R/'scheduler-head'/n for n in ['formal-submit-intent.json','formal-submit-receipt.json','formal-sbatch-response.json']]
files += [R/'logs'/n for n in ['slurm-formal-3348.out','slurm-formal-3348.err']]
files += [R/'tools/deployment'/n for n in ['formal_admission.py','formal.sbatch','formal_worker.sh','formal_inner.sh','formal_driver.py','formal_service_proof.py','ray_cli_compat.py','bind_retrieval_proof.py','submit_formal.py']]
records=[];total=0
for p in files:
 if p.resolve()!=p or p.is_symlink() or not stat.S_ISREG(p.stat().st_mode):raise RuntimeError('nonordinary evidence '+str(p))
 before=p.stat()
 if before.st_size>16*1024*1024:raise RuntimeError('unexpected large startup file')
 raw=p.read_bytes();after=p.stat();keys=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 if keys(before)!=keys(after):raise RuntimeError('failure evidence changing '+str(p))
 total+=len(raw)
 if total>64*1024*1024:raise RuntimeError('startup archive bound exceeded')
 records.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})
queries={}
for name,args in [('slurm',['scontrol','show','job','3348','-o']),('queue',['squeue','-h','-o','%i|%T|%R|%k']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader']),('interfaces',['ip','-j','-4','address','show','up'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);queries[name]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3348','run':str(RUN),'files':records,'total_bytes':total,'queries':queries,'scope':'RAW_STARTUP_FAILURE_NO_TERMINAL_SLURM_CLAIM_NO_RESUME_OR_TRAINING_ACCEPTANCE'}))
