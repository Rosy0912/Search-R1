import json,sys,time,hashlib,gzip,base64,subprocess
from pathlib import Path
job=sys.argv[1];assert job.isdigit();root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs'/('tree-policy-safety-native-'+job)
files={}
for p in run.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.ready'):
  data=p.read_bytes();files[str(p.relative_to(run))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),gzip_base64=base64.b64encode(gzip.compress(data)).decode())
commands={}
for args in (['scontrol','show','job',job],['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES','-P'],['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']):
 p=subprocess.run(args,capture_output=True,text=True);commands[' '.join(args)]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
background={}
for pid in (1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446):
 p=Path('/proc')/str(pid)/'stat';background[str(pid)]=p.read_text() if p.exists() else None
print(json.dumps(dict(job_id=job,epoch=time.time(),files=files,commands=commands,background_proc_stat=background)))
