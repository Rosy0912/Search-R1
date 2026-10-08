"""Read terminal native metadata once; no tensors downloaded or training mutation."""
import datetime,hashlib,json,subprocess
from pathlib import Path
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004');run=r/'runs/gigpo-e1-quality10-native-job3243';out=r/'logs/quality10-native3243-terminal-physical-payload-v1.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not out.exists(),'Existing payload: inspect, do not overwrite'
terminal=r/'logs/quality10-native-terminal-3243-raw-v1.json';snapshot=json.loads(terminal.read_text())
assert snapshot['fields']['JobId']=='3243' and snapshot['fields']['JobState']=='COMPLETED' and snapshot['fields']['ExitCode']=='0:0'
result=json.loads((run/'result.json').read_text());assert result['status']=='PASS' and result['actual_outer_updates']==0 and result['cleanup_residual_states']=={} and result['final_gpu']['processes']==[]
assert result['code_commit']=='f1ed53b45d6f5d4d54e9fb2ef81994ebfa2117e3' and result['restore_step']==25
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True);assert gpu.returncode==0 and not gpu.stdout.strip()
cgroups=[str(p) for p in Path('/sys/fs/cgroup').glob('*/slurm_h100-8/uid_0/job_3243')];assert not cgroups
queue=subprocess.check_output(['squeue','-h','-j','3243','-o','%i %T'],text=True);assert not queue.strip()
files=[]
for p in sorted(run.rglob('*')):
 if p.is_file() and p.suffix in {'.json','.jsonl','.log'}:files.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
for p in [terminal,r/'logs/quality10-native-3243.log',r/'logs/quality10-f1-native-submit-receipt-v1.json',r/'logs/quality10-native-selected25-f1ed53b-reviewed-binding-v1.json']:
 files.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
payload={'status':'ACTUAL_NATIVE3243_TERMINAL_PHYSICAL_METADATA_PASS_UNREVIEWED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3243','run':str(run),'result_path':str(run/'result.json'),'result_sha256':sha(run/'result.json'),'result':result,'terminal':snapshot,'gpu_compute_query':{'returncode':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr},'cgroups':cgroups,'queue':queue,'files':files,'scope':'Actual native GPU/state validation only, zero training outer; independent acceptance pending'}
with out.open('x') as f:f.write(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'files':len(files),'result_sha256':payload['result_sha256'],'actual_outer_updates':0}),flush=True)
