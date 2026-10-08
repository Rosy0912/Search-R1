"""One confirmed owned job: read only final text metadata, no tensor assets."""
from pathlib import Path
import gzip,hashlib,json,shlex,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from remote import route
REMOTE=r'''
from pathlib import Path
import gzip,hashlib,json,socket,subprocess,sys,time
assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');job=3296
journal=json.loads((root/'instruct300/SINGLE_TP1_WINDOW_BOUNDARIES3295_ONCE.json.result.json').read_bytes())
assert journal['job_id']==str(job) and journal['status']=='SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE'
run=root/('runs/tree-probability-probe-'+str(job))
watch=root/('instruct300/probe-scheduler-observation-'+str(job)+'.jsonl')
assert watch.is_file()
last=json.loads(watch.read_text().splitlines()[-1]);assert last['terminal_state_observed'] is True
allowed=['initial-allocation.json','launch.json','torchrun-command.json','bundle.json','termination-cost.json','result.json','resources.jsonl','process-resources.jsonl','guard.log','probe.log']
paths=[run/name for name in allowed]+[root/('logs/tree-probability-probe-'+str(job)+'.out'),watch]
for rank in range(8):
 paths += [run/('gpu-records/rank-%02d/'%rank)/name for name in ['ready.json','sampling-params-transition.json']]
paths=[p for p in paths if p.is_file()]
assert len(paths)==len(set(paths)) and all(p.suffix in {'.json','.jsonl','.log','.out'} for p in paths)
inventory=[(p,p.stat().st_size) for p in paths]
assert all(n<=128*1024*1024 for p,n in inventory) and sum(n for p,n in inventory)<=512*1024*1024,'Audit reader bound; no runtime gate changed'
records=[]
for p,n in inventory:
 b=p.read_bytes();assert len(b)==n
 records.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode('utf-8')))
q=subprocess.run(['scontrol','show','job',str(job)],capture_output=True,text=True,timeout=15)
value=dict(host=socket.gethostname(),job_id=job,epoch=time.time(),mutation=False,tensor_assets_read=False,journal=journal,scheduler=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr),source_records=records,reader_scope='Owned final JSON/log metadata only, inventory bounded128MiB/file512MiB total; no PT/model/index/dataset assets')
sys.stdout.buffer.write(gzip.compress(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode(),compresslevel=5))
'''
output=Path(__file__).resolve().parents[1]/'outputs/tree-instruct-300/PPO_WINDOW_GPU3296_ACTUAL_RAW.json'
assert not output.exists()
result=subprocess.run(route('/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -B -c '+shlex.quote(REMOTE)),capture_output=True)
stderr=Path(__file__).resolve().parent/'tree-vllm-window-boundaries3295/ACTUAL3296_CAPTURE_STDERR.txt'
stderr.write_bytes(result.stderr)
if result.returncode:
 print(json.dumps(dict(status='CAPTURE_FAILED_NO_RUNTIME_ACCEPTANCE',returncode=result.returncode,stderr_path=str(stderr))))
 raise SystemExit(result.returncode)
data=gzip.decompress(result.stdout);value=json.loads(data)
with output.open('xb') as f:f.write(data)
print(json.dumps(dict(raw_path=str(output),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),record_count=len(value['source_records']),compressed_transfer_bytes=len(result.stdout),no_tensor_assets_read=True),sort_keys=True))
