"""Read-only original HF bytes + postHF fullprefix/process/lease replay. No model load."""
import os,sys,json,hashlib,time,subprocess
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');RUN=ROOT/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315';BASE=RUN/'logs/first5-evidence-v4-20261007-a2';TOOLS=ROOT/'first5-evidence-tools-3315-v4'
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2';HF_SHA='e80263830380db2d728bd1aa95723b3bdb26aa4d29fa53d06c62dcb4c210b204';COLLECTION_SHA='a08423cda2f9f6b1baafebf6d0c586b60ff50fe9c6937ad2f2c19ef52d7bff09'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
need(sys.platform=='linux'and sys.prefix==str(ROOT/'env')and os.environ.get('CUDA_VISIBLE_DEVICES')=='','own CPU-only env')
need(digest(BASE/'hf.json')==HF_SHA and digest(BASE/'collection.json')==COLLECTION_SHA,'original HF/collection changed')
sys.path[:0]=[str(TOOLS),str(ROOT/'repo')]
from first5_helpers import process_proof,lease_proof,scheduler_proof,verify_snapshot_prefix
from guard_snapshot import replay_snapshot
h=json.loads((BASE/'hf.json').read_text());c=json.loads((BASE/'collection.json').read_text())
resource=h['resources_after_cpu_hf'];replayed=replay_snapshot(Path(resource['raw_path']),3315,str(RUN),SOURCE,ROOT/'tools/resource_watch.py',resource['snapshot_binding'])
need(replayed==resource and digest(resource['raw_path'])==h['resource_journal_after_cpu_hf']['sha256'],'postHF resource raw/fullprefix changed')
need(scheduler_proof(Path(h['slurm_after_hf']['path']),3315)==h['allocation_after_cpu_hf'],'actual saved postHF Slurm differs')
current=subprocess.check_output(['scontrol','show','job','3315'],text=True)
for token in ('JobId=3315','JobState=RUNNING','Restarts=0','Requeue=0','NodeList=h100-5','NumNodes=1'):
 need(token in current.split(),'live source job state differs')
driver=process_proof(760122,RUN);need(driver['start_ticks']==c['driver_process']['start_ticks']==h['before_process']['start_ticks']==h['after_process']['start_ticks'],'driver PID reused')
lease=lease_proof(driver,resource['allocated_uuids'],RUN)
need(lease['guard']['pid']==c['leases']['guard']['pid'] and lease['guard']['start_ticks']==c['leases']['guard']['start_ticks'],'resource guard PID epoch changed')
need(not (RUN/'logs/fresh200-step5-independent-release.json').exists(),'already released actualpause')
# Only small JSON evidence, never PT/model/native digest replay in this reader.
read_count=0
for item in c['evidence']:
 path=Path(item['path'])
 if path.suffix in ('.json','.py'):
  need(digest(path)==item['sha256'],'small original evidence changed '+path.name);read_count+=1
need(digest(BASE/'hf.json')==HF_SHA,'HF report mutated during audit')
print(json.dumps(dict(status='PASS_ACTUAL_READONLY_POSTHF_SMALL_EVIDENCE_RESOURCE_PID_LEASE_REPLAY_NOT_RELEASE',source_commit=SOURCE,job_id=3315,hf_sha256=HF_SHA,collection_sha256=COLLECTION_SHA,
 driver_pid=760122,driver_birth=driver['start_ticks'],resource_guard_pid=lease['guard']['pid'],resource_guard_birth=lease['guard']['start_ticks'],read_only_exclusive_flock_records=len(lease['leases']),small_json_code_files_rehashed=read_count,
 resource_after_hf_sha256=resource['raw_sha256'],resource_counts={k:resource[k]for k in ('samples','freeze_events','recovery_samples','safe_resume_events','abort_events')},
 same_process_live=True,unreleased=True,hf_loading_or_forward_executed=False,large_model_files_read=False,release_authorized=False),allow_nan=False))
