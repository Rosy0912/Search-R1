"""Post-start ownership/lease proof; it deliberately does not apply idle thresholds."""
import json,math,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'strict-tools'))
from hf_activation_capture import capture_lease
from runtime_capture import process_identity
from training_evidence import OWN

def resolve_selected(selected,inventory):
 records=[]
 for device in selected:
  matches=[v for index,v in inventory.items() if device in (index,v['uuid'])]
  if len(matches)!=1:raise ValueError('Actual selected GPU mapping unavailable')
  records.append(matches[0])
 if len(records)!=8 or len({r['uuid'] for r in records})!=8:raise ValueError('Actual selected8 UUID mapping differs')
 return records

def assert_owned(process,job,uid):
 if process['uid']!=uid or process['uids']!=[uid]*4 or not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',process['cgroup']):raise ValueError('Current actual GPU process outside own UID/job')
 return process

def reassert_owned(preflight,expected_count=8):
 if expected_count not in (None,8):raise ValueError('Corrected post-start resource budget requires8')
 job=os.environ['SLURM_JOB_ID'];uid=os.getuid();selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
 inventory,_=preflight.gpu_snapshot();gpu=resolve_selected(selected,inventory);uuids={g['uuid'] for g in gpu}
 for g in gpu:
  values=[float(g[k]) for k in ('utilization','memory_used','memory_total')]
  if any(not math.isfinite(v) for v in values) or not 0<=values[0]<=100 or values[1]<0 or values[2]<=0 or values[1]>values[2]:raise ValueError('Actual post-start GPU inventory invalid')
 leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
 if os.environ.get('SEARCH_R1_GPU_PREFLIGHT')!='1' or len(leases)!=8 or {r['uuid'] for r in leases}!=uuids:raise ValueError('Post-start inherited selected8 lease differs')
 global_lease=capture_lease(9,OWN/'gpu-job.lock');records=[]
 for lease in leases:
  records.append(capture_lease(int(lease['fd']),Path('/tmp')/f'search-r1-gpu-leases-{uid}'/(lease['uuid']+'.lock')))
 raw_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=8)
 active=[]
 for line in raw_apps.splitlines():
  fields=[v.strip() for v in line.split(',')]
  if len(fields)!=2 or not fields[1].isdigit():raise ValueError('Unreadable actual GPU process list')
  if fields[0] in uuids:
   identity=assert_owned(process_identity(int(fields[1])),job,uid)
   if process_identity(int(fields[1]))['process_start_ticks']!=identity['process_start_ticks']:raise ValueError('Actual GPU process PID reused during capture')
   active.append(dict(identity,gpu_uuid=fields[0]))
 return {'job_id':job,'stage_sha256':os.environ['OPSD_CORRECTED_EVAL_STAGE_SHA256'],'process':assert_owned(process_identity(os.getpid()),job,uid),'selected_gpu_inventory':gpu,'global_lease':global_lease,'gpu_leases':records,'active_gpu_processes':active,'raw_compute_apps':raw_apps}
