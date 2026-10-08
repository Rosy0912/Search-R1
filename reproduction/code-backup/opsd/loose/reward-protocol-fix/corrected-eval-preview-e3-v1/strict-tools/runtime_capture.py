"""Read-only capture primitives; NOT integrated with launcher/workers yet.
No model forward/gather/collective/optimizer, and no GPU allocation.
"""
import hashlib,os,re,socket,sys
from pathlib import Path
from training_evidence import digest

def process_identity(pid,proc=Path('/proc')):
 root=proc/str(pid);raw_stat=(root/'stat').read_text();fields=raw_stat.rsplit(')',1)[1].split();status=(root/'status').read_text();ids=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
 uids=ids['Uid'].split()
 if len(uids)!=4 or not all(value.isdigit() for value in uids):raise ValueError('Invalid actual process UID status')
 cgroup=(root/'cgroup').read_text();cmdline=(root/'cmdline').read_bytes()
 return dict(pid=pid,ppid=int(fields[1]),process_start_ticks=fields[19],uid=int(uids[0]),uids=[int(value) for value in uids],cgroup=cgroup,cmdline=[part.decode() for part in cmdline.split(b'\0') if part],raw_stat=raw_stat,raw_status=status,raw_cgroup=cgroup,raw_cmdline_hex=cmdline.hex())

def checker_identity(launcher,checker,job,proc=Path('/proc')):
 """Find an actual descendant running exact immutable checker, not any python PID."""
 parent=process_identity(launcher,proc);records={}
 for path in proc.iterdir():
  if not path.name.isdigit():continue
  try:records[int(path.name)]=process_identity(int(path.name),proc)
  except (OSError,ValueError,IndexError):continue
 wanted=Path(checker).resolve();found=[]
 for pid,record in records.items():
  if str(wanted) not in record['cmdline']:continue
  ancestor=record['ppid'];visited=set()
  while ancestor in records and ancestor not in visited and ancestor!=launcher:
   visited.add(ancestor);ancestor=records[ancestor]['ppid']
  if ancestor!=launcher:continue
  if not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',record['cgroup']):raise ValueError('Checker descendant outside own job')
  record['ancestry']=[];ancestor=record['ppid'];visited=set()
  while ancestor in records and ancestor not in visited:
   visited.add(ancestor);record['ancestry'].append(records[ancestor])
   if ancestor==launcher:break
   ancestor=records[ancestor]['ppid']
  found.append(record)
 if len(found)!=1:raise ValueError('Exactly one actual immutable checker descendant required')
 return dict(launcher=parent,checker=found[0],checker_sha256=digest(wanted))

def imported_source_records(names,repo,modules=None):
 """Actual sys.modules imports only; byte labels or fresh substitute imports are rejected."""
 modules=sys.modules if modules is None else modules;repo=Path(repo).resolve();result={}
 for name in names:
  module=modules.get(name)
  if module is None or not getattr(module,'__file__',None):raise ValueError('Required worker module not actually imported')
  path=Path(module.__file__).resolve()
  if not path.is_relative_to(repo):raise ValueError('Worker actual import outside corrected source')
  result[name]=dict(path=str(path),sha256=digest(path))
 return result

def worker_import_callback(worker,probe):
 """Suitable for existing DP_COMPUTE_PROTO_WITH_FUNC RPC after init_workers.
Future integration must pin this module and compare all8 records before generation.
"""
 import numpy as np,torch
 from verl import DataProto
 if len(probe)!=1:raise ValueError('Exactly one audit probe row per native worker')
 repo=os.environ['OPSD_CORRECTED_EVAL_REPO'];job=os.environ['SLURM_JOB_ID'];stage=os.environ['OPSD_CORRECTED_EVAL_STAGE_SHA256']
 names=('verl.workers.fsdp_workers','verl.workers.rollout.vllm_rollout.vllm_rollout','search_r1.baselines.actor','search_r1.baselines.distillation')
 source=imported_source_records(names,repo)
 own=process_identity(os.getpid())
 if not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',own['cgroup']):raise ValueError('Actual worker outside own Slurm job')
 rollout=worker.rollout
 record=dict(rank=worker.rank,world_size=worker.world_size,pid=os.getpid(),uid=os.getuid(),node=socket.gethostname(),job_id=job,stage_sha256=stage,source_imports=source,process=own,worker_type=type(worker).__module__+'.'+type(worker).__qualname__,rollout_type=type(rollout).__module__+'.'+type(rollout).__qualname__,device_index=torch.cuda.current_device(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],callback_sha256=digest(__file__))
 return DataProto.from_dict(tensors={'audit_probe':torch.zeros((1,1),dtype=torch.int64)},non_tensors={'worker_import_audit':np.array([record],dtype=object)})
