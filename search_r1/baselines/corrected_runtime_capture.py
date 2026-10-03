"""Opt-in read-only corrected evaluation worker/source and call evidence.
No model forward, optimizer dispatch, parameter gather or collective.
"""
import os,re,socket,sys
from pathlib import Path

WORKER_MODULES=('verl.workers.fsdp_workers','verl.workers.rollout.vllm_rollout.vllm_rollout','search_r1.baselines.actor','search_r1.baselines.distillation','search_r1.baselines.corrected_runtime_capture','verl.single_controller.base.worker','verl.single_controller.base.decorator')

def process_identity(pid,proc=Path('/proc')):
 root=proc/str(pid);raw_stat=(root/'stat').read_text();fields=raw_stat.rsplit(')',1)[1].split();status=(root/'status').read_text();ids=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
 uids=ids['Uid'].split()
 if len(uids)!=4 or not all(value.isdigit() for value in uids):raise ValueError('Invalid actual process UID status')
 cgroup=(root/'cgroup').read_text();cmdline=(root/'cmdline').read_bytes()
 return dict(pid=pid,ppid=int(fields[1]),process_start_ticks=fields[19],uid=int(uids[0]),uids=[int(value) for value in uids],cgroup=cgroup,cmdline=[part.decode() for part in cmdline.split(b'\0') if part],raw_stat=raw_stat,raw_status=status,raw_cgroup=cgroup,raw_cmdline_hex=cmdline.hex())

def imported_source_records(names,repo,modules=None):
 from .corrected_evaluation import digest
 modules=sys.modules if modules is None else modules;repo=Path(repo).resolve();result={}
 for name in names:
  module=modules.get(name)
  if module is None or not getattr(module,'__file__',None):raise ValueError('Required worker module not actually imported')
  path=Path(module.__file__).resolve()
  if not path.is_relative_to(repo):raise ValueError('Worker import outside corrected source')
  result[name]=dict(path=str(path),sha256=digest(path))
 return result

def worker_import_callback(worker,probe):
 import numpy as np,torch
 from verl import DataProto
 from .corrected_evaluation import digest
 if len(probe)!=1:raise ValueError('Exactly one audit probe per worker required')
 spec=probe.meta_info['corrected_audit_spec'];repo=Path(spec['eval_repo']).resolve();job=os.environ['SLURM_JOB_ID'];stage=os.environ['OPSD_CORRECTED_EVAL_STAGE_SHA256']
 if stage!=spec['stage_sha256'] or job!=spec['job_id']:raise ValueError('Worker actual job/stage env mismatch')
 own=process_identity(os.getpid());imports=imported_source_records(WORKER_MODULES,repo)
 if not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',own['cgroup']):raise ValueError('Worker outside own Slurm cgroup')
 device=torch.cuda.current_device();properties=torch.cuda.get_device_properties(device);uuid=getattr(properties,'uuid',None)
 if uuid is None:raise ValueError('Actual CUDA device UUID unavailable; independent native mapping required')
 record=dict(rank=worker.rank,world_size=worker.world_size,pid=os.getpid(),uid=os.getuid(),node=socket.gethostname(),job_id=job,stage_sha256=stage,source_imports=imports,process=own,worker_type=type(worker).__module__+'.'+type(worker).__qualname__,rollout_type=type(worker.rollout).__module__+'.'+type(worker.rollout).__qualname__,device_index=device,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu_uuid=str(uuid),callback_sha256=digest(__file__))
 return DataProto.from_dict(tensors={'audit_probe':torch.zeros((1,1),dtype=torch.int64)},non_tensors={'worker_import_audit':np.array([record],dtype=object)})

def validate_worker_rows(rows,spec):
 from .corrected_evaluation import digest
 if len(rows)!=8 or {row['rank'] for row in rows}!=set(range(8)):raise ValueError('Exactly8 unique actual worker ranks required')
 repo=Path(spec['eval_repo']).resolve();pids=set();uuids=set()
 for row in rows:
  if row['world_size']!=8 or row['node'].split('.')[0]!='h100-10' or row['job_id']!=spec['job_id'] or row['stage_sha256']!=spec['stage_sha256']:raise ValueError('Worker rank/node/job/stage mismatch')
  process=row['process']
  if row['pid'] in pids or row['pid']!=process['pid'] or row['uid']!=spec['uid'] or process['uid']!=row['uid'] or not str(process['process_start_ticks']).isdigit():raise ValueError('Actual worker PID/start/UID mismatch')
  pids.add(row['pid'])
  if not re.search(r'(?:^|/)job_'+re.escape(spec['job_id'])+r'(?:/|$)',process['cgroup']):raise ValueError('Worker ownership cgroup mismatch')
  if process['cgroup']!=process['raw_cgroup']:raise ValueError('Worker cgroup not actual raw capture')
  fields=process['raw_stat'].rsplit(')',1)[1].split()
  if fields[19]!=process['process_start_ticks'] or int(fields[1])!=process['ppid'] or not process['raw_stat'].startswith(str(row['pid'])+' '):raise ValueError('Worker stat raw identity mismatch')
  status=dict(line.split(':',1) for line in process['raw_status'].splitlines() if ':' in line)
  if [int(x) for x in status['Uid'].split()]!=process['uids'] or process['uids'][0]!=row['uid']:raise ValueError('Worker UID not actual raw status')
  if [p.decode() for p in bytes.fromhex(process['raw_cmdline_hex']).split(b'\0') if p]!=process['cmdline'] or not process['cmdline']:raise ValueError('Worker command not actual raw capture')
  if row['worker_type']!='verl.workers.fsdp_workers.ActorRolloutRefWorker' or row['rollout_type']!='verl.workers.rollout.vllm_rollout.vllm_rollout.vLLMRollout':raise ValueError('Unexpected actual worker/rollout class')
  if row['device_index']<0 or not row['cuda_visible_devices'] or not row['gpu_uuid']:raise ValueError('Missing CUDA identity')
  uuids.add(row['gpu_uuid'])
  if row['callback_sha256']!=spec['source_files']['search_r1/baselines/corrected_runtime_capture.py']:raise ValueError('Worker callback source differs')
  if set(row['source_imports'])!=set(WORKER_MODULES):raise ValueError('Actual worker imports incomplete')
  for module,record in row['source_imports'].items():
   relative=module.replace('.','/')+'.py';wanted=repo/relative
   if Path(record['path']).resolve()!=wanted or record['sha256']!=spec['source_files'][relative] or digest(wanted)!=record['sha256']:raise ValueError('Actual worker source path/SHA mismatch')
 if len(uuids)!=8:raise ValueError('Actual worker GPU UUIDs not eight unique devices')
 return rows

def capture_workers(trainer,provenance):
 import numpy as np,torch
 from verl import DataProto
 from .corrected_evaluation import canonical
 job=os.environ.get('SLURM_JOB_ID','')
 if not re.fullmatch('[0-9]+',job):raise ValueError('Corrected capture requires actual own Slurm job')
 if trainer.actor_rollout_wg.world_size!=8:raise ValueError('Corrected eval requires eight workers')
 spec=dict(eval_repo=str(Path(__file__).resolve().parents[2]),source_files=provenance['source_files'],job_id=job,stage_sha256=provenance['reviewed_stage_sha256'],uid=os.getuid())
 probe=DataProto.from_dict(tensors={'audit_probe':torch.zeros((8,1),dtype=torch.int64)},meta_info={'corrected_audit_spec':spec})
 output=trainer.actor_rollout_wg.execute_with_func_generator(worker_import_callback,probe)
 if not isinstance(output,DataProto) or len(output)!=8:raise ValueError('Worker RPC result is not eight native DataProto rows')
 rows=canonical(output.non_tensor_batch['worker_import_audit']);validate_worker_rows(rows,spec)
 return dict(spec=spec,rows=rows)

class GenerationCallGuard:
 def __init__(self,group,worker_audit):
  self.group=group;self.original=group.generate_sequences;self.calls=[];self.rejected_calls=0;self.worker_audit=worker_audit
  group.generate_sequences=self.generate
 def generate(self,probe,*args,**kwargs):
  meta=probe.meta_info
  if meta.get('do_sample') is not False or meta.get('validate') is not True:
   self.rejected_calls+=1;raise RuntimeError('Corrected generation requires actual greedy validation calls')
  if not 8<=len(probe)<=64 or len(probe)%8:
   self.rejected_calls+=1;raise RuntimeError('Corrected generation actual padded batch budget invalid')
  output=self.original(probe,*args,**kwargs)
  from verl import DataProto
  if not isinstance(output,DataProto) or len(output)!=len(probe):
   self.rejected_calls+=1;raise RuntimeError('Corrected greedy generation output row budget invalid')
  self.calls.append(dict(sequence=len(self.calls)+1,rows=len(probe),do_sample=False,validate=True))
  return output
 def record(self):
  from .corrected_evaluation import digest
  if self.group.generate_sequences!=self.generate or self.rejected_calls or not self.calls:raise ValueError('Generation guard replaced/rejected/unused')
  return dict(source_sha256=digest(__file__),worker_audit=self.worker_audit,generation_calls=self.calls,rejected_calls=0)

def validate_completed_runtime(audit,provenance):
 if audit.get('source_sha256')!=provenance['source_files']['search_r1/baselines/corrected_runtime_capture.py'] or audit.get('rejected_calls')!=0:raise ValueError('Completed runtime source/call audit invalid')
 spec=audit['worker_audit']['spec']
 if spec['source_files']!=provenance['source_files'] or spec['stage_sha256']!=provenance['reviewed_stage_sha256']:raise ValueError('Runtime differs from externally reviewed stage/source')
 validate_worker_rows(audit['worker_audit']['rows'],spec)
 calls=audit['generation_calls']
 if not calls or [row['sequence'] for row in calls]!=list(range(1,len(calls)+1)) or any(type(row['rows']) is not int or not 8<=row['rows']<=64 or row['rows']%8 or row['do_sample'] is not False or row['validate'] is not True for row in calls):raise ValueError('Actual generation call sequence/budget invalid')
