"""Actual teacher eight-rank KV release, then actual student eight-rank ACKs."""
import hashlib,json,os,pathlib,time
from pipeline import atomic

def require_teacher_release(data):
 import torch
 step=int(data.meta_info['topd_outer_step']);root=pathlib.Path(os.environ['TOPD_TEACHER_DIRECTORY'])
 path=root/('round-%04d.json'%step);raw=path.read_bytes();release=json.loads(raw)
 if release['status']!='ACTUAL_TP4_KV_RELEASE_ACK_BEFORE_STUDENT_UPDATE' or release['outer_step']!=step or release['job']!=os.environ['SLURM_JOB_ID']:raise RuntimeError('Teacher release job/round identity mismatch')
 rows=release['released_workers']
 if len(rows)!=4 or {r['rank'] for r in rows}!=set(range(4)) or {r['uuid'] for r in rows}!=set(release['leased_uuids']) or any(not r['cache_engine_absent'] or not r['gpu_cache_absent'] or r['requires_grad_parameters'] or r['model_training'] for r in rows):raise RuntimeError('Actual frozen teacher TP4 KV release incomplete')
 rank=torch.distributed.get_rank();target=pathlib.Path(os.environ['R1_RESOURCE_AUDIT_DIRECTORY'])
 receipt={'rank':rank,'outer_step':step,'pid':os.getpid(),'job':os.environ['SLURM_JOB_ID'],'teacher_release_path':str(path),'teacher_release_sha256':hashlib.sha256(raw).hexdigest(),'acknowledged_epoch':time.time()}
 atomic(target/('student-release-ack-rank%d-outer%d.json'%(rank,step)),receipt)
 torch.distributed.barrier()
 for index in range(4):
  ack=json.loads((target/('student-release-ack-rank%d-outer%d.json'%(index,step))).read_text())
  if ack['rank']!=index or ack['outer_step']!=step or ack['job']!=receipt['job'] or ack['teacher_release_sha256']!=receipt['teacher_release_sha256']:raise RuntimeError('Student ALL4 release ACK differs')
