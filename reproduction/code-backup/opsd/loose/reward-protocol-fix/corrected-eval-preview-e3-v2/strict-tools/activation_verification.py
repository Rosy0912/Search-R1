"""Strict retained activation/source/lease/process/resources verification; CPU only."""
import hashlib,re,subprocess
from pathlib import Path
from training_evidence import loads,digest,COMMIT
from hf_activation_capture import validate_activation_audit,lock_records,validate_process_raw
from pre_stage_runtime import verify_resource_rows

def verify_lease(record,path,fd):
 p=Path(path).resolve();stat=p.stat()
 if set(record)!={'lock_path','fd','device','inode','raw_locks_before','raw_locks_after','matched_locks','reasserted_same_descriptor'}:raise ValueError('Exact lease raw schema required')
 if record['lock_path']!=str(p) or record['fd']!=fd or record['device']!=stat.st_dev or record['inode']!=stat.st_ino or record['reasserted_same_descriptor'] is not True:raise ValueError('Lease descriptor/inode mismatch')
 lock_records(record['raw_locks_before'],stat)
 if lock_records(record['raw_locks_after'],stat)!=record['matched_locks']:raise ValueError('Lease raw kernel lock record mismatch')

def verify_activation_evidence(paths,job,base):
 folder=paths['checker_manifest'].parent;manifest=loads(paths['checker_manifest'].read_text());audit=loads(paths['activation_result'].read_text())
 raw=paths['activation_raw'].read_text();marker='ACTUAL_HF500_ACTIVATION_AUDIT_JSON='
 lines=raw.splitlines()
 if len(lines)!=1 or not lines[0].startswith(marker) or loads(lines[0][len(marker):])!=audit:raise ValueError('Exact activation raw marker/file equality missing')
 uid_tokens=re.findall(r'\bUserId=[^\s(]+\(([0-9]+)\)',paths['slurm_terminal'].read_text())
 if uid_tokens!=[str(audit['uid'])]:raise ValueError('Activation UID does not match external Slurm user')
 capture=folder/'hf_activation_capture.py';launcher=folder/'hf_launcher.py';checker=paths['checker'];lock=base/'gpu-job.lock'
 validate_activation_audit(audit,job,checker,lock,manifest['file_sources']['hf_activation_capture.py'])
 if audit['launcher_source_sha256']!=manifest['file_sources']['hf_launcher.py']:raise ValueError('Actual launcher source differs from completed HF bundle')
 argv=audit['launch_argv']
 if len(argv)!=3 or argv[1:]!=['-u',str(checker.resolve())] or audit['checker']['cmdline']!=argv:raise ValueError('Immutable checker actual argv mismatch')
 if str(launcher.resolve()) not in audit['launcher']['cmdline']:raise ValueError('Actual launcher process is not pinned helper')
 ancestry=audit['checker'].get('ancestry',[])
 if not ancestry:raise ValueError('Checker actual ancestry capture missing')
 parent=audit['checker']['ppid'];visited=set()
 for record in ancestry:
  validate_process_raw(record,job,audit['uid'])
  if record['pid']!=parent or record['pid'] in visited:raise ValueError('Checker ancestry raw parent chain mismatch')
  visited.add(record['pid']);parent=record['ppid']
 if ancestry[-1]['pid']!=audit['launcher']['pid'] or ancestry[-1]['process_start_ticks']!=audit['launcher']['process_start_ticks']:raise ValueError('Actual checker is not descendant of pinned launcher')
 verify_lease(audit['global_lease_before_launch'],lock,9)
 selected=audit['selected_gpu_preflight'];devices=audit['selected_cuda_devices']
 if len(selected)!=1 or len(devices)!=1 or selected[0]['uuid']!=audit['gpu_uuid'] or devices[0] not in {str(selected[0]['index']),selected[0]['uuid']}:raise ValueError('Selected native preflight inventory identity mismatch')
 from math import isfinite
 for field in ('utilization','memory_used','memory_total'):
  if not isinstance(selected[0][field],(int,float)) or not isfinite(selected[0][field]):raise ValueError('Nonfinite native preflight occupancy')
 gpu=selected[0]
 if gpu['memory_total']<=0 or gpu['memory_used']<0 or not 0<=gpu['utilization']<=15 or 100*gpu['memory_used']/gpu['memory_total']>15:raise ValueError('Native preflight threshold exceeded')
 lease=audit['selected_gpu_lease'];pergpu=Path('/tmp')/f"search-r1-gpu-leases-{audit['uid']}"/(audit['gpu_uuid']+'.lock')
 if type(lease['fd']) is not int or lease['fd']<3 or lease['fd']==9:raise ValueError('Invalid single GPU inherited lease fd')
 verify_lease(lease,pergpu,lease['fd'])
 repo=base/'repo';preflight=repo/'search_r1/baselines/preflight.py';source=audit['preflight_source']
 if Path(source['path']).resolve()!=preflight.resolve() or digest(preflight)!=source['sha256']:raise ValueError('Actual preflight imported source path/SHA mismatch')
 original=subprocess.check_output(['git','-C',str(repo),'show',COMMIT+':search_r1/baselines/preflight.py'])
 if hashlib.sha256(original).hexdigest()!=source['sha256']:raise ValueError('Preflight differs from frozen training8e source')
 return verify_resource_rows([loads(line) for line in paths['resources'].read_text().splitlines()],job,audit['gpu_uuid'],str(audit['uid']),audit['checker_pid'],audit['checker_start_ticks'])
