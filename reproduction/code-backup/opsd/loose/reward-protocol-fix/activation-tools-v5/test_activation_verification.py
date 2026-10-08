"""Synthetic CPU activation-schema fixtures only; no Slurm completion/GPU run."""
import copy,json,os,shutil
from pathlib import Path
import pytest
from test_pre_stage_runtime import activation
from activation_verification import verify_activation_evidence
from training_evidence import digest
from hf_activation_capture import emit_activation
HERE=Path(__file__).parent
JOB='CPU_SCHEMA_FIXTURE'

def process(pid,ppid,command):
 uid=os.getuid();fields=['S',str(ppid)]+['0']*18;fields[19]='123';cg=f'0::/slurm/job_{JOB}/step_batch\n';cmd=b'\0'.join(part.encode() for part in command)+b'\0'
 return dict(pid=pid,ppid=ppid,process_start_ticks='123',uid=uid,uids=[uid]*4,cgroup=cg,cmdline=command,raw_stat=f'{pid} (python) '+' '.join(fields),raw_status='Uid: '+' '.join([str(uid)]*4)+'\n',raw_cgroup=cg,raw_cmdline_hex=cmd.hex())

def lease(path,fd):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.touch();s=p.stat();line=f'1: FLOCK ADVISORY WRITE 100 {os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino} 0 EOF'
 return dict(lock_path=str(p.resolve()),fd=fd,device=s.st_dev,inode=s.st_ino,raw_locks_before=line+'\n',raw_locks_after=line+'\n',matched_locks=[line],reasserted_same_descriptor=True)

@pytest.fixture
def evidence(tmp_path):
 repo=HERE.parent/'integration-candidate-9fb-v3/repo';base=tmp_path/'own';base.mkdir();(base/'repo').symlink_to(repo.resolve(),target_is_directory=True)
 folder,manifest,_=activation(tmp_path);checker=folder/'checkpoint_reload.py';launcher=folder/'hf_launcher.py';uid=os.getuid();uuid='GPU-CPU-schema-'+tmp_path.name;gpu_lock=Path('/tmp')/f'search-r1-gpu-leases-{uid}'/(uuid+'.lock')
 parent=process(100,1,['python',str(launcher.resolve())]);child=process(101,100,['python','-u',str(checker.resolve())]);child['ancestry']=[copy.deepcopy(parent)];global_lease=lease(base/'gpu-job.lock',9);per_gpu=lease(gpu_lock,3)
 audit=dict(schema='actual-hf500-activation-v1',job_id=JOB,node='h100-10',uid=uid,gpu_uuid=uuid,checker_pid=101,checker_start_ticks='123',own_cgroup=parent['cgroup'],capture_process=copy.deepcopy(parent),launcher=parent,checker=child,checker_sha256=digest(checker),lease=copy.deepcopy(global_lease),capture_source_sha256=digest(folder/'hf_activation_capture.py'),launcher_source_sha256=digest(launcher),launch_argv=child['cmdline'],selected_gpu_preflight=[dict(index='0',uuid=uuid,utilization=0,memory_used=0,memory_total=100)],selected_gpu_processes='',selected_cuda_devices=['0'],selected_gpu_lease=per_gpu,global_lease_before_launch=global_lease,preflight_source=dict(path=str((base/'repo/search_r1/baselines/preflight.py').resolve()),sha256=digest(base/'repo/search_r1/baselines/preflight.py')))
 run=base/'cpu-schema-records';run.mkdir();emit_activation(audit,run)
 terminal=run/'uid-only.txt';terminal.write_text(f'UserId=CPU_SCHEMA_FIXTURE({uid})\n') # Not a fabricated COMPLETED terminal.
 proc=dict(pid=101,uid=str(uid),process_start_ticks='123',in_slurm_job=True)
 row=dict(time='2026-10-04T00:00:00+00:00',job_id=JOB,nominal_sleep_seconds=2,gpus=[dict(index='0',uuid=uuid,util=1,used_mib=20,total_mib=100)],gpu_processes=[dict(proc,uuid=uuid,gpu_used_mib='20')],job_processes=[proc],cgroup={f'/sys/fs/cgroup/slurm/job_{JOB}/memory.current':'123'})
 resources=run/'resources.jsonl';resources.write_text(json.dumps(row)+'\n')
 paths=dict(checker_manifest=manifest,checker=checker,job_script=folder/'reload_job.sh',activation_result=run/'hf-activation.json',activation_raw=run/'hf-activation.raw.log',slurm_terminal=terminal,resources=resources)
 yield base,paths,audit
 gpu_lock.unlink(missing_ok=True)

def rewrite(paths,audit):
 raw=json.dumps(audit,sort_keys=True);paths['activation_result'].write_text(raw+'\n');paths['activation_raw'].write_text('ACTUAL_HF500_ACTIVATION_AUDIT_JSON='+raw+'\n')

def test_complete_leaf_schema_only_no_gpu_or_scheduler_proof(evidence):
 base,paths,audit=evidence;assert verify_activation_evidence(paths,JOB,base)['samples']==1

@pytest.mark.parametrize('mutation',['raw','uid','launcher_source','capture_source','argv','parent','ancestry','lease_fd','lease_inode','lease_raw','per_gpu_fd','uuid','inventory_index','threshold','nan','preflight_source','resource_pid','resource_uid'])
def test_activation_raw_source_inventory_lease_ownership_tampering(evidence,mutation):
 base,paths,audit=evidence
 if mutation=='raw':paths['activation_raw'].write_text('wrong marker\n')
 if mutation=='uid':audit['uid']+=1
 if mutation=='launcher_source':audit['launcher_source_sha256']='0'*64
 if mutation=='capture_source':audit['capture_source_sha256']='0'*64
 if mutation=='argv':audit['launch_argv']=['python','fake.py']
 if mutation=='parent':audit['checker']['ppid']=999
 if mutation=='ancestry':audit['checker']['ancestry']=[]
 if mutation=='lease_fd':audit['lease']['fd']=8
 if mutation=='lease_inode':audit['lease']['inode']+=1
 if mutation=='lease_raw':audit['lease']['raw_locks_after']=''
 if mutation=='per_gpu_fd':audit['selected_gpu_lease']['fd']=9
 if mutation=='uuid':audit['selected_gpu_preflight'][0]['uuid']='other'
 if mutation=='inventory_index':audit['selected_cuda_devices']=['wrong']
 if mutation=='threshold':audit['selected_gpu_preflight'][0]['memory_used']=16
 if mutation=='nan':audit['selected_gpu_preflight'][0]['memory_used']=float('nan')
 if mutation=='preflight_source':audit['preflight_source']['sha256']='0'*64
 if mutation.startswith('resource_'):
  row=json.loads(paths['resources'].read_text());row['gpu_processes'][0]['pid']=999 if mutation=='resource_pid' else 101
  if mutation=='resource_uid':row['gpu_processes'][0]['uid']='foreign'
  paths['resources'].write_text(json.dumps(row)+'\n')
 if mutation not in {'raw','resource_pid','resource_uid'}:rewrite(paths,audit)
 with pytest.raises((ValueError,AssertionError)):verify_activation_evidence(paths,JOB,base)
