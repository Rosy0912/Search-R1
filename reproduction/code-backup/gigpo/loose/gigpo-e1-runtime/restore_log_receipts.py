"""Restore only a log transport view after the real policy driver exits.
No trainer, model, optimizer, scheduler, RNG or original source-log mutation.
"""
import os,json,hashlib,time,select,sys
from pathlib import Path
run=Path(sys.argv[1]).resolve();job=sys.argv[2];training_pid=int(sys.argv[3]);supervisor_pid=int(sys.argv[4]);worker_pid=int(sys.argv[5]);source=Path(sys.argv[6]).resolve()
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
assert run.parent==root/'runs' and f'job{job}-' in run.name
assert source.is_relative_to(run/'ray') and source.name.endswith(f'-{worker_pid}.out')
assert os.environ.get('SLURM_JOB_ID')==job and not os.environ.get('CUDA_VISIBLE_DEVICES')
scope=f'/job_{job}/step_0'
def identity(pid):
 p=Path('/proc')/str(pid);cg=(p/'cgroup').read_text();assert any(line.split(':',2)[-1].endswith(scope) for line in cg.splitlines()),cg
 cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 stat=(p/'stat').read_text();return dict(pid=pid,start_ticks=stat.rsplit(')',1)[1].split()[19],command=cmd,cgroup=cg)
ids=[identity(p) for p in (training_pid,supervisor_pid,worker_pid)]
assert 'search_r1.baselines.main' in ids[0]['command'] and str(run) in ids[0]['command']
assert 'formal_run_v3.py' in ids[1]['command']
assert 'main_task' in ids[2]['command']
fds=[os.pidfd_open(p,0) for p in (training_pid,supervisor_pid,worker_pid)]
assert [identity(p) for p in (training_pid,supervisor_pid,worker_pid)]==ids,'PID identity changed while acquiring pidfds'
def dead(fd):
 poll=select.poll();poll.register(fd,select.POLLIN);return bool(poll.poll(0))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def exclusive(path,value):
 with path.open('x') as f:f.write(json.dumps(value,indent=2));f.flush();os.fsync(f.fileno())
intent=run/'log-transport-restore-intent.json';exclusive(intent,dict(job=job,identities=ids,source=str(source),helper_sha256=sha(Path(__file__).read_bytes())))
markers=[b'BASELINE_EVALUATION_METRICS_JSON=',b'SEARCH_R1_INSTRUCT300_COMPLETE_JSON=']
deadline=time.monotonic()+47*3600
published=False
try:
 while not dead(fds[0]):
  if dead(fds[1]) or time.monotonic()>deadline:raise RuntimeError('Supervisor ended or metadata deadline exceeded')
  time.sleep(.2)
 assert not dead(fds[1]),'Supervisor already ended; retain true status'
 raw=source.read_bytes()
 snapshot=run/'log-transport-worker-original.out'
 with snapshot.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 events=[];offset=0
 for line in raw.splitlines(keepends=True):
  for marker in markers:
   if marker in line:
    payload=line.split(marker,1)[1].strip();value=json.loads(payload)
    events.append(dict(marker=marker.decode(),offset=offset,bytes=len(line),line_sha256=sha(line),value=value,line=line))
  offset+=len(line)
 evaluations=[x['value'] for x in events if x['marker']==markers[0].decode()]
 completions=[x['value'] for x in events if x['marker']==markers[1].decode()]
 assert len(evaluations)==9 and len(completions)==1 and evaluations[0]['val/qa_count/all']==512 and evaluations[-1]['val/qa_count/all']==51713
 assert sum(x['val/qa_count/all']==512 for x in evaluations)==7 and sum(x['val/qa_count/all']==64 for x in evaluations)==1 and sum(x['val/qa_count/all']==51713 for x in evaluations)==1
 complete=completions[0];assert complete['actual_new_updates']==300 and complete['logical_step']==300
 assert complete['final_metrics']==evaluations[-1] and events[-1]['marker']==markers[1].decode()
 launch=json.loads((run/'launch-manifest.json').read_text());manifest_path=run/'checkpoints/actor/global_step_300/training_state/manifest.json';manifest_raw=manifest_path.read_bytes();manifest=json.loads(manifest_raw)
 assert launch['slurm_job']==job and launch['code_commit']=='bb1adde13c7fd489aef4feb68a36aeb0d29d5da3' and launch['fresh_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
 expected=dict(job_id=job,logical_step=300,actual_new_updates=300,code_commit=launch['code_commit'],config_sha256=launch['config_sha256'],data_identity='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',source=str(root/'assets/qwen2.5-3b-instruct-aa8e725'),recovery_mode='fresh_official_instruct')
 assert all(manifest.get(k)==v for k,v in expected.items()),'Final manifest identity mismatch'
 original=run/'training.log';assert original.is_file() and not original.is_symlink()
 old=original.read_bytes();backup=run/'training-forwarded-original.log';os.link(original,backup)
 # All forwarded copies remain in the preserved inode. The derived view replaces
 # their complete lines with every real raw event once, in byte-offset order.
 retained=b''.join(line for line in old.splitlines(keepends=True) if not any(m in line for m in markers))
 header=b'\nRAW_WORKER_RECEIPT_TRANSPORT_VIEW; original inode preserved as training-forwarded-original.log\n'
 restored=retained+header+b''.join(x['line'] for x in events)
 pending=run/'training-receipt-view.pending';
 with pending.open('xb') as f:f.write(restored);f.flush();os.fsync(f.fileno())
 receipt=dict(status='VERBATIM_RECEIPT_VIEW_READY_NOT_TRAINING_ACCEPTANCE',job=job,identities=ids,source=str(source),source_snapshot_path=str(snapshot),final_manifest_path=str(manifest_path),final_manifest_sha256=sha(manifest_raw),source_snapshot_bytes=len(raw),source_snapshot_sha256=sha(raw),original_backup=str(backup),original_snapshot_sha256=sha(old),derived_sha256=sha(restored),events=[{k:v for k,v in x.items() if k!='line'} for x in events],trainer_model_state_changed=False,exit_zero_not_inferred=True)
 exclusive(run/'log-transport-source-receipt.json',receipt)
 assert not dead(fds[1]),'Supervisor ended before publication; no status repair'
 os.replace(pending,original);published=True
 directory=os.open(run,os.O_DIRECTORY);os.fsync(directory);os.close(directory)
 exclusive(run/'log-transport-restored.json',dict(status='DERIVED_VIEW_PUBLISHED',derived_sha256=sha(restored),source_receipt=str(run/'log-transport-source-receipt.json')))
except BaseException as error:
 exclusive(run/'log-transport-failure.json',dict(status='PUBLISHED_RECORDING_FAILED' if published else 'NOT_REPAIRED',published=published,error=repr(error)))
 raise
finally:
 for fd in fds:os.close(fd)
