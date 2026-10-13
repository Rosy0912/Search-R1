"""Preserve immutable native manifest JSON metadata; never read PT/tensor contents."""
import argparse
import importlib.util
import json
import shlex
from pathlib import Path

PARENT_SOURCE = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
CHILD_SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
RUN_ROOT = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs'
REMOTE = r'''
import ctypes,errno,fcntl,hashlib,json,os,re,time
from pathlib import Path

def _native_require(condition, message):
 if not condition:raise ValueError(message)

def _native_sha(raw):return hashlib.sha256(raw).hexdigest()

def _native_read(path):
 _native_require(not path.is_symlink() and path.is_file(),'Manifest is missing/symlink: '+str(path))
 before=path.stat();raw=path.read_bytes();after=path.stat()
 _native_require(len(raw)<=1000000 and before.st_size==len(raw)==after.st_size and before.st_mtime_ns==after.st_mtime_ns,'Manifest not a stable small snapshot: '+str(path))
 return raw,json.loads(raw),after

def _native_atomic_new(path, raw):
 if path.exists():
  _native_require(not path.is_symlink() and path.read_bytes()==raw,'Existing immutable history bytes differ: '+str(path))
  return False
 temporary=path.with_name('.'+path.name+'.pending-'+str(os.getpid())+'-'+str(time.time_ns()))
 try:
  with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
  # Atomic publication without replacing a concurrently-created destination.
  libc=ctypes.CDLL(None,use_errno=True);rename=libc.renameat2
  rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
  result=rename(-100,os.fsencode(temporary),-100,os.fsencode(path),1)
  if result:
   code=ctypes.get_errno()
   if code==errno.EEXIST:
    _native_require(not path.is_symlink() and path.read_bytes()==raw,'Concurrent immutable history differs: '+str(path));return False
   raise OSError(code,os.strerror(code),str(path))
  directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(directory)
  finally:os.close(directory)
  _native_require(path.read_bytes()==raw,'Written history readback differs: '+str(path))
  return True
 finally:temporary.unlink(missing_ok=True)

def _native_capture(path, source, role, history):
 raw,value,stat=_native_read(path);identity=value.get('identity') or {};step=identity.get('completed_step')
 _native_require(value.get('status')=='COMPLETE_NATIVE_ROLLING_STATE_SAVED' and value.get('snapshot_kind')=='rolling_completed_outer','Not complete rolling native snapshot')
 _native_require(type(step) is int and (step==1 if role=='parent_outer1' else step>=2),'Outside parent1/child>=2 scope')
 _native_require(path.parent.name=='global_step_'+str(step),'Manifest directory/step differs')
 run=path.parents[4]
 _native_require(identity.get('code_commit')==source and identity.get('run_id')==run.name and identity.get('job_id')=='not-slurm;systemd:'+source,'Source/run/job identity differs')
 for key in ('config_sha256','data_identity','model_identity'):
  _native_require(re.fullmatch('[0-9a-f]{64}',str(identity.get(key,''))) is not None,'Missing manifest identity '+key)
 accounting=identity.get('guard_continue_accounting') or {};counts=accounting.get('total_optimizer_calls_per_rank')
 _native_require(accounting.get('completed_outer_updates')==step and isinstance(counts,dict) and set(counts)==set(map(str,range(8))),'Missing coherent native rank accounting')
 _native_require(all(type(n) is int and n>=0 for n in counts.values()) and len(set(counts.values()))==1,'Eight-rank recorded native optimizer totals disagree')
 workers=value.get('worker_states') or []
 _native_require(len(workers)==8 and sorted(w.get('identity',{}).get('rank',-1) for w in workers)==list(range(8)),'Native rank coverage is not eight')
 states=[]
 for worker in sorted(workers,key=lambda w:w['identity']['rank']):
  wi=worker['identity'];rank=wi['rank'];calls=worker.get('actual_total_optimizer_calls');adam=worker.get('observed_adam_steps')
  _native_require(wi==dict(identity,rank=rank,world_size=8,cuda_visible_devices=wi.get('cuda_visible_devices')) and isinstance(wi.get('cuda_visible_devices'),str),'Native worker identity differs')
  _native_require(worker.get('full_worker_state_finite') is True and worker.get('scheduler_last_epoch')==step,'Native finite/scheduler evidence differs')
  _native_require(type(calls) is int and calls==counts[str(rank)] and isinstance(adam,list) and adam and all(type(n) is int and n==calls for n in adam),'Native recorded Adam/count evidence differs')
  expected=path.parent/('rank-%02d.pt'%rank);actual=Path(worker.get('path',''))
  _native_require(actual==expected and not actual.is_symlink() and actual.is_file(),'Native rank file path is missing/outside snapshot')
  file_stat=actual.stat()
  _native_require(file_stat.st_size==worker.get('bytes') and file_stat.st_size>0,'Native rank stat bytes differ')
  for key in ('sha256','logical_state_sha256'):
   _native_require(re.fullmatch('[0-9a-f]{64}',str(worker.get(key,''))) is not None,'Native rank missing recorded '+key)
  states.append(dict(rank=rank,full_worker_state_finite=worker['full_worker_state_finite'],actual_total_optimizer_calls=calls,observed_adam_steps=adam,scheduler_last_epoch=worker['scheduler_last_epoch'],file_bytes_at_capture=file_stat.st_size,file_mtime_ns_at_capture=file_stat.st_mtime_ns,recorded_file_sha256=worker['sha256'],recorded_logical_state_sha256=worker['logical_state_sha256']))
 driver=value.get('driver') or {};driver_path=Path(driver.get('path',''))
 _native_require(driver_path==path.parent/'driver.pt' and not driver_path.is_symlink() and driver_path.is_file() and driver.get('next_step')==step+1,'Native driver missing/incoherent')
 for key in ('sha256','logical_state_sha256'):
  _native_require(re.fullmatch('[0-9a-f]{64}',str(driver.get(key,''))) is not None,'Native driver missing '+key)
 driver_stat=driver_path.stat();_native_require(driver_stat.st_size>0,'Empty native driver')
 again,_,_=_native_read(path);_native_require(again==raw,'Source manifest changed while collecting')
 history.mkdir(parents=True,exist_ok=True);locks=history/'.locks';locks.mkdir(exist_ok=True)
 key='source-'+source+'-outer-%06d'%step;archived=history/(key+'.manifest.json');receipt_path=history/(key+'.receipt.json');source_sha=_native_sha(raw)
 with (locks/(key+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if archived.exists():
   _native_require(not archived.is_symlink() and _native_sha(archived.read_bytes())==source_sha,'Existing manifest history SHA differs; no overwrite')
  created=_native_atomic_new(archived,raw)
  if receipt_path.exists():
   _,receipt,_=_native_read(receipt_path)
   _native_require(receipt.get('source_manifest_sha256')==source_sha and receipt.get('source_manifest_bytes')==len(raw) and receipt.get('source_commit')==source and receipt.get('completed_outer')==step and receipt.get('history_manifest_path')==str(archived),'Existing history receipt binding differs; no overwrite')
  else:
   receipt=dict(schema='native_manifest_history_metadata_v1',status='COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED',captured_epoch=time.time(),scope=role,source_commit=source,completed_outer=step,source_manifest_path=str(path),source_manifest_sha256=source_sha,source_manifest_bytes=len(raw),source_manifest_mtime_ns=stat.st_mtime_ns,history_manifest_path=str(archived),history_manifest_sha256=source_sha,workers=states,driver=dict(file_bytes_at_capture=driver_stat.st_size,file_mtime_ns_at_capture=driver_stat.st_mtime_ns,recorded_file_sha256=driver['sha256'],recorded_logical_state_sha256=driver['logical_state_sha256'],next_step=driver['next_step']),tensor_contents_read=False,PT_file_sha256_recomputed=False,PT_presence_is_capture_time_only=True,PT_current_availability_after_capture_is_unknown=True,independent_cold_reload_performed=False,exact_executor_resume_claimed=False,limits='Original complete manifest metadata preserved byte-for-byte; recorded finite/logical hashes are original native writer evidence. No tensor read/hash/reload; old PT may later be pruned.')
   _native_atomic_new(receipt_path,(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
  _native_require(_native_sha(archived.read_bytes())==source_sha,'Archived manifest final SHA differs')
 return dict(scope=role,source_commit=source,completed_outer=step,source_manifest_sha256=source_sha,source_manifest_bytes=len(raw),history_manifest_path=str(archived),receipt_path=str(receipt_path),created=created,full_worker_state_finite_by_rank={str(w['rank']):w['full_worker_state_finite'] for w in states},actual_optimizer_calls_per_rank={str(w['rank']):w['actual_total_optimizer_calls'] for w in states},scheduler_last_epoch_by_rank={str(w['rank']):w['scheduler_last_epoch'] for w in states},source_PT_present_at_capture=True,independent_cold_reload_performed=False)

def _native_collect(config):
 started=time.time();history=Path(config['destination_run'])/'output/native-manifest-history';results=[];skipped=[];errors=[]
 for item in config['sources']:
  rolling=Path(item['run'])/'checkpoints/training_state/rolling-completed'
  for path in sorted(rolling.glob('global_step_*/manifest.json')):
   suffix=path.parent.name.removeprefix('global_step_')
   if not suffix.isdecimal() or (item['scope']=='parent_outer1' and int(suffix)!=1) or (item['scope']=='child_native1_continuation' and int(suffix)<2):continue
   try:
    _,value,_=_native_read(path)
    if value.get('status')!='COMPLETE_NATIVE_ROLLING_STATE_SAVED':
     skipped.append(dict(path=str(path),reason='incomplete snapshot status'));continue
    results.append(_native_capture(path,item['source_commit'],item['scope'],history))
   except FileNotFoundError:skipped.append(dict(path=str(path),reason='source pruned or saving; not archived'))
   except Exception as exc:errors.append(dict(path=str(path),audit_error=repr(exc)))
 indexed={item['source_commit']:[] for item in config['sources']}
 for path in sorted(history.glob('source-*-outer-*.receipt.json')):
  try:
   _,receipt,_=_native_read(path);source=receipt.get('source_commit');step=receipt.get('completed_outer')
   if source in indexed and type(step) is int and (step==1 if source==config['sources'][0]['source_commit'] else step>=2):indexed[source].append(step)
  except Exception as exc:errors.append(dict(path=str(path),audit_error=repr(exc)))
 return dict(status='NATIVE_MANIFEST_HISTORY_COLLECTED' if not errors else 'NATIVE_MANIFEST_HISTORY_AUDIT_ERROR',observed_start_epoch=started,observed_end_epoch=time.time(),history_directory=str(history),records_checked_this_call=results,indexed_steps_by_source={source:sorted(set(steps)) for source,steps in indexed.items()},skipped=skipped,audit_error=errors or None,audit_errors=errors,tensor_contents_read=False,training_signals_sent=False,frozen_source_modified=False,independent_cold_reload_performed=False,coverage_scope='Receipt metadata index; this call validates currently retained complete native manifests, not independently reloads historical PT.')

try:
 native_manifest_history_receipt=_native_collect(__NATIVE_CONFIG__)
except Exception as _native_error:
 native_manifest_history_receipt=dict(status='NATIVE_MANIFEST_HISTORY_AUDIT_ERROR',audit_error=repr(_native_error),audit_errors=[repr(_native_error)],tensor_contents_read=False,training_signals_sent=False,frozen_source_modified=False)
'''


def remote_code(run_directory):
    expected = RUN_ROOT + '/tree-nested-branch-fresh200-not-slurm;systemd:' + CHILD_SOURCE
    if run_directory != expected:
        raise ValueError('Collector is bound to the authorized ba9 native1 continuation run')
    config = dict(destination_run=run_directory, sources=[
        dict(source_commit=PARENT_SOURCE, scope='parent_outer1', run=RUN_ROOT + '/tree-nested-branch-fresh200-not-slurm;systemd:' + PARENT_SOURCE),
        dict(source_commit=CHILD_SOURCE, scope='child_native1_continuation', run=run_directory),
    ])
    return REMOTE.replace('__NATIVE_CONFIG__', repr(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--run-directory', default=RUN_ROOT + '/tree-nested-branch-fresh200-not-slurm;systemd:' + CHILD_SOURCE)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('native_history_ssh18', Path(__file__).with_name('ssh_retained18.py'))
    route = importlib.util.module_from_spec(spec); spec.loader.exec_module(route)
    code = remote_code(args.run_directory) + '\nprint(json.dumps(native_manifest_history_receipt))\n'
    response = route.run('python3 -c ' + shlex.quote(code), target='h100-10', timeout=60)
    if response.returncode:
        result = dict(status='NATIVE_MANIFEST_HISTORY_AUDIT_ERROR', audit_error=response.stderr[-2000:], training_signals_sent=False)
    else:
        result = json.loads(response.stdout)
    encoded = json.dumps(result, indent=2) + '\n'
    if len(encoded.encode()) > 100000:
        raise RuntimeError('Collector scalar summary exceeds 100KB')
    args.output.write_text(encoded)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
