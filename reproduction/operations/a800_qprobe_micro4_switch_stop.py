"""Stop only the superseded A800 run after the reviewed micro4 run is prepared.

The default action sends at most one PID-fd SIGTERM to the pinned old pipeline.
--audit-only invokes the separately published C4 read-only cleanup verifier.
Neither action launches, retries, restores, changes runtime code, or manually
removes any model, checkpoint, teacher cache, log, or other project's process.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path('/Users/shaoshitong/Documents/Search-G1')
OPS=Path(__file__).resolve().parents[2]
OLD_RUNTIME=ROOT/'qprobe-a800-flex-placement'
NEW_RUNTIME=ROOT/'qprobe-a800-micro4'
OLD_HANDLE=ROOT/'preparation/qprobe-a800-flex-deployment.json'
AUTHORITY=ROOT/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json'
HUMAN=ROOT/'preparation/qprobe-teacher-warmup-a800-human-intent.json'
OLD_SOURCE='d85979d8d2af24e6468ca7565078acd8bc1ae1f7'
OLD_RUN='00028e46b70b40a2ba6980b1c988f92c'
OLD_EXECUTION='8b53bd8975264854aa520682b0512e78'
OLD_BASE='/data/shaoshitong/Search-G1-a800-fresh200-b9860d6d2a5e43a59d0b25f99fa13bce'
OLD_PID,OLD_TICKS=3063879,605227026
LIBRARY=ROOT/'actor-probes-ten'
LIBRARY_SOURCE='c4d156de124de7a94652d75858bcbabc8c56083c'
REMOTE_LIBRARY='/data/shaoshitong/Search-G1-a800-fresh200-6047f7f9674b49fdad718a256ab3db95/repo'
STOP_INTENT='preparation/human-micro4-switch-stop-intent.json'
STOP_SIGNAL='logs/human-micro4-switch-stop-signal.json'
BIRTH_KEYS=('pid','start_ticks','sid','uid')

SOURCE_CHECK=r'''
def sealed(path,limit=16*1024**2):
 path=Path(path);assert path==path.resolve() and not path.is_symlink()
 fd=os.open(path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
 try:
  before=os.fstat(fd);assert stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size<=limit
  with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(before.st_size+1)
  after=os.fstat(fd)
  assert len(raw)==before.st_size and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert path.stat().st_ino==before.st_ino
 finally:os.close(fd)
 return raw
def document(path,limit=16*1024**2):return json.loads(sealed(path,limit))
def verify_source(repo,commit,mapping,count=None,allow_hydra=False):
 repo=Path(repo);assert repo==repo.resolve() and repo.is_dir() and repo.stat().st_uid==os.getuid()
 assert re.fullmatch('[0-9a-f]{40}',commit)
 def git(*args,**kw):return subprocess.check_output(['git','-C',str(repo),*args],timeout=30,**kw)
 assert git('rev-parse','HEAD').decode().strip()==commit
 assert not git('status','--porcelain').strip()
 def source(relative):
  p=Path(relative)
  assert not p.is_absolute() and '..' not in p.parts and not any(c in relative for c in '\r\n\0')
  return (p.suffix in ('.py','.sh','.yaml','.yml','.json') or relative in ('verl/version/version','verl/single_controller/version/version')) and not {'.git','__pycache__','evidence'}.intersection(p.parts)
 names=[x.decode() for x in git('ls-tree','-rz','--name-only',commit).split(b'\0') if x]
 names=sorted(x for x in names if source(x));assert names==sorted(mapping)
 if count is not None:assert len(names)==count
 data=git('cat-file','--batch',input=''.join(commit+':'+x+'\n' for x in names).encode());offset=0
 for name in names:
  end=data.index(b'\n',offset);header=data[offset:end].split();assert len(header)==3 and header[1]==b'blob'
  size=int(header[2]);start=end+1;raw=data[start:start+size]
  assert len(raw)==size and data[start+size:start+size+1]==b'\n'
  assert hashlib.sha256(raw).hexdigest()==mapping[name]==hashlib.sha256(sealed(repo/name)).hexdigest()
  offset=start+size+1
 assert offset==len(data)
 actual=[]
 for path in repo.rglob('*'):
  if not path.is_file():continue
  relative=str(path.relative_to(repo))
  if not source(relative):continue
  if relative not in mapping:
   assert allow_hydra and re.fullmatch(r'outputs/[0-9]{4}-[0-9]{2}-[0-9]{2}/[0-9]{2}-[0-9]{2}-[0-9]{2}/\.hydra/(config|hydra|overrides)\.yaml',relative)
   sealed(path);continue
  actual.append(relative)
 assert sorted(actual)==names
 return mapping
'''
exec(SOURCE_CHECK)

REMOTE_COMMON=r'''
import hashlib,json,os,re,socket,stat,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
p=json.load(sys.stdin);intent=p['old_intent'];base=Path(intent['base']);repo=base/'repo'
'''+SOURCE_CHECK+r'''
assert sys.executable=='/data/search-g1/env/bin/python'
assert str(base)==p['old_base'] and base==base.resolve() and base.stat().st_uid==os.getuid()
assert intent['source_commit']==p['old_source'] and intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800'
assert document(base/'preparation/qprobe-intent.json')==intent
run=document(base/'logs/run-manifest.json');launch=document(base/'logs/deployment-launch-receipt.json')
claim=document(base/'logs/pipeline-launch-claim.json');allocation=document(base/'logs/native-allocation-admission.json')
assert run['run_id']==launch['run_id']==claim['run_id']==allocation['run_id']==p['old_run']
assert run['execution_id']==launch['execution_id']==claim['execution_id']==allocation['execution_id']==p['old_execution']
assert run['source_commit']==launch['source_commit']==p['old_source'] and run['source_sha256']==intent['source_sha256']
assert run['execution_backend']=='ssh_local' and run['slurm_job_id'] is None and run['execution_host']==socket.gethostname()
verify_source(repo,p['old_source'],run['source_sha256'],470,allow_hydra=True)
training=run['training_physical_indices'];auxiliary=run['auxiliary_physical_indices'];assigned=training+auxiliary
assert run['training_world_size']==4 and len(training)==len(set(training))==4 and all(type(i) is int for i in assigned)
assert auxiliary==run['retriever_physical_indices']==[5,6] and run['probe_physical_index']==6 and len(set(assigned))==6
physical={int(row[0]):row[1].strip() for row in run['physical_gpu_inventory']}
assert len(physical)==len(run['physical_gpu_inventory']) and len(set(physical.values()))==len(physical)
assert set(assigned)<=set(physical) and run['excluded_physical_indices']==sorted(set(physical)-set(assigned))
assert run['assigned_gpu_uuids']=={str(i):physical[i] for i in assigned} and run['training_gpu_uuids']==[physical[i] for i in training]
assert run['gpu_placement_bound'] is True and run['gpu_placement_binding_receipt']=='logs/local-gpu-placement-binding.json'
binding_path=base/run['gpu_placement_binding_receipt'];binding=document(binding_path)
assert hashlib.sha256(sealed(binding_path)).hexdigest()==run['gpu_placement_binding_sha256']
for key in ('run_id','execution_id','source_commit','training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','training_gpu_uuids','assigned_gpu_uuids','physical_gpu_inventory','gpu_placement_bound_unix'):assert binding[key]==run[key]
for key in ('training_physical_indices','auxiliary_physical_indices','excluded_physical_indices','assigned_gpu_uuids','gpu_placement_binding_sha256'):
 assert launch[key]==claim[key]==allocation[key]==run[key]
assert allocation['accepted'] is True and allocation['actual_allocated_indices']==assigned and allocation['actual_cuda_runtime_uuids']==run['training_gpu_uuids']
keys=('pid','start_ticks','sid','uid');producer=run['pipeline_identity']
assert producer==claim['pipeline_identity']==allocation['pipeline_identity']
assert tuple(producer[k] for k in keys)==(p['old_pid'],p['old_ticks'],p['old_pid'],os.getuid())
assert tuple(launch[k] for k in ('pid','start_ticks','sid'))==(p['old_pid'],p['old_ticks'],p['old_pid'])
ledger=document(base/'logs/process-ledger.json',256*1024**2);assert ledger['run_id']==run['run_id'] and ledger['jobs']
jobs={}
for name,job in ledger['jobs'].items():
 assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,64}',name)
 leader=job['leader'];records=[leader]+list(job['ledger'].values())
 assert leader['pid']==leader['sid'] and leader['uid']==os.getuid()
 assert all(type(row[k]) is int for row in records for k in keys)
 assert all(row['pid']>0 and row['start_ticks']>0 and (row['sid'],row['uid'])==(leader['sid'],leader['uid']) for row in records)
 births=sorted({tuple(row[k] for k in keys) for row in records})
 jobs[name]=dict(leader={k:leader[k] for k in keys},births=[dict(zip(keys,row)) for row in births])
inventory=dict(identity={k:run[k] for k in ('run_id','execution_id','source_commit','execution_backend','training_world_size','profile','slurm_job_id')},jobs=jobs)
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
'''

REMOTE_STOP=REMOTE_COMMON+r'''
import signal
# Refuse unsupported PID-fd APIs before creating the remote intent or signalling.
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
new=p['new_intent'];newbase=Path(new['base']);assert newbase!=base and newbase==newbase.resolve() and newbase.stat().st_uid==os.getuid()
assert document(newbase/'preparation/qprobe-intent.json')==new
newrun=document(newbase/'logs/run-manifest.json');prepared=document(newbase/'logs/qprobe-prepared.json');bound=document(newbase/'logs/qprobe-bound.json')
assert newrun['source_commit']==new['source_commit'] and newrun['source_sha256']==new['source_sha256']
assert (newrun['run_id'],newrun['execution_id'])==(p['new_run'],p['new_execution']) and newrun['run_id']!=run['run_id'] and newrun['execution_id']!=run['execution_id']
assert prepared['source_commit']==bound['source_commit']==new['source_commit'] and prepared['run_id']==bound['run_id']==newrun['run_id']
assert prepared['execution_id']==newrun['execution_id'] and prepared['stage']=='CPU_PREPARED_NO_GPU_OR_PIPELINE_LAUNCH' and bound['stage']=='QPROBE_BOUND_PRELAUNCH'
assert prepared['trainer_started'] is False and bound['trainer_started'] is False
assert newrun['execution_backend']=='ssh_local' and newrun['slurm_job_id'] is None and newrun['execution_host']==socket.gethostname()
assert newrun['training_world_size']==4 and newrun['completed_updates']==0 and newrun['target_updates']==200 and newrun['actor_micro_batch_per_rank']==4
budget=newrun['budget'];assert budget['world_size']==4 and budget['per_rank_actor_micro_batch']==4 and budget['ppo_micro_batch_size']==16 and budget['ppo_mini_batch_size']==128
assert (budget['ppo_mini_batch_size']//newrun['training_world_size'])//newrun['actor_micro_batch_per_rank']==8
assert budget['per_rank_log_prob_micro_batch']==8 and budget['lr_warmup_steps']==57
assert newrun['initial_probe_version'] is None and newrun.get('full_state_restore') is None and newrun['gpu_placement_bound'] is False
assert newrun['initial_model']==newrun['reference_model']==run['initial_model']==run['reference_model']
assert newrun['initial_model_sha256']==run['initial_model_sha256'] and newrun['input_sha256']==run['input_sha256']
q=newrun['question_probe'];assert (q['candidate_questions'],q['selected_questions'],q['responses_per_question'],q['exploration_questions'],q['bootstrap_max_groups_per_outer'])==(5120,512,5,64,5120)
for name in ('pipeline-launch-claim.json','training-launch-claim.json','deployment-launch-receipt.json','native-allocation-admission.json','process-ledger.json','actual-updates.jsonl'):
 assert not (newbase/'logs'/name).exists() and not (newbase/'logs'/name).is_symlink()
verify_source(newbase/'repo',new['source_commit'],new['source_sha256'])
def counters(relative,rank=None):
 path=base/relative
 if not path.exists():return dict(exists=False,complete_rows=0,last_complete_outer=None)
 raw=sealed(path,32*1024**2);end=raw.rfind(b'\n')+1;rows=[];processed=written=0
 for line in raw[:end].splitlines():
  row=json.loads(line);assert row['run_id']==run['run_id'] and row.get('execution_id',run['execution_id'])==run['execution_id']
  if rank is None:
   assert type(row['step']) is int and 1<=row['step']<=200
   rows.append({k:row[k] for k in ('step','opportunities_processed','actual_optimizer_writes')})
  else:
   assert row['rank']==rank
   if row.get('kind')=='OPPORTUNITY' and row.get('processed') is True:processed+=1;written+=int(row.get('optimizer_write_succeeded') is True)
 result=dict(exists=True,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),partial_tail_bytes=len(raw)-end)
 if rank is None:result.update(complete_rows=len(rows),last_complete_outer=rows[-1]['step'] if rows else None,completed_update_rows=rows)
 else:result.update(rank=rank,processed=processed,returned_adam_writes=written,skips=processed-written)
 return result
updates=counters('logs/actual-updates.jsonl');rank_counts=[counters('logs/optimizer-rank-%d.jsonl'%rank,rank) for rank in range(4)]
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import identity,same,signal_owned
live=same(producer);state=None
if live:
 actual=identity(producer['pid']);state=actual['state']
 assert 'reproduction.a800_common.run_pipeline' in actual['command'] and str(base) in actual['command']
def once(path,value):
 raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
assert not (base/p['stop_intent']).exists() and not (base/p['stop_signal']).exists()
snapshot=dict(role='human_micro4_corrected_prepared_switch_stop_intent',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 pipeline_identity={k:producer[k] for k in keys},preserved_births=inventory,old_counts=updates,old_rank_counts=rank_counts,
 new_run_id=p['new_run'],new_execution_id=p['new_execution'],new_source_commit=new['source_commit'],new_base=str(newbase),
 human_intent_sha256=p['human_intent_sha256'],correction_recorded_unix=p['correction_recorded_unix'],authority_sha256=p['authority_sha256'],
 reviewed_prepared_evidence_sha256=p['evidence_sha256'],actor_local_micro=4,actor_global_micro=16,gradient_accumulation=8,
 old_source_log_checkpoint_teacher_cache_preserved=True,no_manual_deletion=True,private_corpus_cleanup_delegated_to_pipeline=True,
 scope='one verified old pipeline only; its native cleanup handles its owned components; no counters inherited by the fresh micro4 run',checked_unix=time.time())
once(base/p['stop_intent'],snapshot)
sent=False
if live and state not in ('Z','X'):signal_owned(producer,signal.SIGTERM);sent=True
receipt=dict(role='human_micro4_switch_pipeline_signal_result',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 pipeline_identity={k:producer[k] for k in keys},signal_sent=sent,signal='SIGTERM' if sent else None,state_before=state,
 stop_intent_sha256=hashlib.sha256(sealed(base/p['stop_intent'])).hexdigest(),old_counts=updates,old_rank_counts=rank_counts,
 cleanup_accepted=False,cleanup_requires_actual_read_only_audit=True,no_automatic_retry=True,checked_unix=time.time())
once(base/p['stop_signal'],receipt)
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
print('MICRO4_SWITCH_JSON:'+json.dumps(receipt,sort_keys=True,allow_nan=False),flush=True)
'''

REMOTE_AUDIT=REMOTE_COMMON+r'''
snapshot=document(base/p['stop_intent']);signal_receipt=document(base/p['stop_signal'])
for row in (snapshot,signal_receipt):
 assert (row['run_id'],row['execution_id'],row['source_commit'])==(run['run_id'],run['execution_id'],run['source_commit'])
 assert row['pipeline_identity']=={k:producer[k] for k in keys}
assert hashlib.sha256(sealed(base/p['stop_intent'])).hexdigest()==signal_receipt['stop_intent_sha256']
library=Path(p['library_repo']);verify_source(library,p['library_source'],p['library_source_sha256'],455,allow_hydra=True)
sys.path.insert(0,str(library))
from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup,birth_inventory
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
current=birth_inventory(base,run)
actual=verify_cleanup(base,run,preserved_births=snapshot['preserved_births'])
assert actual['accepted'] is True and actual['actual_pipeline_and_owned_sessions_exited'] is True and actual['owned_cuda_contexts_remaining']==0
assert actual['private_cache_removed'] is True and actual['original_assets_present'] is True
imported={}
for name,module in list(sys.modules.items()):
 path=getattr(module,'__file__',None)
 if not path:continue
 path=Path(path).resolve()
 if library not in path.parents:continue
 relative=str(path.relative_to(library));assert relative in p['library_source_sha256']
 digest=hashlib.sha256(sealed(path)).hexdigest();assert digest==p['library_source_sha256'][relative];imported[relative]=digest
assert 'reproduction/common8/actor_probe_owned_cleanup.py' in imported
assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
receipt=dict(role='human_micro4_switch_actual_owned_cleanup_audit',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],
 accepted=True,no_stop_or_signal=True,no_remote_mutation=True,old_counts=snapshot['old_counts'],old_rank_counts=snapshot['old_rank_counts'],
 stop_intent_sha256=signal_receipt['stop_intent_sha256'],auditor_source_commit=p['library_source'],imported_module_sha256=imported,
 verification={k:actual[k] for k in ('actual_pipeline_and_owned_sessions_exited','owned_cuda_contexts_remaining','private_cache_removed','original_assets_present','reused_pids_belonging_to_other_births','reused_session_ids','recorded_births')},
 files=actual['files'],checked_unix=actual['checked_unix'])
print('MICRO4_SWITCH_JSON:'+json.dumps(receipt,sort_keys=True,allow_nan=False),flush=True)
'''


def sha(path):return hashlib.sha256(sealed(path)).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def once(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def evidence(path):
    path=Path(path);path.relative_to(ROOT/'preparation')
    return document(path),sha(path)


def authority_and_human():
    handoff=document(AUTHORITY);current=handoff['current_authoritative'];instruction=document(HUMAN)
    assert handoff['goal_active'] is True and current['pending_human_micro_batch_correction'] is True
    assert tuple(current[k] for k in ('run_id','execution_id','source_commit','base','handle_path'))==(OLD_RUN,OLD_EXECUTION,OLD_SOURCE,OLD_BASE,str(OLD_HANDLE))
    assert (current['pipeline_pid'],current['pipeline_start_ticks'],current['pipeline_sid'])==(OLD_PID,OLD_TICKS,OLD_PID)
    correction=instruction['latest_micro_batch_correction']
    assert correction['raw']=='你跑的有问题，应该也是4，全局16这样子'
    assert tuple(correction[k] for k in ('actor_micro_batch_per_rank','actor_micro_batch_global','gradient_accumulation','ppo_mini_batch','world_size'))==(4,16,8,128,4)
    assert type(correction['received_recorded_unix']) in (int,float) and 0<correction['received_recorded_unix']<=time.time()
    return sha(AUTHORITY),sha(HUMAN),correction['received_recorded_unix']


def approved_review(review,intent):
    assert review['role']=='a800_qprobe_independent_code_review' and review['approved'] is True and review['independent'] is True
    assert review['reviewer'] and review['implementation_actor'] and review['reviewer']!=review['implementation_actor']
    assert review['source_commit']==intent['source_commit'] and review['source_sha256']==intent['source_sha256']
    assert review.get('blocking_findings',[])==[]
    checks=review['checks'];assert checks and all(value is True for value in checks.values())
    assert checks['four_rank_bf16_local4_global16_accum8_ref8'] is True
    assert type(review['reviewed_unix']) in (int,float) and 0<review['reviewed_unix']<=time.time()


def prepared_micro4(args):
    intent,handle_sha=evidence(args.new_handle)
    assert args.new_handle==ROOT/'preparation/qprobe-a800-micro4-deployment.json'
    assert intent['role']=='a800_qprobe_prospective_intent' and intent['host']=='baai-a800' and intent['source_commit']!=OLD_SOURCE
    assert re.fullmatch('[0-9a-f]{32}',intent['deployment_id']) and intent['base']=='/data/shaoshitong/Search-G1-a800-fresh200-'+intent['deployment_id'] and intent['base']!=OLD_BASE
    verify_source(NEW_RUNTIME,intent['source_commit'],intent['source_sha256'])
    publication,pub_sha=evidence(args.source_publication);review,review_sha=evidence(args.code_review)
    assert publication['source_commit']==publication['actual_remote_ref_sha']==intent['source_commit'] and publication['GitHub_actual_bytes_verified'] is True
    assert publication['source_map_sha256']==canonical_sha(intent['source_sha256'])
    assert publication['independent_review_path']==str(args.code_review) and publication['independent_review_sha256']==review_sha
    approved_review(review,intent)
    prepared,prepare_sha=evidence(args.prepare_receipt);bound,bind_sha=evidence(args.bind_receipt);deployment,deployment_sha=evidence(args.deployment_review)
    assert prepared['action']=='prepare' and bound['action']=='bind' and prepared['exit_code']==bound['exit_code']==0
    assert prepared['deployment_id']==bound['deployment_id']==intent['deployment_id']
    assert len(prepared['observations'])==len(bound['observations'])==1
    first,last=prepared['observations'][0],bound['observations'][0]
    assert first['stage']=='CPU_PREPARED_NO_GPU_OR_PIPELINE_LAUNCH' and last['stage']=='QPROBE_BOUND_PRELAUNCH'
    assert first['source_commit']==last['source_commit']==intent['source_commit'] and first['run_id']==last['run_id'] and first['run_id']!=OLD_RUN
    assert first['base']==intent['base'] and first['trainer_started'] is False and last['trainer_started'] is False
    assert last['target_updates']==200 and last['completed_training_outers']==0 and last['extra_frozen_teacher_warmup_outers']==1
    q=last['question_probe'];assert (q['candidate_questions'],q['selected_questions'],q['responses_per_question'],q['exploration_questions'],q['bootstrap_max_groups_per_outer'])==(5120,512,5,64,5120)
    approved_review(deployment,intent)
    assert (deployment['run_id'],deployment['execution_id'],deployment['base'],deployment['deployment_id'])==(first['run_id'],first['execution_id'],intent['base'],intent['deployment_id'])
    assert deployment['actual_checks'] and all(value is True for value in deployment['actual_checks'].values())
    actual=deployment['direct_actual_prelaunch_observation']
    assert actual['pipeline_claimed'] is False and actual['trainer_claimed'] is False and actual['actual_updates_ledger']['exists'] is False
    assert actual['run_manifest']['run_id']==first['run_id'] and actual['run_manifest']['execution_id']==first['execution_id']
    for path,digest in ((args.new_handle,handle_sha),(args.prepare_receipt,prepare_sha),(args.bind_receipt,bind_sha)):
        records=[row for row in deployment['evidence_files'].values() if row['path']==str(path)]
        assert len(records)==1 and records[0]['sha256']==digest
    hashes=dict(new_handle=handle_sha,source_publication=pub_sha,code_review=review_sha,deployment_review=deployment_sha,prepare_receipt=prepare_sha,bind_receipt=bind_sha)
    return intent,first['run_id'],first['execution_id'],hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('new-handle','code-review','source-publication','deployment-review','prepare-receipt','bind-receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    operator_commit=subprocess.check_output(['git','-C',str(OPS),'rev-parse','HEAD'],text=True,timeout=10).strip()
    relative=str(Path(__file__).resolve().relative_to(OPS))
    assert subprocess.check_output(['git','-C',str(OPS),'show',operator_commit+':'+relative],timeout=10)==sealed(Path(__file__)), 'Publish this exact switch operator first'
    authority_sha,human_sha,recorded=authority_and_human()
    new,new_run,new_execution,hashes=prepared_micro4(args)
    old=document(OLD_HANDLE);assert old['source_commit']==OLD_SOURCE and old['base']==OLD_BASE
    verify_source(OLD_RUNTIME,OLD_SOURCE,old['source_sha256'],470)
    old_pub=document(ROOT/'preparation/qprobe-flex-source-publication.json')
    assert old_pub['source_commit']==old_pub['actual_remote_ref_sha']==OLD_SOURCE and old_pub['GitHub_actual_bytes_verified'] is True and old_pub['source_map_sha256']==canonical_sha(old['source_sha256'])
    payload=dict(old_intent=old,old_source=OLD_SOURCE,old_run=OLD_RUN,old_execution=OLD_EXECUTION,old_base=OLD_BASE,old_pid=OLD_PID,old_ticks=OLD_TICKS,
        new_intent=new,new_run=new_run,new_execution=new_execution,authority_sha256=authority_sha,human_intent_sha256=human_sha,correction_recorded_unix=recorded,
        evidence_sha256=hashes,stop_intent=STOP_INTENT,stop_signal=STOP_SIGNAL)
    remote=REMOTE_AUDIT if args.audit_only else REMOTE_STOP
    if args.audit_only:
        mapping=verify_source(LIBRARY,LIBRARY_SOURCE,document(ROOT/'preparation/actor-probes-ten-production-after-3ecc-preclaim-exit-deployment.json')['source_sha256'],455)
        payload.update(library_repo=REMOTE_LIBRARY,library_source=LIBRARY_SOURCE,library_source_sha256=mapping)
    ast.parse(remote)
    spec=importlib.util.spec_from_file_location('pinned_micro4_stop_existing_route',OLD_RUNTIME/'reproduction/a800_common/deploy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert sha(Path(module.__file__))==old['source_sha256']['reproduction/a800_common/deploy.py']
    route=module.ssh_command(remote);assert route[-1].startswith('python3 -c ')
    route[-1]=shlex.quote('/data/search-g1/env/bin/python')+' -c '+shlex.quote(remote)
    assert authority_and_human()==(authority_sha,human_sha,recorded), 'Human authority changed before dispatch'
    stem='qprobe-micro4-old00028-cleanup-audit-'+str(time.time_ns()) if args.audit_only else 'qprobe-micro4-old00028-stop-once'
    claim=ROOT/'preparation'/(stem+'.json');output=claim.with_name(claim.stem+'.result.json')
    request=dict(role='micro4_switch_readonly_audit_request' if args.audit_only else 'micro4_switch_one_pipeline_stop_claim',run_id=OLD_RUN,execution_id=OLD_EXECUTION,source_commit=OLD_SOURCE,
        expected_pipeline_birth=dict(pid=OLD_PID,start_ticks=OLD_TICKS,sid=OLD_PID),new_run_id=new_run,new_execution_id=new_execution,new_source_commit=new['source_commit'],
        operator_commit=operator_commit,operator_sha256=sha(Path(__file__)),remote_code_sha256=hashlib.sha256(remote.encode()).hexdigest(),authority_sha256=authority_sha,human_intent_sha256=human_sha,
        reviewed_prepared_evidence_sha256=hashes,read_only=args.audit_only,no_automatic_retry=True,claimed_unix=time.time())
    once(claim,request)
    with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
        try:
            result=subprocess.run(route,input=json.dumps(payload,sort_keys=True,allow_nan=False).encode(),stdout=stdout,stderr=stderr,timeout=60)
            status=dict(exit_code=result.returncode,status='READ_RETURNED' if args.audit_only else 'STOP_REQUEST_RETURNED')
        except (OSError,subprocess.TimeoutExpired) as failure:
            status=dict(exit_code=None,status='TRANSPORT_UNKNOWN_NO_RETRY',failure_type=type(failure).__name__)
        status.update(stdout_bytes=stdout.tell(),stderr_bytes=stderr.tell());stdout.seek(0);raw=stdout.read(256*1024+1)
        stderr.seek(0);error=stderr.read(64*1024)
        status['bounded_stderr_sha256']=hashlib.sha256(error).hexdigest()
        diagnostic=error.decode(errors='replace')
        status['remote_exception_frames']=[dict(file=Path(match.group(1)).name[:160],line=int(match.group(2)),function=match.group(3)[:96])
            for match in re.finditer(r'^[ \t]*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>]+)[ \t]*$',diagnostic,re.MULTILINE)][-8:]
        types=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))(?::|$)',diagnostic,re.MULTILINE)
        status['remote_exception_type']=types[-1][:96] if types else None
    rows=[]
    if status['exit_code']==0 and len(raw)<=256*1024:
        try:
            rows=[json.loads(line.split(':',1)[1]) for line in raw.decode().splitlines() if line.startswith('MICRO4_SWITCH_JSON:')]
            assert len(rows)==1 and (rows[0]['run_id'],rows[0]['execution_id'],rows[0]['source_commit'])==(OLD_RUN,OLD_EXECUTION,OLD_SOURCE)
            if args.audit_only:assert rows[0]['accepted'] is True and rows[0]['no_stop_or_signal'] is True
        except (AssertionError,ValueError,UnicodeDecodeError,KeyError,TypeError) as failure:
            rows=[];status.update(status='OUTPUT_REJECTED_NO_RETRY',failure_type=type(failure).__name__)
    elif len(raw)>256*1024:status['status']='OUTPUT_LIMIT_EXCEEDED_NO_RETRY'
    receipt=dict(request,**status,results=rows,observed_unix=time.time());once(output,receipt)
    print(json.dumps(dict(receipt=str(output),**status,results=rows,no_automatic_retry=True),allow_nan=False))
    return 0 if status['exit_code']==0 and len(rows)==1 else 1


if __name__=='__main__':raise SystemExit(main())
