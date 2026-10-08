"""Own P job completion follow-up: verifies actual saves, then submits one pure load.
No training edits, no GPU imports, no waiver of speed/learning acceptance.
"""
import fcntl,hashlib,json,math,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
SOURCE='author-native-checkpoint-fixed-five-p'
RELOAD='author-native-checkpoint-fixed-five-p-reload'
run=ROOT/'runs'/SOURCE
status=ROOT/(SOURCE+'-save-reload-followup.json')

def emit(state,**extra):
 value=dict(state=state,source_run=SOURCE,reload_run=RELOAD,checked_at=time.time(),accepted=False,**extra)
 pending=status.with_suffix('.pending');pending.write_text(json.dumps(value,allow_nan=False));pending.replace(status)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()

def queue():
 return subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i %T %N'],text=True).strip()

def selected_records():
 wanted={'outer','all_rank_ack_gate','generation_start','generation_end','checkpoint_saved_full_digest'}
 result=[]
 for p in (run/'evidence').rglob('*.jsonl'):
  with p.open() as f:
   for line in f:
    x=json.loads(line)
    if x.get('kind') in wanted:result.append(x)
 return result

def finite(v):
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 return math.isfinite(v) if isinstance(v,(int,float)) else True

def verified_saves():
 launch=json.loads((run/'launch.json').read_text());assert int(launch['job'])==3125
 assert launch['node']=='h100-16' and launch['source_commit']=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
 rows=selected_records();outer=sorted([x for x in rows if x['kind']=='outer'],key=lambda x:x['step'])
 assert [x['step'] for x in outer]==[1,2,3,4,5] and all(finite(x['metrics']) for x in outer)
 val=[x for x in rows if x['kind']=='generation_start' and x.get('validation')]
 end=[x for x in rows if x['kind']=='generation_end' and x.get('validation')]
 assert len(val)==len(end)==1 and val[0]['input_rows']==64 and end[0]['time_ns']>val[0]['time_ns']
 ack=[x for x in rows if x['kind']=='all_rank_ack_gate'];assert len(ack)==1 and ack[0]['passed']
 for x in ack[0]['records']:assert not Path('/proc',str(x['pid'])).exists(),'Old workers still live; wait resource release'
 expected={(role,rank) for role in ['main','reranker'] for rank in range(4)}
 saved=[x for x in rows if x['kind']=='checkpoint_saved_full_digest']
 assert len(saved)==8 and {(x['role'],x['rank']) for x in saved}==expected
 for x in saved:
  assert int(x['source_job'])==3125 and x['world_size']==4
  path=Path(x['path']);assert path.parent==run.resolve()/'checkpoints/global_step_5'
  assert path.name==('actor' if x['role']=='main' else 'reranker_actor_rollout')
  integrity=json.loads((path/f"native-integrity-rank-{x['rank']}.json").read_text())
  for k in ['state','files','role','source_manifest_sha256','rank','world_size','path','source_job']:assert integrity[k]==x[k],k
  assert x['state']['master_dtypes']==['torch.float32'] and x['state']['model_master']['tensor_bytes']>0 and x['state']['optimizer']['tensor_bytes']>0
  assert set(x['files'])=={f"{prefix}_world_size_4_rank_{x['rank']}.pt" for prefix in ['model','optim','extra_state']}
  for n,v in x['files'].items():assert (path/n).stat().st_size==v['bytes']>0 and sha(path/n)==v['sha256'],n
  assert x['source_manifest_sha256']==sha(run/'tools/source-manifest.json')
 ready=json.loads((ROOT/'tools/preparation-ready.json').read_text())
 assert ready['tools_sha256']==json.loads((run/'tools/preparation-ready.json').read_text())['tools_sha256']
 for n,h in ready['tools_sha256'].items():assert sha(ROOT/'tools'/n)==h==sha(run/'tools'/n),n
 assert sha(ROOT/'tools/source-manifest.json')==sha(run/'tools/source-manifest.json')
 return dict(source_job=3125,exact_outer_steps=[1,2,3,4,5],validation_questions=64,full_saved_ranks=8,actual_files_sha_verified=True,saved_states=[{k:x[k] for k in ['role','rank','state','files','path','source_job']} for x in saved])

def main():
 # A separate lock prevents duplicate CPU follow-ups; launcher uses existing shared lock.
 followup_lock=(ROOT/(SOURCE+'-save-reload-followup.lock')).open('a')
 fcntl.flock(followup_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 emit('waiting_source_driver_exit')
 while not (run/'driver-exit.json').exists():time.sleep(60)
 driver=json.loads((run/'driver-exit.json').read_text())
 if driver['returncode']!=0:emit('source_failed_no_reload',driver=driver);return
 emit('waiting_source_resource_release')
 while queue():time.sleep(60)
 evidence=verified_saves()
 lock=(ROOT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not queue(),'Allocation changed during verification; no submission'
 claim=ROOT/(RELOAD+'-launch-claim.json')
 if claim.exists():emit('already_claimed_no_resubmit',claim=json.loads(claim.read_text()));return
 assert not (ROOT/'runs'/RELOAD).exists()
 command=['srun','--immediate=60','-p','gpu','-w','h100-16','-N1','--ntasks=1','--gres=gpu:h100:8','--exclusive','--cpus-per-task=128','--mem=768G','--job-name=cosearch-author-reload','env','COSEARCH_NATIVE_RUN_ID='+RELOAD,'COSEARCH_NATIVE_RELOAD_ONLY=1','COSEARCH_NATIVE_GPU_PROBE_ONLY=0','COSEARCH_NATIVE_SOURCE_RUN_ID='+SOURCE,'bash',str(ROOT/'tools/start_native_entry.sh')]
 with claim.open('x') as f:json.dump(dict(command=command,status='claimed_not_accepted',source_verification=evidence),f)
 shell=shlex.join(command)+' > '+shlex.quote(str(ROOT/'logs'/(RELOAD+'.log')))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',RELOAD,shell],check=True)
 emit('pure_reload_submitted',command=command,source_verification=evidence)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  emit('followup_failed_no_retry',error=type(exc).__name__+': '+str(exc));raise
