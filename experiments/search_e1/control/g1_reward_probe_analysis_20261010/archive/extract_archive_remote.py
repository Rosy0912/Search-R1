"""Read-only HKUST4 original Archive4 scalar extraction; no raw log/model transfer."""
import datetime, hashlib, json, subprocess, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parent
REMOTE = r'''
import datetime,hashlib,json,os,re,socket,subprocess,time
from pathlib import Path
B=Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
RUN='f04fe593830c4831918ede872812c016'
D={'observed_unix':time.time(),'observed_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'hostname':socket.gethostname(),'base':str(B),'run_id':RUN,'read_only':True,'raw_logs_transferred':False,'curve':[],'validation_log_records':[],'sources':[],'original_pids':[],'checkpoint_inventory':[],'terminal_evidence':[]}
p=B/'runs/archive/round-1-grpo/train.log'; raw=p.read_bytes(); text=raw.decode(errors='replace'); lines=text.splitlines(); st=p.stat()
D['log']={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime':st.st_mtime,'line_count':len(lines)}
metric_keys={'audit/strict_em','audit/legacy_em','audit/score_disagreement','critic/score/mean','critic/rewards/mean','critic/advantages/mean','critic/advantages/max','critic/advantages/min','actor/entropy_loss','actor/pg_loss','actor/pg_clipfrac','actor/ppo_kl','actor/kl_loss','actor/grad_norm','actor/lr','env/finish_ratio','env/ratio_of_valid_action','env/number_of_actions/mean','env/number_of_valid_action','env/number_of_valid_search','response_length/mean','state_tokens/coverage','timing_s/step'}
ansi=re.compile(r'\x1b\[[0-9;]*m'); pattern=re.compile(r'([\w/]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
D['recent_fatal_lines']=[]; D['config_evidence']=[]
for n,line in enumerate(lines,1):
 clean=ansi.sub('',line); match=re.search(r'\bstep:(\d+)\s*-',clean)
 if match:
  metrics={k:float(v) for k,v in pattern.findall(clean)}
  if 'critic/rewards/mean' in metrics:
   D['curve'].append({'iteration':int(match[1]),'line':n,'line_sha256':hashlib.sha256(clean.encode()).hexdigest(),'metrics':{k:v for k,v in metrics.items() if k in metric_keys or k.startswith('val/')}})
 if ('val/test_score' in clean and not match) or 'Final validation' in clean:
  D['validation_log_records'].append({'line':n,'line_sha256':hashlib.sha256(clean.encode()).hexdigest(),'text':clean[:2500]})
 if n>len(lines)-2000 and re.search(r'Traceback \(most recent call last\)|\b(?:RuntimeError|ValueError|OutOfMemoryError|AssertionError|FloatingPointError):|CUDA out of memory',clean):
  D['recent_fatal_lines'].append({'line':n,'text':clean[:700]})
 if n<220 and any(s in clean for s in ["'train_batch_size'","'n_agent'","'n_gpus_per_node'","'use_kl_loss'","'lr'","'lr_warmup_steps'","'total_training_steps'","'ppo_mini_batch_size'","'ppo_micro_batch_size'","'entropy_coeff'","'clip_ratio'"]):
  D['config_evidence'].append({'line':n,'text':clean[:1200]})
for rel in ['receipts/archive-round-1-grpo.json','receipts/pipeline-stage.txt','receipts/source-commit.txt','receipts/human-stop-archive-f04fe593-20261009-4921e2a715fe472d8adf0cbe0d955483.json']:
 p=B/rel
 if p.is_file():
  z=p.read_bytes();entry={'path':str(p),'bytes':len(z),'sha256':hashlib.sha256(z).hexdigest(),'mtime':p.stat().st_mtime}
  if len(z)<=80000:
   try:entry['data']=json.loads(z)
   except ValueError:entry['text']=z.decode(errors='replace')
  D['terminal_evidence'].append(entry)
for pid in [1432632,1434651,1437474,1749337]:
 p=Path('/proc')/str(pid); e={'pid':pid,'exists':p.exists()}
 if e['exists']:
  try:
   stat=(p/'stat').read_text(); rest=stat[stat.rfind(')')+2:].split(); status=(p/'status').read_text(); uid=next(z.split()[1] for z in status.splitlines() if z.startswith('Uid:'))
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   e.update(starttick=int(rest[19]),state=rest[0],ppid=int(rest[1]),uid=int(uid),command=cmd[:1600],cwd=str((p/'cwd').resolve()))
  except OSError as ex:e['read_error']=repr(ex)
 D['original_pids'].append(e)
for rel in ['runtime/archive/upstream/Search-R1/verl/trainer/main_ppo.py','runtime/archive/upstream/Search-R1/verl/trainer/ppo/ray_trainer.py','runtime/archive/upstream/Search-R1/verl/utils/reward_score/qa_em.py','runtime/archive/upstream/Search-R1/search_r1/qa_protocol.py','runtime/archive/scripts/grpo.sh','runtime/archive-jit-20261005/upstream/Search-R1/verl/trainer/main_ppo.py','runtime/archive-jit-20261005-env-v2/upstream/Search-R1/verl/trainer/main_ppo.py']:
 p=B/rel
 if not p.is_file():continue
 z=p.read_bytes();ls=z.decode(errors='replace').splitlines(); e={'path':str(p),'bytes':len(z),'sha256':hashlib.sha256(z).hexdigest(),'mtime':p.stat().st_mtime}
 if p.name in ['main_ppo.py','qa_em.py','qa_protocol.py']:
  if p.name=='main_ppo.py':ranges=[(22,32),(65,115)]
  elif p.name=='qa_em.py':ranges=[(88,140)]
  else:ranges=[(1,145)]
  e['relevant_lines']=[{'line':i,'text':ls[i-1]} for a,b in ranges for i in range(a,min(b,len(ls))+1)]
 elif p.name=='ray_trainer.py':
  e['relevant_lines']=[{'line':i,'text':s} for i,s in enumerate(ls,1) if 'use_kl_loss' in s or 'token_level_rewards' in s or 'audit/' in s or 'compute_data_metrics' in s][:35]
 else:e['relevant_lines']=[{'line':i,'text':s} for i,s in enumerate(ls,1) if re.search(r'lr|batch|warmup|entropy|mini|micro|total_training_steps|temperature|n_agent',s)][:60]
 D['sources'].append(e)
for step in [50,100,150,199,200]:
 p=B/f'runs/archive/round-1-grpo/actor/global_step_{step}';e={'iteration':step,'path':str(p),'exists':p.is_dir()}
 if p.is_dir():
  e['files']=[{'name':z.name,'bytes':z.stat().st_size,'mtime':z.stat().st_mtime} for z in p.iterdir() if z.is_file()]
  q=p/'model.safetensors.index.json'
  if q.is_file():
   z=q.read_bytes();idx=json.loads(z);e['index_sha256']=hashlib.sha256(z).hexdigest();shards=set(idx['weight_map'].values());e['all_indexed_shards_nonempty']=all((p/f).is_file() and (p/f).stat().st_size>0 for f in shards)
 D['checkpoint_inventory'].append(e)
try:
 q=subprocess.run(['tmux','list-panes','-a','-F','#{session_name} #{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True,timeout=8)
 D['tmux']={'rc':q.returncode,'owned_matching_rows':[s for s in q.stdout.splitlines() if 'e1-priority-5iter-archive-20261004' in s]}
except Exception as ex:D['tmux']={'error':repr(ex)}
D['curve_summary']={'count':len(D['curve']),'iterations':[r['iteration'] for r in D['curve']],'latest':D['curve'][-1] if D['curve'] else None,'missing_1_to_200':sorted(set(range(1,201))-{r['iteration'] for r in D['curve']})}
print(json.dumps(D,ensure_ascii=False,separators=(',',':')))
'''

def main():
 label=sys.argv[1] if len(sys.argv)>1 else datetime.datetime.now().strftime('%H%M%S')
 argv=['/usr/bin/ssh','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oControlMaster=no','-S','none','-oBatchMode=yes','-oConnectTimeout=10','hkust4','python3 -']
 start=time.time(); q=subprocess.run(argv,input=REMOTE.encode(),capture_output=True,timeout=120);end=time.time()
 p=OUT/('observation-'+label+'.json');e=OUT/('execution-'+label+'.json')
 if q.returncode==0:
  data=json.loads(q.stdout);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 else:
  data={'error':'remote_failed','rc':q.returncode,'stderr':q.stderr.decode(errors='replace')[:2000]};p.write_text(json.dumps(data,indent=2)+'\n')
 proof={'started_unix':start,'finished_unix':end,'rc':q.returncode,'stderr':q.stderr.decode(errors='replace'),'route':argv,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'observation_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'remote_writes':False,'raw_log_model_transferred':False}
 e.write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps({'observation':str(p),'bytes':p.stat().st_size,'rc':q.returncode,'cst':data.get('observed_cst'),'curves':data.get('curve_summary'),'original_pids':data.get('original_pids'),'fatal_count':len(data.get('recent_fatal_lines',[]))},ensure_ascii=False))
 return q.returncode
if __name__=='__main__':raise SystemExit(main())
