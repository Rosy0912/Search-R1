"""CPU-only independent raw inventory/contracts; never an allocation/teacher approval."""
import argparse,hashlib,json,os,pathlib,time
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def need(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(p.read_text())
def inspect(directory):
 d=pathlib.Path(directory).resolve();files={};checks={};errors=[];pending=[]
 def artifact(name):
  p=d/name
  if not p.is_file():pending.append(name);return None
  need(not p.is_symlink(),'symlink raw artifact: '+name)
  files[name]={'size':p.stat().st_size,'sha256':sha(p)};return p
 def check(name,fn):
  try:checks[name]=fn()
  except Exception as e:errors.append({'check':name,'error':str(e)})
 configp=artifact('frozen-input-config.json');config=read(configp) if configp else None
 tp=artifact('generation-trace.json');trace=read(tp) if tp else None
 if trace:
  def generations():
   need(trace['nq']==300 and len(trace['questions'])==300 and len(set(trace['questions']))==300,'not300 unique actual questions')
   need(len(trace['gens'])==1500 and trace['flat_qi']==[i for i in range(300) for _ in range(5)],'not1500 original group-ordered generations')
   need(trace['valid']==[i for i,g in enumerate(trace['gens']) if '<answer>' in g],'valid filtering differs original answer-tag rule')
   n=len(trace['valid']);need(n>0,'no valid original rollouts')
   for k in ('commit_ctx','ans_ctx','flip_ctx','flip2_ctx','answers','ems','flip_ans','flips','flips2'):need(len(trace[k])==n,'unaligned trace '+k)
   for k in ('q_cb','ems','flips','flips2'):need(all(type(v)==int and v in (0,1) for v in trace[k]),'nonbinary '+k)
   need(len(trace['cb_ans'])==len(trace['q_cb'])==300,'closedbook question count')
   return dict(questions=300,generated=1500,valid=n,filtered=1500-n,actual_question_sha256=hashlib.sha256(json.dumps(trace['questions'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
  check('generation_structure',generations)
 tokenp=artifact('hidden-input-token-trace.json')
 if tokenp and trace:
  def masks():
   tokens=read(tokenp);need(set(tokens)=={'commit_ctx','ans_ctx','prompts'},'hidden-token key drift');out={}
   for key,batches in tokens.items():
    rows=0;active=0
    for batch in batches:
     ids=np.asarray(batch['input_ids']);mask=np.asarray(batch['attention_mask']);need(ids.ndim==2 and ids.shape==mask.shape and 1<=ids.shape[0]<=8 and ids.shape[1]<=4096,'token shape/length')
     need(np.issubdtype(ids.dtype,np.integer) and (ids>=0).all() and (ids<152064).all(),'token vocab')
     need(np.isin(mask,[0,1]).all() and (mask.sum(1)>0).all() and (np.diff(mask,axis=1)>=0).all(),'left-padding mask')
     rows+=len(ids);active+=int(mask.sum())
    need(rows==len(trace[key]),'hidden token row mismatch '+key);out[key]={'rows':rows,'active_tokens':active}
   return out
  check('hidden_token_masks',masks)
 gp=artifact('generation-token-trace.json')
 if gp:
  def generated_tokens():
   calls=read(gp);need(calls,'empty token calls');rows=0;count=0
   for call in calls:
    need(len(call['prompts'])==len(call['outputs']),'generation call row alignment')
    for row in call['outputs']:
     for key in ('prompt_token_ids','token_ids'):need(all(type(t)==int and 0<=t<152064 for t in row[key]),'generated vocab')
     rows+=1;count+=len(row['token_ids'])
   return {'calls':len(calls),'rows_including_counterfactuals':rows,'generated_tokens':count,'text_decode_replay':'PENDING_EXPLICIT_OFFICIAL_TOKENIZER'}
  check('generation_tokens',generated_tokens)
 resultp=artifact('result.json');result=read(resultp) if resultp else None
 if config:
  tag=config['dump_args']['tag'];offset=config['dump_args']['offset'];npz=artifact(f'dump/{tag}_{offset}.npz');jl=artifact(f'dump/{tag}_{offset}.jsonl')
  if npz and trace:
   def arrays():
    expected={'y_em','y_flip','y_flip2','q_cb','gid','ds'}|{f'{p}_{L}' for p in ('Xc','Xa','Xp') for L in (21,24,27)};n=len(trace['valid'])
    with np.load(npz,allow_pickle=False) as a:
     need(set(a.files)==expected,'NPZ does not have exactly15 original fields');stats={}
     expected_values={'y_em':trace['ems'],'y_flip':trace['flips'],'y_flip2':trace['flips2'],'q_cb':[trace['q_cb'][trace['flat_qi'][i]] for i in trace['valid']],'gid':[trace['flat_qi'][i] for i in trace['valid']]}
     for k in a.files:
      v=a[k];need(v.shape[0]==n,'NPZ row mismatch '+k)
      if k.startswith(('Xc_','Xa_','Xp_')):need(v.shape==(n,5120) and v.dtype==np.float16 and np.isfinite(v).all(),'hidden finite/shape/dtype '+k)
      elif k=='ds':need(v.shape==(n,) and all(str(t)==tag for t in v),'dataset identity')
      else:need(v.shape==(n,) and v.dtype==(np.int32 if k=='gid' else np.int8) and np.array_equal(v,expected_values[k]),'original label/gid mismatch '+k)
      stats[k]={'shape':list(v.shape),'dtype':str(v.dtype)}
    if result:need(result['npz_sha256']==sha(npz) and jl and result['jsonl_sha256']==sha(jl),'declared output SHA drift')
    return stats
   check('npz_original15',arrays)
 for phase in 'AB':
  receiptp=artifact(f'phase-{phase}-receipt.json');barrierp=artifact(f'phase-{phase}-owner-barrier.json')
  if receiptp and barrierp and config:
   def phase_check(rp=receiptp,bp=barrierp,phase=phase):
    r,b=read(rp),read(bp);need(r['phase']==phase and r['config_sha256']==sha(configp) and b['config_sha256']==sha(configp),'phase/config SHA')
    need(set(r['uuids'])==set(config['uuids']) and len(set(r['uuids']))==8,'actual8phase UUID')
    need(b['returncode']==0 and b['own_processes_remaining']==b['own_gpu_records_remaining']==[] and b['baseline_preserved'] and b['original_global_gpu_baseline_preserved'],'phase cleanup/baseline')
    need(set(b['seen_actual_gpu_uuids'])==set(config['uuids']),'actual all8 owner seen')
    return {'pid':r['pid'],'phase':phase,'observed_actual8':True,'exit_returncode':0}
   check('phase_'+phase,phase_check)
 return {'scope':'READ_ONLY_RAW_CONTRACTS_NOT_INDEPENDENT_APPROVAL_NO_TRAINING','status':'INCOMPLETE' if pending else ('CONTRADICTIONS' if errors else 'CONTRACTS_CHECKED_UNREVIEWED'),'directory':str(d),'raw_files':files,'checks':checks,'errors':errors,'missing':pending,'manual_pending':['original source/official24HF identity','tokenizer decoding and original EM/flip replay','head grouped-CV/fitted predict_proba provenance','three real HTTP gates and every runtime HTTP200','READY/EXIT/ACK fullfresh guardian/raw resources/final foreign baseline']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();began=time.monotonic();r=inspect(a.directory);r['reader_source_sha256']=sha(__file__);r['elapsed_seconds']=time.monotonic()-began
 # The only mutation is the caller-selected new report, never an input artifact.
 fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(r,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':r['status'],'errors':r['errors'],'missing':r['missing']}))
if __name__=='__main__':main()
