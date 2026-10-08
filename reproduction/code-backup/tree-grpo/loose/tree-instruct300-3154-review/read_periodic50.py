from pathlib import Path
import json,hashlib,math,re,subprocess,time
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');R=ROOT/'runs/tree-instruct300-3154';C=ROOT/'instruct300/candidates/tree-f994af3-aa9e3161';proof=[]
def read(p,asjson=True):
 b=p.read_bytes();proof.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)));return json.loads(b) if asjson else b.decode()
M=read(C/'manifest.json');cfg=read(R/'resolved.yaml',False);entrycfg=read(R/'output/entry-audit/entry-config.yaml',False);assert cfg==entrycfg;cfgsha=hashlib.sha256(cfg.encode()).hexdigest();assert cfgsha=='63296d0602c5978d1ff67d258b401054ce587e3a9a2c55052c35fbeac626c198'
entry=read(R/'output/entry-audit/entry.jsonl',False)
reportpaths=[]
for p in (R/'output/validation-cost').glob('*.json'):
 d=json.loads(p.read_bytes())
 if d.get('global_step')==50 and d.get('phase')=='periodic':reportpaths.append(p)
assert len(reportpaths)==1;p=reportpaths[0];report=read(p);ledger=read(p.with_suffix('.jsonl'),False);events=[json.loads(v) for v in ledger.splitlines()];assert hashlib.sha256(ledger.encode()).hexdigest()==report['ledger_sha256']
assert report['global_step']==50 and report['phase']=='periodic' and report['pid']==4048887 and report['job_id']=='3154' and report['run']==str(R/'output') and report['observer_source_sha256']=='7d17b52594ba78542f4dac2c85b136f6cb806935c9b42bd863c6d8eae41d4eef' and report['status']=='VALIDATION_COST_COMPLETE'
assert report['dataset_rows']==report['evaluated_rows']==512 and report['validation_sequence']==2 and report['error_type'] is None
assert [e['event'] for e in events]==list(range(1,len(events)+1)) and all(e['pid']==4048887 and e['global_step']==50 and e['phase']=='periodic' and e['validation_sequence']==2 for e in events) and events[0]['kind']=='validation_start' and events[-1]['kind']=='validation_completed' and events[-1]['evaluated_rows']==512 and not any(e['kind'].endswith('_failed') for e in events)
c=report['counters'];sums=dict(real_sample_rows=sum(e['rows'] for e in events if e['kind']=='real_sample_batch'),submitted_rows=sum(e['rows'] for e in events if e['kind']=='generation_start'),returned_rows=sum(e['rows'] for e in events if e['kind']=='generation_completed'),submitted_prefill_token_events=sum(e['prefill_token_events'] for e in events if e['kind']=='generation_start'),returned_nonpad_model_token_events=sum(e['nonpad_model_token_events'] for e in events if e['kind']=='generation_completed'),active_row_events=sum(e['active_rows'] for e in events if e['kind']=='gpu_padding_submission'),submitted_padding_row_events=sum(e['padding_rows'] for e in events if e['kind']=='gpu_padding_submission'),http_query_events=sum(e['queries'] for e in events if e['kind']=='http_start'),http_returned_query_rows=sum(e['query_result_rows'] for e in events if e['kind']=='http_completed'))
for k,v in sums.items():assert c[k]==v
for label,kind in [('generation_attempts','generation_start'),('generation_completed','generation_completed'),('http_attempts','http_start'),('http_completed','http_completed')]:assert c[label]==sum(e['kind']==kind for e in events)
assert c['real_sample_rows']==512 and c['generation_attempts']==c['generation_completed'] and c['http_attempts']==c['http_completed'] and c['generation_failed']==c['http_failed']==0
# Read a snapshot of append-only stage log; preserve only exact50 lines and file snapshot hash.
text=read(R/'stage.log',False);clean=re.sub(r'\x1b\[[0-9;]*m','',text);lines=[l for l in clean.splitlines() if re.search(r'(?<![\w/])step:50 - ',l)];assert len(lines)==1
metrics={k:float(v) for k,v in re.findall(r'(?:^| - )([^\s]+):([-+0-9.eE]+)(?= - |$)',lines[0])};assert all(math.isfinite(v) for v in metrics.values()) and metrics['training/actual_outer_updates']==50 and metrics['actor/optimizer_steps']==24
assert metrics['val/qa_count/nq']==metrics['val/qa_count/hotpotqa']==metrics['val/evaluated_count/nq']==metrics['val/evaluated_count/hotpotqa']==256 and metrics['val/qa_count/all']==512
for label in ['f1','em','answer_format','format_rate','generated_tokens','search_calls']:
 mean=sum(metrics[f'val/{label}/{k}']*256 for k in ['nq','hotpotqa'])/512;assert abs(mean-metrics[f'val/{label}/all'])<1e-14
state=read(R/'output/checkpoints/tree-instruct300/training_state/global_step_50/manifest.json');identity=state['identity'];assert state['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and identity['completed_step']==50 and identity['actual_update_limit']==300 and identity['job_id']=='3154' and identity['code_commit']==M['code_commit'] and identity['config_sha256']==cfgsha and identity['data_identity']==M['common_data_identity'] and identity['model_identity']==M['model_identity']['sha256']
workers=state['worker_states'];assert len(workers)==8 and {w['identity']['rank'] for w in workers}==set(range(8));statefiles=[]
for w in workers:
 assert all(w['identity'][k]==v for k,v in identity.items()) and w['identity']['world_size']==8 and w['scheduler_last_epoch']==50 and w['optimizer_state_entries']==37
 assert all(abs(v-50e-6/57)<1e-20 for v in w['optimizer_lrs']);q=Path(w['path']);assert q.is_file() and q.stat().st_size>0 and q.parent==R/'output/checkpoints/tree-instruct300/training_state/global_step_50'
 statefiles.append(dict(path=str(q),bytes=q.stat().st_size,declared_sha256=w['sha256'],sha_independently_recomputed=False))
q=Path(state['driver']['path']);assert q.is_file() and q.stat().st_size>0 and state['driver']['next_step']==51;statefiles.append(dict(path=str(q),bytes=q.stat().st_size,declared_sha256=state['driver']['sha256'],sha_independently_recomputed=False))
hfroot=R/'output/checkpoints/tree-instruct300/actor/global_step_50';assert state['hf_checkpoint']==str(hfroot);index=read(hfroot/'model.safetensors.index.json');keys=index['weight_map'];shards=set(keys.values());assert len(shards)==3 and len(keys)==435;hf=[]
assert {str(p.relative_to(hfroot)) for p in hfroot.rglob('*') if p.is_file()}==set(state['hf_identity']['files'])
for name,declared in state['hf_identity']['files'].items():
 q=hfroot/name;assert q.is_file() and q.stat().st_size>0;hf.append(dict(path=str(q),bytes=q.stat().st_size,declared_sha256=declared,sha_independently_recomputed=False))
ranks=[];last={}
for step in range(1,51):
 for rank in range(8):
  a=read(R/f'output/tree-rank-audit/step-{step:06d}-rank-{rank:02d}.json');assert a['outer_step']==step and a['rank']==rank and a['world_size']==8 and a['job_id']=='3154' and a['run_id']==R.name and a['code_commit']==M['code_commit'] and a['config_sha256']==cfgsha and a['data_identity']==M['common_data_identity']
  assert a['actor_source_sha256']==M['files']['source/repo/verl/workers/actor/dp_actor.py'] and a['auditor_source_sha256']==M['files']['source/repo/search_r1/baselines/tree_rank_audit.py'] and (a['local_rows'],a['local_minibatch'],a['local_microbatch'],a['epochs'])==(384,16,2,1)
  assert a['successful_optimizer_calls']==a['finite_grad_norms']==a['nonzero_grad_norms']==len(a['existing_grad_norms'])==24 and a['skipped_empty_minibatches']==0 and all(math.isfinite(v) and v>0 for v in a['existing_grad_norms'])
  d=a['local_shard_sample_delta'];assert d['count']==d['finite']==8192 and a['local_shard_sample_layout_unchanged'] and (d['nonzero']==0 if step==1 else d['nonzero']>0)
  if rank in last:assert a['pid']==last[rank]['pid'] and a['local_shard_sample_before_sha256']==last[rank]['local_shard_sample_after_sha256']
  last[rank]=a;ranks.append(dict(step=step,rank=rank,pid=a['pid'],sample_nonzero=d['nonzero'],calls=24))
slurm=subprocess.check_output(['scontrol','show','job','3154'],text=True);assert 'JobState=RUNNING' in slurm and 'Restarts=0' in slurm
print(json.dumps(dict(status='PERIODIC50_READ_ONLY_RECOMPUTED_PASS',epoch=time.time(),slurm=slurm,manifest=M,config_sha256=cfgsha,entry=entry,report=report,ledger=ledger,metrics=metrics,metric_line=lines[0],state_manifest=state,state_files=statefiles,hf_index=index,hf_files=hf,rank_records_count=len(ranks),ranks=ranks,proof=proof,scope='Actual remote periodic512 and400directrank summaries; checkpoint presence/identity/file-size metadata only, no fullstate50/HF50 rehash or reload')))
