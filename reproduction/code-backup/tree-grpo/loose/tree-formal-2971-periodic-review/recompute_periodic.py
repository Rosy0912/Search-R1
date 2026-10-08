import pathlib,json,hashlib,re,math
R=pathlib.Path(__file__).resolve().parents[2];P=R/'outputs/evidence/tree-formal-2971-periodic50-100-prefix.json';x=json.loads(P.read_text());f=x['files'];sha=lambda b:hashlib.sha256(b).hexdigest()
for n,v in f.items():assert sha(v['text'].encode())==v['sha256']
cs=f['resolved.yaml']['sha256'];assert f['output/entry-audit/entry-config.yaml']['sha256']==cs
entry=json.loads(f['output/entry-audit/entry.jsonl']['text']);assert entry['actual_config_sha256']==entry['config_sha256']==cs and entry['commit']=='4d133efeb1f675139d2558640f1e21d50780f133'
logs=x['selected_stage_log'];inst=json.loads(next(z.split('TREE_VALIDATION_COST_INSTALLED_JSON=',1)[1] for z in logs if 'TREE_VALIDATION_COST_INSTALLED_JSON=' in z));assert inst['observer_source_sha256']==sha((R/'work/validation_cost_hooks.py').read_bytes())
metrics={}
for line in logs:
 m=re.search(r'\bstep:(50|100) - (.*)',line)
 if m:
  vals={}
  for item in m[2].split(' - '):k,v=item.rsplit(':',1);vals[k]=float(v);assert math.isfinite(vals[k])
  metrics[int(m[1])]=vals
assert set(metrics)=={50,100}
validation=[]
for n,v in f.items():
 if '/validation-cost/' not in n or not n.endswith('.json'):continue
 s=json.loads(v['text']);raw=f[n+'l'];assert s['ledger_sha256']==raw['sha256'];ledger=[json.loads(z) for z in raw['text'].splitlines()]
 assert s['status']=='VALIDATION_COST_COMPLETE' and s['dataset_rows']==s['evaluated_rows']==64 and s['pid']==inst['pid']==3489451 and s['observer_source_sha256']==inst['observer_source_sha256'] and s['job_id']=='2971' and s['error_type'] is None
 assert [z['event'] for z in ledger]==list(range(1,len(ledger)+1)) and all(z['pid']==s['pid'] and z['global_step']==s['global_step'] for z in ledger)
 ev=lambda kind:[z for z in ledger if z['kind']==kind];c=s['counters']
 countfields={'generation_attempts':len(ev('generation_start')),'generation_completed':len(ev('generation_completed')),'http_attempts':len(ev('http_start')),'http_completed':len(ev('http_completed'))}
 for field,kind,key in [('submitted_rows','generation_start','rows'),('returned_rows','generation_completed','rows'),('submitted_prefill_token_events','generation_start','prefill_token_events'),('returned_nonpad_model_token_events','generation_completed','nonpad_model_token_events'),('real_sample_rows','real_sample_batch','rows'),('active_row_events','gpu_padding_submission','active_rows'),('submitted_padding_row_events','gpu_padding_submission','padding_rows'),('http_query_events','http_start','queries'),('http_returned_query_rows','http_completed','query_result_rows')]:countfields[field]=sum(z[key] for z in ev(kind))
 for k,v in countfields.items():assert c[k]==v
 assert c['http_failed']==c['generation_failed']==0 and c['http_attempts']==c['http_completed'] and c['generation_attempts']==c['generation_completed']
 row=dict(step=s['global_step'],phase=s['phase'],sequence=s['validation_sequence'],pid=s['pid'],counters=c,wall_seconds=s['wall_seconds'],ledger_sha256=s['ledger_sha256'])
 if s['global_step'] in metrics:
  m=metrics[s['global_step']];counts={k.split('/')[-1]:v for k,v in m.items() if k.startswith('val/evaluated_count/')};assert len(counts)==7 and sum(counts.values())==m['val/qa_count/all']==64 and all(m['val/qa_count/'+d]==v for d,v in counts.items())
  for k in ['f1','em','answer_format','format_rate','generated_tokens','search_calls']:assert abs(sum(v*m['val/'+k+'/'+d] for d,v in counts.items())/64-m['val/'+k+'/all'])<1e-12
  row.update(counts=counts,f1=m['val/f1/all'],em=m['val/em/all'],em_correct=round(64*m['val/em/all']))
  assert m['training/actual_outer_updates']==s['global_step'] and m['actor/optimizer_steps']==12
 validation.append(row)
assert {s['step'] for s in validation}=={0,50,100}
rank=[json.loads(v['text']) for n,v in f.items() if '/tree-rank-audit/' in n];assert len(rank)==8 and {v['rank'] for v in rank}==set(range(8));deltas={}
for v in rank:
 assert v['outer_step']==100 and v['job_id']=='2971' and v['config_sha256']==cs and v['code_commit']==entry['commit'] and v['world_size']==8 and v['successful_optimizer_calls']==v['finite_grad_norms']==v['nonzero_grad_norms']==12 and v['skipped_empty_minibatches']==0
 assert v['actor_source_sha256']==sha((R/'work/tree-grpo/verl/workers/actor/dp_actor.py').read_bytes()) and v['auditor_source_sha256']==sha((R/'work/tree-grpo/search_r1/baselines/tree_rank_audit.py').read_bytes())
 assert v['local_shard_sample_delta']['finite']==8192 and v['local_shard_sample_delta']['nonzero']>0 and v['local_shard_sample_layout_unchanged'] and all(math.isfinite(z) and z>0 for z in v['existing_grad_norms']);deltas[v['rank']]=v['local_shard_sample_delta']['nonzero']
state=json.loads(f['output/checkpoints/tree_grpo-qwen2.5-3b/algorithm/global_step_100/state.json']['text']);assert state['method']=='tree_grpo' and state['global_step']==100 and state['resume_supported'] is False
import yaml
cfg=yaml.safe_load(f['resolved.yaml']['text']);assert state['base_commit']==cfg['baselines']['base_commit'] and state['algorithm_source_commit']==cfg['baselines']['algorithm_source_commit'] and state['teacher_path'] is None
assert cfg['trainer']['total_training_steps']==501 and int(501*cfg['actor_rollout_ref']['actor']['optim']['lr_warmup_steps_ratio'])==142
for step,m in metrics.items():assert abs(m['actor/lr']-step*1e-6/142)<1e-22
idx=json.loads(f['output/checkpoints/tree_grpo-qwen2.5-3b/actor/global_step_100/model.safetensors.index.json']['text']);shards=set(idx['weight_map'].values());assert len(idx['weight_map'])==435 and len(shards)==3 and all(x['checkpoint100_files'][n]>0 for n in shards) and len(x['checkpoint100_files'])==12
H=R/'outputs/evidence/tree-formal-2971-heartbeat-20261003-1952.json';h=json.loads(H.read_text());assert h['actual_outer_updates']==124 and h['rank_records_complete']==992 and h['all_completed_rank_calls_finite_nonzero'] and h['all_console_metrics_finite']
out=dict(status='PERIODIC50_100_PREFIX_VERIFIED_NOT_TERMINAL_ACCEPTANCE',prefix_sha256=sha(P.read_bytes()),config_sha256=cs,validations=validation,rank100_nonzero_sample_counts=deltas,checkpoint100=dict(algorithm_step=100,index_keys=435,shards=sorted(shards),presence_file_count=12,full_reload_proven=False),heartbeat=dict(sha256=sha(H.read_bytes()),actual_outer_updates=124,rank_records_reported=992,scope='Owner checked heartbeat aggregate; this snapshot does not contain all992rawrankrecords for independent reread'),limitations=['64question monitoring; no raw answers for independent rescoring, no GRPOeffectcomparison.','HF10012file sizes/index presence only, not fullweightsfinite/reload/forward proof.','Final500/final51713/HF500/cleanup pending; no terminal acceptance.'])
(R/'work/tree-formal-2971-periodic-review/PERIODIC_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
