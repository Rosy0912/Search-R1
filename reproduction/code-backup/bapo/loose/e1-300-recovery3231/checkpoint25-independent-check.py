"""Local independent identity/hash-chain/stage replay: no GPU or resume release."""
import pathlib,json,hashlib,math
B=pathlib.Path(__file__).resolve().parents[2]
C=B/'work/evidence/formal3231-checkpoint25-fullhash-cpu-candidate.json'
RAW=B/'work/evidence/formal3231-fault-raw-20261005T0918'
d=json.loads(C.read_text());run=pathlib.Path(d['run']);rawrun=RAW/'runs'/run.name
assert d['status']=='FULL25_HASH_AND_CPU_DRIVER_CANDIDATE_NOT_NEW_PROCESS_RELOAD'
assert d['source']=='39cdcdd1404213b897bb0de99cc52d72707682f0' and run.name=='bapo-instruct300-formal300-3231'
assert d['step']==25 and d['next_step']==26 and d['sampler_cursor']==12800 and d['sampler_epoch']==0
assert d['generation_rng_in_old_checkpoint']is False
entries={x['path']:x for x in d['files']};assert len(entries)==len(d['files'])==31
hf={'added_tokens.json','config.json','generation_config.json','merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'}
expected={str(run/'checkpoints/actor/global_step_25'/x)for x in hf}
expected|={str(run/'checkpoints/actor/global_step_25/training_state'/f'rank_{rank}.{ext}')for rank in range(8)for ext in ('pt','json')}
expected|={str(run/'checkpoints/algorithm/global_step_25/state.json'),str(run/'checkpoints/full_state/global_step_25/driver-reload.json'),str(run/'checkpoints/full_state/global_step_25/driver.pt')}
assert set(entries)==expected
checked=[]
for name,x in entries.items():
 assert type(x['bytes'])is int and x['bytes']>0 and len(x['sha256'])==64
 path=RAW/pathlib.Path(name).relative_to(run.parent.parent)
 if path.exists():
  assert len(path.read_bytes())==x['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==x['sha256'];checked.append(name)
assert len(d['native_reports'])==8
for rank,r in enumerate(d['native_reports']):
 actual=json.loads((rawrun/f'checkpoints/actor/global_step_25/training_state/rank_{rank}.json').read_text())
 assert r==actual and r['rank']==rank and r['world_size']==8 and r['step']==r['scheduler_last_epoch']==25
 assert r['status']=='FULL_STATE_SERIALIZED_OK' and r['model_state_dict_type']=='SHARDED_STATE_DICT'
 assert r['model_dtensor_entries']==435 and r['optimizer_state_entries']==37
 assert entries[r['path']]['sha256']==r['sha256']
reload=json.loads((rawrun/'checkpoints/full_state/global_step_25/driver-reload.json').read_text())
assert reload['state_digest']==d['driver_digest'] and reload['sha256']==entries[reload['path']]['sha256'] and reload['step']==25 and reload['sampler_cursor']==12800
assert hashlib.sha256((rawrun/'resolved-config.yaml').read_bytes()).hexdigest()==d['config_sha256']
assert (rawrun/'source-commit.txt').read_text().strip()==d['source']
state=json.loads((rawrun/'checkpoints/algorithm/global_step_25/state.json').read_text())
assert state['bapo']==d['bapo'] and state['algorithm_source_commit']==d['source'] and state['actual_outer_updates']==25 and state['resume_supported']is False
train=[]
for line in (rawrun/'train.log').read_text().splitlines():
 if 'BAPO_TRAIN_METRICS_JSON='in line:train.append(json.loads(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]))
assert [e['step']for e in train[:25]]==list(range(1,26))
best=-math.inf;stalled=0;early=True
for e in train[:25]:
 m=e['metrics'];score=m['bapo/stage_validation'];assert m['bapo/stage_evaluated_count']==64
 if score>best:best=score;stalled=0
 else:stalled+=1
 if stalled>=5:early=False
 assert m['bapo/stalled']==stalled and bool(m['bapo/early_stage'])==early and m['bapo/actual_outer_updates']==e['step']
assert d['bapo']==dict(best=best,stalled=stalled,early=early,patience=5)
r=json.loads((rawrun/'logs/step5-independent-release.json').read_text())
assert r['status']=='PASS_INDEPENDENT_E1_STEP5_REVIEW' and r['run_root']==str(run) and r['source_commit']==d['source'] and r['actual_job_id']==3231 and r['horizon']==300 and r['warmup_steps']==114
available=[];large_missing=[]
for x in r['evidence']+[r['independent_review']]:
 path=RAW/pathlib.Path(x['path']).relative_to(run.parent.parent)
 if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==x['sha256'];available.append(x['path'])
 else:large_missing.append(x['path'])
print(json.dumps(dict(status='PASS_LOCAL_CHECKPOINT25_CANDIDATE_IDENTITY_AND_RAW_CHAIN_NOT_RESTORE',candidate_sha256=hashlib.sha256(C.read_bytes()).hexdigest(),files31_total_bytes=sum(x['bytes']for x in entries.values()),raw_small_files_independently_rehashed=len(checked),native_reports8_match=True,driver_cpu_digest_matches_producer=True,stage25_replayed=dict(best=best,stalled=stalled,early=early),source_and_config_match=True,first5_available_bindings_verified=len(available),first5_nonarchived_large_bindings=large_missing,generation_rng_recorded=False,new_process_native_reload_accepted=False),indent=2))
