"""Read-only persistence/recoverability audit, not a reward replay or correlation.

No Torch, model inference, GPU, service call, signal or Node write. Raw inputs,
features, weights, question text and gold answers never leave the Node. Default
reads metadata + all small native gate files + first/last full trace schemas;
--verify-trace-sha additionally streams each committed raw trace for fresh SHA.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, datetime, hashlib, json, os, socket, sys

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def meta(p):
 if not p.is_file():return dict(path=str(p),exists=False)
 s=p.stat();return dict(path=str(p),exists=True,bytes=s.st_size,uid=s.st_uid,mtime_ns=s.st_mtime_ns)
def find_inputs(config,run,base):
 vals=config.get('data',{}).get('train_files',[])
 if isinstance(vals,str):vals=[vals]
 result=[]
 for v in vals:
  p=Path(v)
  if not p.is_absolute():p=base/'repo'/p
  result.append(p)
 return result
def trace_schema(p):
 j=json.loads(p.read_bytes());rows=j['actor_rows'];ts=j['trajectories'];parents=j['parent_group_ids'];byid={t['trajectory_id']:t for t in ts}
 length_bad=mask_bad=identity_bad=0;turns=obs=correction=0;rowkeys=Counter();tkeys=Counter();turnkeys=Counter();field_names=set();group_hashes=[];uidset=set()
 for i,t in enumerate(ts):
  tkeys[','.join(sorted(t))]+=1
  identity_bad+=int(t.get('parent_occurrence_index')!=i//5 or t.get('sample_index')!=i%5)
  for turn in t['turns']:
   turns+=1;turnkeys[','.join(sorted(turn))]+=1;field_names.update(turn);obs+=len(turn.get('observation_token_ids',[]));correction+=len(turn.get('correction_token_ids',[]))
 for i,row in enumerate(rows):
  rowkeys[','.join(sorted(row))]+=1;field_names.update(row)
  sizes=[len(row[x]) for x in ['input_ids','attention_mask','info_mask','position_ids']]
  length_bad+=int(len(set(sizes))!=1 or not 0<=row['prompt_width']<=sizes[0]);mask_bad+=sum(x not in [0,1] for k in ['attention_mask','info_mask'] for x in row[k])
  identity_bad+=int(row['trajectory_id'] not in byid);uidset.add(row['trajectory_id'])
 for parent in parents:group_hashes.append(parent.split('-',1)[1])
 return dict(path=str(p),run_id=j['run_id'],logical_update=j['logical_update'],top_level_fields=sorted(j),actor_row_count=len(rows),trajectory_count=len(ts),parent_count=len(parents),unique_trajectory_ids=len(uidset),actor_row_key_schemas=dict(rowkeys),trajectory_key_schemas=dict(tkeys),turn_key_schemas=dict(turnkeys),all_recorded_field_names=sorted(field_names),turn_count=turns,observation_token_count=obs,correction_token_count=correction,bad_aligned_row_lengths=length_bad,bad_mask_elements=mask_bad,bad_group_identity_count=identity_bad,explicit_reward_or_probe_or_advantage_fields=[k for k in field_names|set(j) if any(x in k.lower() for x in ['reward','advantage','boundary','flip','ground_truth'])],raw_tokens_and_masks_present=True,complete_reward_model_call_input_reconstruction_requires_frozen_tokenizer_and_decode_segments=True,parent_index_hashes=group_hashes)

ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--max-step',type=int,default=150);ap.add_argument('--verify-trace-sha',action='store_true');args=ap.parse_args();b=Path(args.base);runp=b/'logs/run-manifest.json';run=json.loads(runp.read_bytes());assert run['run_id']=='db474b6abf654ad9892b02ae67ab6f86';assert args.max_step<=150
cfgp=b/'logs/resolved-training-config.json';cfgraw=cfgp.read_bytes();cfgproof=json.loads(cfgraw);assert cfgproof['run_id']==run['run_id'];config=cfgproof['resolved_config'];assert isinstance(config,dict) and isinstance(config.get('data'),dict);assert hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()==cfgproof['resolved_config_sha256']
out=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),run_id=run['run_id'],source_commit=run['source_commit'],scope_steps=[1,args.max_step],readonly=True,no_reward_replay=True,source=[],raw_reward_persistence_status='NOT_YET_ASSERTED',trajectory_coverage=[],trace_schema_reads=[],probe_version_assets=[],training_gold_identity={},directory_inventory=[])
for rel in ['reproduction/common8/behavior_trace.py','reproduction/common8/behavior_runtime.py','verl/trainer/main_ppo_eco.py','probe_server.py','verl/utils/dataset/rl_dataset.py','search_g1/signals.py','reproduction/common8/behavior_probability.py','reproduction/common8/configuration.py']:
 p=b/'repo'/rel;out['source'].append(dict(**meta(p),fresh_sha256=sha(p),run_bound_sha256=run['source_sha256'].get(rel)))
out['all_listed_source_sha_equal_run_binding']=all(x['fresh_sha256']==x['run_bound_sha256'] for x in out['source']);assert out['all_listed_source_sha_equal_run_binding'];out['resolved_configuration_identity']=dict(path=str(cfgp),fresh_file_sha256=hashlib.sha256(cfgraw).hexdigest(),verified_resolved_config_sha256=cfgproof['resolved_config_sha256'],run_id=cfgproof['run_id'])
out['static_resource_binding']={k:run.get(k) for k in ['execution_backend','execution_id','execution_host','training_physical_indices','auxiliary_physical_indices','retriever_physical_indices','probe_physical_index','excluded_physical_indices','probe_visible_device','retrieval_placement','pipeline_identity']}
out['static_resource_binding']['run_manifest_owner_uid']=runp.stat().st_uid
out['static_resource_binding']['no_free_memory_or_idle_gpu_implies_authorization']=True
pdir=Path('/proc/1338978')
try:
 cwd=str((pdir/'cwd').resolve());cmd=(pdir/'cmdline').read_bytes().replace(b'\0',b' ').decode();st=(pdir/'stat').read_text();tail=st[st.rfind(')')+2:].split()
 if cwd.startswith(str(b)) and 'main_ppo' in cmd:
  env=dict(x.split(b'=',1) for x in (pdir/'environ').read_bytes().split(b'\0') if b'=' in x)
  out['static_resource_binding']['observed_owned_train_environment']=dict(pid=1338978,start_ticks=int(tail[19]),cwd=cwd,uid=pdir.stat().st_uid,CUDA_VISIBLE_DEVICES=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode(),only_whitelisted_environment_field_read=True)
except OSError as e:out['static_resource_binding']['owned_train_environment_status']=type(e).__name__
up=b/'logs/actual-updates.jsonl';ub=up.read_bytes();updates=[]
for line in ub.splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if r.get('run_id')==run['run_id'] and 1<=r.get('step',0)<=args.max_step:updates.append(r)
assert len({r['step'] for r in updates})==len(updates)
out['update_ledger']=dict(**meta(up),fresh_read_sha256=hashlib.sha256(ub).hexdigest(),complete_scope_rows=len(updates),completed_steps=[r['step'] for r in updates],source_metric_names=sorted({k for r in updates for k in r.get('metrics',{})}),future_uncompleted_requested_steps=[k for k in range(1,args.max_step+1) if k not in {r['step'] for r in updates}])
all_group_hashes=set();trace_paths=[]
for r in updates:
 aid=r['attempt_id'];p=b/'artifacts/behavior-traces'/('attempt-'+aid+'.json');gp=b/'logs'/('behavior-probability-attempt-'+aid+'.json');graw=gp.read_bytes();g=json.loads(graw)
 row=dict(step=r['step'],attempt_id=aid,probe_version=r['probe_version'],trace=meta(p),trace_sha256_bound_native_record=r['behavior_trace_sha256'],native_gate=meta(gp),native_gate_fresh_sha256=hashlib.sha256(graw).hexdigest(),native_gate_sha_matches_update=hashlib.sha256(graw).hexdigest()==r['behavior_probability_gate_sha256'],native_gate_trace_sha_matches_update=g.get('private_trace_sha256',g.get('trace_sha256'))==r['behavior_trace_sha256'],native_gate_run_and_step_match=g['run_id']==run['run_id'] and g['logical_update']==r['step'],native_gate_accepted=g.get('accepted'),native_recorded_trajectory_count=g.get('trajectory_count'),native_recorded_actor_count=len(g.get('actor_token_counts',[])),native_recorded_unique_trajectory_ids=len(set(g.get('trajectory_ids',[]))),native_gate_top_fields=sorted(g),raw_reward_fields_in_native_gate=[k for k in g if any(x in k.lower() for x in ['reward','advantage','boundary','flip','ground_truth'])])
 if args.verify_trace_sha and p.is_file():row['trace']['fresh_full_sha256']=sha(p);row['trace']['fresh_sha_matches_update']=row['trace']['fresh_full_sha256']==r['behavior_trace_sha256']
 values=set(g.get('upstream_group_sha256',{}).values());all_group_hashes.update(values);row['native_recorded_unique_upstream_index_hashes']=len(values)
 out['trajectory_coverage'].append(row);trace_paths.append(p)
out['committed_trace_shape_claim_verified_by_fresh_native_gates']=all(r['native_gate_accepted'] and r['native_gate_run_and_step_match'] and r['native_gate_sha_matches_update'] and r['native_gate_trace_sha_matches_update'] and r['native_recorded_trajectory_count']==2560 and r['native_recorded_actor_count']==2560 and r['native_recorded_unique_trajectory_ids']==2560 and r['trace']['exists'] for r in out['trajectory_coverage'])
out['total_committed_trace_bytes']=sum(x['trace'].get('bytes',0) for x in out['trajectory_coverage']);out['total_committed_trajectory_slots']=len(updates)*2560
for p in list(dict.fromkeys(trace_paths[:1]+trace_paths[-1:])):
 schema=trace_schema(p);all_group_hashes.update(schema.pop('parent_index_hashes'));out['trace_schema_reads'].append(schema)
out['all_committed_raw_trace_bytes_fresh_hashed']=args.verify_trace_sha;out['all_fresh_hashed_traces_match_native_records']=all(r['trace'].get('fresh_sha_matches_update',False) for r in out['trajectory_coverage']) if args.verify_trace_sha else None
for root in [b/'artifacts',b/'logs',b/'data',b/'checkpoints']:
 if not root.exists():continue
 files=[p for p in root.rglob('*') if p.is_file()]
 out['directory_inventory'].append(dict(path=str(root),total_file_count=len(files),total_bytes=sum(p.stat().st_size for p in files),file_suffix_counts=dict(Counter(p.suffix for p in files)),reward_advantage_probe_named_files=[str(p) for p in files if any(t in p.name.lower() for t in ['reward','advantage','score','probe'])][:300]))
needed_versions={r['probe_version'] for r in updates}
for d in sorted((b/'artifacts').glob('round_*')):
 mp=d/'version-manifest.json'
 if not mp.is_file():continue
 m=json.loads(mp.read_bytes())
 if m['version'] not in needed_versions:continue
 rr=dict(version=m['version'],round=m['round'],manifest_path=str(mp),manifest_fresh_sha256=sha(mp),manifest_keys=sorted(m),head_files=[],encoder_path=m['encoder'],encoder_files=[],encoder_hash_coverage_note='Actual encoder files exist and are compared by filename/size here; full encoder weight SHA replay is not performed by this metadata audit.')
 for kind in ['commit','boundary']:
  p=Path(m[kind]);rr['head_files'].append(dict(kind=kind,**meta(p),fresh_sha256=sha(p),manifest_sha256=m['head_sha256'][kind+'_probe.pkl']))
 encoder=Path(m['encoder']);expected=m.get('encoder_sha256',{})
 rr['encoder_manifest_sha256_type']=type(expected).__name__
 for p in sorted(encoder.rglob('*')) if encoder.exists() else []:
  if not p.is_file():continue
  row=meta(p);row['relative_path']=str(p.relative_to(encoder));row['manifest_sha256']=expected.get(row['relative_path']) if isinstance(expected,dict) else None
  if p.suffix not in ['.bin','.safetensors','.pt'] and p.stat().st_size<64*1024*1024:row['fresh_sha256']=sha(p)
  rr['encoder_files'].append(row)
 rr['exact_head_sha_verified']=all(x['fresh_sha256']==x['manifest_sha256'] for x in rr['head_files']);rr['encoder_directory_exists']=encoder.is_dir();out['probe_version_assets'].append(rr)
out['every_committed_probe_version_has_actual_manifest_and_verified_heads']={x['version'] for x in out['probe_version_assets']}==needed_versions and all(x['exact_head_sha_verified'] for x in out['probe_version_assets'])
inputs=find_inputs(config,run,b);assert inputs and all(p.is_file() for p in inputs),'HOLD_missing_resolved_train_files';out['training_gold_identity']['resolved_train_paths']=[str(p) for p in inputs];out['training_gold_identity']['run_input_sha256']=run.get('input_sha256')
try:
 import pyarrow.parquet as pq
 mapping=defaultdict(set);index_rows=Counter();files=[]
 for p in inputs:
  table=pq.read_table(p,columns=['extra_info','reward_model']);golds=table.to_pylist()
  files.append(dict(**meta(p),fresh_sha256=sha(p),rows=len(golds),columns=table.column_names))
  for r in golds:
   ix=(r.get('extra_info') or {}).get('index',0);h=hashlib.sha256(canonical(ix)).hexdigest();index_rows[h]+=1;mapping[h].add(hashlib.sha256(canonical(r['reward_model']['ground_truth'])).hexdigest())
 covered=[h for h in all_group_hashes if h in mapping];missing=[h for h in all_group_hashes if h not in mapping];ambiguous=[h for h in covered if len(mapping[h])!=1]
 out['training_gold_identity'].update(files=files,total_data_rows=sum(f['rows'] for f in files),unique_index_hashes=len(mapping),actual_committed_upstream_index_hashes=len(all_group_hashes),hash_join_covered=len(covered),hash_join_missing=len(missing),hash_join_multiple_distinct_gold=len(ambiguous),hash_join_multiple_rows=sum(index_rows[h]>1 for h in covered),gold_mapping_from_persisted_hashed_index_unique=len(missing)==len(ambiguous)==0,raw_index_or_gold_not_exported=True)
except Exception as e:out['training_gold_identity']['controlled_audit_error']=type(e).__name__+': '+str(e)[:300]
out['raw_reward_persistence_status']='NO_EXPLICIT_PER_TRAJECTORY_C_OR_SHAPED_R_FIELDS_IN_AUDITED_NATIVE_SCHEMAS; complete actor token/mask inputs may allow independent historical-version replay, which was not performed.'
out['raw_online_reward_reconstructed']=False;out['full_individual_TRAIN_pearson']='NOT_COMPUTED';out['torch_imported']='torch' in sys.modules;assert not out['torch_imported'];out['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES');out['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(out,ensure_ascii=False))
