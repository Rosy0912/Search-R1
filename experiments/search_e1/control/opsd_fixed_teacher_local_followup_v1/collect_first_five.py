"""Read native receipts/scalars only, not optimizer tensors or raw token contents."""
import subprocess,json,sys,pathlib
script=r'''import pathlib,json,hashlib,math,time,shutil
b=pathlib.Path('/data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010');cfg=json.loads((b/'receipts/training-config.json').read_text());rows=[];previous=None;updates=0
for i in range(5):
 o=b/'runs/outers'/f'outer-{i:05d}';d=json.loads((o/'training/outer-complete.json').read_text());assert d['accepted'] and d['optimizer_updates']==updates+math.ceil(d['K']/32);updates=d['optimizer_updates'];ref=d['checkpoint'];raw=pathlib.Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];cp=json.loads(raw)
 assert cp['complete_rank_count']==8 and cp['equal_optimizer_steps'] and len(set(cp['all_rank_adapter_sha256']))==1 and len(set(cp['all_rank_optimizer_tensor_sha256']))==1 and cp['all_rank_adapter_sha256'][0]!=cfg['common_initial_adapter_tensor_sha256']
 sampling=json.loads((o/'sampling-complete.json').read_text());identity=sampling['sampling_model_identity'];assert identity['kind']=='raw-source100' and identity['optimizer_updates']==0 and identity['checkpoint'] is None and identity['adapter_tensor_sha256']==cfg['common_initial_adapter_tensor_sha256'];assert all(x['actual_wait_returncode']==0 for x in sampling['exit_proof']['child_exit_evidence']) and sampling['exit_proof']['gpu_release']['all_owned_release']['accepted']
 manifest=json.loads((o/'outer-manifest.json').read_text());assert manifest['previous_complete_checkpoint']==previous and manifest['outer_old_adapter']==(previous or cfg['common_initial_adapter_manifest']);previous=ref
 diagnostics=json.loads((o/'teacher-diagnostics/paired/complete.json').read_text());assert diagnostics['accepted']
 exits=[json.loads(x.read_text()) for x in (o/'trainer-logs').glob('attempt-*-exit.json')];success=[x for x in exits if x.get('accepted') and all(y['actual_wait_returncode']==0 for y in x['child_exit_evidence'])];assert success and success[-1]['gpu_release']['all_owned_release']['accepted']
 scalars=[]
 for r in cp['outer_update_receipts']:
  v=json.loads(pathlib.Path(r['path']).read_text());assert v['accepted'];pairs=[p for rank in v['all_rank_actual_token_metrics'] for p in rank['pairs']];assert len(pairs)==v['global_actual_pairs'];assert all(math.isfinite(p[k]) for p in pairs for k in ('raw_loss','post_loss','weighted_post_loss'))
  scalars.extend(pairs)
 assert len(scalars)==d['K']
 rows.append(dict(outer=i+1,K=d['K'],updates=updates,policy_tokens=d['global_valid_tokens'],checkpoint=ref,native_all_rank_adapter_digest=cp['all_rank_adapter_sha256'][0],native_all_rank_optimizer_digest=cp['all_rank_optimizer_tensor_sha256'][0],fixed_teacher_sampling=True,outer_old_previous_student=True,native_successful_wait_release=True,diagnostic_cases=diagnostics['count'],recorded_raw_trajectory_mean=sum(p['raw_loss'] for p in scalars)/len(scalars),recorded_weighted_post_trajectory_mean=sum(p['weighted_post_loss'] for p in scalars)/len(scalars)))
s=json.loads((b/'runs/status.json').read_text());six=b/'runs/outers/outer-00005';files=list(six.glob('sampling-attempt-*/generation-trace-*.jsonl'));actual=[]
for p in files:
 if p.stat().st_size:
  with p.open() as f:x=json.loads(next(f))
  actual.append(dict(file=str(p),bytes=p.stat().st_size,first_actual_generated_tokens=len(x['generated_token_ids']),sampling_model_identity=x['sampling_model_identity']))
assert actual and all(x['first_actual_generated_tokens']>0 for x in actual)
m=json.loads((b/'receipts/memory-live.json').read_text());assert time.time()-m['updated_unix']<30 and m['samples_below_limit']==0
print(json.dumps(dict(accepted_initial_five=True,full_goal_complete=False,total_budget=200,run=cfg['run_id'],config_sha256=hashlib.sha256((b/'receipts/training-config.json').read_bytes()).hexdigest(),rows=rows,total_pairs=sum(x['K'] for x in rows),total_policy_tokens=sum(x['policy_tokens'] for x in rows),real_Adam=updates,sixth_actual_generation=actual,current_phase=s['phase'],current_outer_index=s.get('outer_index'),actual_owner=s['owner'],memory_age=time.time()-m['updated_unix'],memory_below=0,disk_free_bytes=shutil.disk_usage(b).free,checked_unix=time.time(),scope='Native receipt/scalar readback; no independent optimizer tensor deserialization, no logits recomputation, no population EM')))
'''
r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=8','hkust1','python3','-'],input=script,text=True,capture_output=True)
if r.returncode:sys.stderr.write(r.stderr);sys.exit(r.returncode)
v=json.loads(r.stdout);p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v[k] for k in ('accepted_initial_five','total_pairs','real_Adam','current_phase','current_outer_index')}))
