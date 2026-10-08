import pathlib,json,hashlib,re,math
p=pathlib.Path('work/tree-recovery-review/native3234');b=json.loads((p/'ACTUAL_RAW_BUNDLE.json').read_text());f={x['path']:x for x in b['files']}
for r in b['files']:assert hashlib.sha256(r['text'].encode()).hexdigest()==r['sha256']
run=b['run'];tool='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/recovery-502e9fa-f91c4f17';m=json.loads(f[tool+'/manifest.json']['text'])
assert f[tool+'/manifest.json']['sha256']=='f91c4f17a3a3454bb16a81e9ca400f4db07a56a0a27cf6cfac2c1814459c3087'
assert f[tool+'/PRE_DISPATCH_RECEIPT.json']['sha256']=='8f05b47e6e418e93d16483ae6e21d145b657f9903aad81818d0a9fb28a3989c3'
assert m['parent_manifest']['identity']['completed_step']==100 and m['parent_manifest']['identity']['job_id']=='3154' and m['parent_manifest']['sha256']=='c20922f74f7dc77e7a233fcdcf329f2914bfaff335ed34d02209e1e6d09a3e86'
for file in ('gpu-kl-proof.py','native_fullstate_probe.py'):assert f[tool+'/'+file]['sha256']==m['files'][file]
tensor=json.loads((p/'TENSOR_STATES_CPU_RAW.json').read_text());assert len(tensor['rows'])==8
kl=[];native=[]
for rank in range(8):
 r=json.loads(f[run+f'/gpu-kl-proof/rank-{rank}.json']['text']);assert r['status']=='PASS' and r['rank']==rank and r['torch'].startswith('2.4.0')
 assert r['core_algos_file']==tool+'/source/repo/verl/trainer/ppo/core_algos.py';assert {x['dtype'] for x in r['records']}=={'torch.float32','torch.bfloat16','torch.float16'}
 for x in r['records']:
  assert x['points']==407 and x['native_dtype_values_and_gradients_exact'] is True and x['finite_backward'] is True and x['nonfinite_inputs_refused'] is True
  assert math.isfinite(x['max_abs_gradient_delta']) and math.isfinite(x['max_abs_value_delta'])
  assert x['fp64_stable_gradient_points']+len(x['fp64_boundary_points'])==407
  assert all(abs(z['unclamped_fp64']-10)<=.5 for z in x['fp64_boundary_points'])
  boundarymax=max(abs(z['actual_dtype_gradient']-z['fp64_gradient']) for z in x['fp64_boundary_points'])
  # Overall max includes excluded clamp-boundary dtype effects. The frozen
  # GPU proof asserts relative+absolute tolerances only on stable gradients.
  assert x['max_abs_gradient_delta']<=max(boundarymax,x['tolerance'])
  kl.append({'rank':rank,**{k:x[k] for k in ['dtype','points','max_abs_value_delta','max_abs_gradient_delta','fp64_stable_gradient_points','native_dtype_points_compared']},'boundary_max_grad_difference':boundarymax})
 r=json.loads(f[run+f'/native-fullstate/rank-{rank:02d}.json']['text']);d=json.loads(f[run+f'/native-fullstate/rank-{rank:02d}-diagnostic.json']['text']);assert r['identity']==dict(rank=rank,world_size=8,job='3234',code=m['code_commit'])
 assert r['status']=='PASS' and r['cold_new_wrapper_verified'] is True and r['same_world_sixth_outer_replay_equal'] is True and r['exact_same_world_reload_verified'] is True
 assert r['optimizer_steps_at5']==120 and r['optimizer_steps_at6']==130 and r['recovery_optimizer_calls']==10
 assert d['differences']==[] and d['uninterrupted_digest']==d['replayed_digest']
 native.append(dict(rank=rank,saved_logical_digest=r['logical_state_sha256'],sixth_digest=d['replayed_digest'],optimizer_steps=[120,130],complete_exact_replay=True,storage_alias_flags=[v['same_storage'] for v in d['restored_storage'].values()]))
samples=[json.loads(x) for x in f[run+'/resource-samples.jsonl']['text'].splitlines()]
for x in samples:
 assert x['job_id']=='3234' and len(x['gpus'])==8 and not x.get('event')
 for g in x['gpus']:assert g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15
fields=dict(re.findall(r'(\S+?)=(\S+)',b['scontrol_stdout']));assert fields['JobId']=='3234' and fields['JobState']=='RUNNING' and fields['Restarts']=='0' and fields['NodeList']==fields['BatchHost']=='h100-9'
r=dict(status='PASS_NATIVE8_COLD_TINY_FULLSTATE_AND_KL_PREFIX_ONLY',job_id='3234',code_commit=m['code_commit'],manifest_sha256=f[tool+'/manifest.json']['sha256'],parent_manifest_sha256=m['parent_manifest']['sha256'],actual_native8_cold_replay_accepted=True,actual_gpu_kl_records=24,actual_gpu_kl=kl,native_ranks=native,raw_snapshot_epoch=b['snapshot_epoch'],guard_complete_samples=len(samples),guard_stop_events=0,guard_last_sample_age_seconds=b['snapshot_epoch']-samples[-1]['epoch'],slurm_snapshot_state='RUNNING',requires_independent_recovery105_before106=True,actual_recovery105_accepted=False,actual_recovery300_accepted=False,limits='Tiny actual8-GPU BF16 FSDP/offload with fresh wrapper+Adam,120->130 exact replay; not independent real3B GPU optimizer update replay or effect acceptance',proof_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [p/'ACTUAL_RAW_BUNDLE.json',p/'TENSOR_STATES_CPU_RAW.json',p/'APPROVAL_HELPER_CPU_PROOF.json']})
(p/'NATIVE3234_RECEIPT.json').write_text(json.dumps(r,indent=2));print('NATIVE3234 RECEIPT',hashlib.sha256((p/'NATIVE3234_RECEIPT.json').read_bytes()).hexdigest())
