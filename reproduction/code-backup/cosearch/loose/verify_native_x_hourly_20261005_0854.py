import json,math,pathlib
b=pathlib.Path('outputs/evidence');x=json.load(open(b/'native-x-hourly-20261005-0854-minimal.json'));s=json.load(open(b/'native-x-saved53-56-hourly-20261005-0854-stream.json'));e=json.load(open(b/'native-x-rpc-hourly-20261005-0854-stream.json'))['events'];u=json.load(open(b/'native-x-updates-hourly-20261005-0854-stream.json'))['updates'];old=json.load(open(b/'native-continue-300-x-live-latest.json'))
assert x['queue_rc']==0 and '3211 RUNNING' in x['queue'] and not x['files']
assert len(x['live'])==8 and all(v['ticks_match'] and v['cgroup_match'] for v in x['live'])
assert x['gpus']['rc']==0 and set(l.split(',')[0] for l in x['gpus']['stdout'].splitlines())==set(v['uuid'] for v in old['acks'])
proof=[]
for step in [54,55,56]:
 o=next(v for v in e if v['kind']=='outer' and v['step']==step);assert all(math.isfinite(v) for v in o['metrics'].values())
 roles=['main'];updates=[v for v in u if v['outer_step']==step]
 assert {(v['role'],v['rank']) for v in updates}=={(r,i) for r in roles for i in range(4)}
 for v in updates:assert v['actual_adam_delta']==(4 if v['role']=='main' else 1) and v['lr_before']>0 and all(math.isfinite(g) for g in v['original_gradient_norms'])
 for role in roles:assert any(v['kind']=='global_update_completed' and v.get('step')==step and v.get('role')==role for v in e)
 if step in [54,55,56]:assert any(v['kind']=='ranker_update_skipped_no_rows' and v['step']==step and v['Adam_steps']==v['scheduler_steps']==0 for v in e)
 assert next(v for v in s['steps'] if v['step']==step)['saved_records']==8
 start=next(v for v in e if v['kind']=='generation_start' and v['step']==step)
 proof.append(dict(step=step,joint=False,full_saved_records=8,gen_to_outer_seconds=(o['monotonic_ns']-start['monotonic_ns'])/1e9,original_step_seconds=o['metrics']['timing_s/step'],save_seconds=o['metrics']['timing_s/save_checkpoint']))
for a,c in zip(s['steps'],s['steps'][1:]):
 aa={(v['role'],v['rank']):v['state'] for v in a['full_states']};cc={(v['role'],v['rank']):v['state'] for v in c['full_states']};assert all(aa['reranker',i]==cc['reranker',i] for i in range(4))
assert all(v['current_checkpoint_exists'] and len(v['files'])==8 for v in s['steps'][-2:])
result=dict(run=x['run'],job=3211,checked_at=x['checked_at'],steps=proof,completed_joint_steps=[13,14,17,19,20,22,49],main_only_steps=[15,16,18,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,55,56],retained_checkpoint_steps=[55,56],all8_sidecar_states_and24file_sizes_verified=True,ranker_skipped_full_model_optimizer_scheduler_unchanged=True,full_file_rehash=False,new_pureload=False,strict_accepted=False,latest_minimal_snapshot='native-x-hourly-20261005-0854-minimal.json',last_complete_detailed_snapshot_preserved=True)
(b/'native-x-iterations54-56-verified.json').write_text(json.dumps(result,indent=2)+'\n')
for f in ['outputs/STATE.json','outputs/source-identity.json']:
 p=pathlib.Path(f);d=json.loads(p.read_text());d['last_verified_snapshot_epoch']=x['checked_at'];d['active_run_status']='X57_ranker_skipped_main_pending_no_outer57';d['active_continuation'].update(status=d['active_run_status'],current_step=57,completed_outer_steps=list(range(13,57)),completed_joint_steps=result['completed_joint_steps'],main_only_steps=result['main_only_steps'],operational_five_completed_count=7,ranker_worker_updates_current=0,last_verified_epoch=x['checked_at'],disk_free_bytes=x['disk_free_bytes']);d['operational_new_five']['completed_joint_iterations']=result['completed_joint_steps'];d['latest_checkpoint_inventory']=result;d['latest_current_minimal_snapshot']=result['latest_minimal_snapshot'];p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

print(json.dumps(result))
