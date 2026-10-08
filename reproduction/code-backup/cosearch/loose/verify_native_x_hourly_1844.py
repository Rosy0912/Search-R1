import json,math,pathlib,os
base=pathlib.Path('outputs/evidence');p=base/'native-x-hourly-1844-attempt.json';x=json.loads(p.read_text());s=json.loads((base/'native-x-saved13-18-hourly-1844.json').read_text())
assert x['run']=='author-native-continue-300-x' and str(x['job'])=='3211' and x['queue_query']['returncode']==0 and 'RUNNING' in x['queue']
assert not any('exit' in k or 'failure' in k for k in x['files'])
assert len(x['resource_snapshot']['live_training_pids'])==8 and all(v['current_start_ticks_match'] and v['current_cgroup_match'] for v in x['resource_snapshot']['live_training_pids'])
assert len(x['loaded_states'])==8 and all(v['full_model_optimizer_scheduler_equal'] for v in x['loaded_states'])
proof=[]
for step in [13,14,15,16,17,18]:
 o=next(v for v in x['outers'] if v['step']==step);assert all(math.isfinite(v) for v in o['metrics'].values())
 roles=['main'] if step in [15,16,18] else ['main','reranker'];ups=[v for v in x['worker_updates'] if v['outer_step']==step]
 assert {(v['role'],v['rank']) for v in ups}=={(r,i) for r in roles for i in range(4)}
 for v in ups:
  assert v['actual_adam_delta']==(4 if v['role']=='main' else 1) and v['lr_before']>0 and all(math.isfinite(g) for g in v['original_gradient_norms'])
 for role in roles:assert any(v['kind']=='global_update_completed' and v.get('step')==step and v.get('role')==role for v in x['controller_events'])
 if step in [15,16,18]:assert any(v['kind']=='ranker_update_skipped_no_rows' and v.get('step')==step and v['Adam_steps']==v['scheduler_steps']==0 for v in x['controller_events'])
 save=next(v for v in s['steps'] if v['step']==step);assert save['saved_records']==8
 start=next(v for v in x['controller_events'] if v['kind']=='generation_start' and v['step']==step)
 proof.append(dict(step=step,joint=step not in [15,16,18],actual_worker_updates=len(ups),full_saved_records=8,gen_to_outer_seconds=(o['monotonic_ns']-start['monotonic_ns'])/1e9,original_step_seconds=o['metrics']['timing_s/step'],save_seconds=o['metrics']['timing_s/save_checkpoint'],qualified_ranker_learning=False))
for current in [15,16,18]:
 a=next(v for v in s['steps'] if v['step']==current-1);b=next(v for v in s['steps'] if v['step']==current)
 ra={(v['role'],v['rank']):v for v in a['full_states']};rb={(v['role'],v['rank']):v for v in b['full_states']};assert all(ra['reranker',i]['state']==rb['reranker',i]['state'] for i in range(4))
result=dict(run=x['run'],job=3211,checked_at=x['checked_at'],steps=proof,completed_joint_steps=[13,14,17],main_only_steps=[15,16,18],skipped_ranker_full_model_optimizer_scheduler_equal_to_previous=True,retained_checkpoint_steps=[17,18],retained24_file_sizes_and_all8_sidecar_states_verified=True,independent_full_file_rehash=False,new_pureload_verified=False,operational_joint_five_completed_count=3,strict_accepted=False)
(base/'native-x-iterations13-18-verified.json').write_text(json.dumps(result,indent=2)+'\n')
t=base/'native-continue-300-x-live-latest.json';tmp=t.with_suffix('.tmp');tmp.write_text(p.read_text());os.replace(tmp,t)
for f in ['outputs/STATE.json','outputs/source-identity.json']:
 q=pathlib.Path(f);d=json.loads(q.read_text());d['last_verified_snapshot_epoch']=x['checked_at'];d['active_run_status']='X19_ranker4_update_completed_main_pending';v=d['active_continuation'];v.update(status=d['active_run_status'],last_verified_epoch=x['checked_at'],current_step=19,completed_outer_steps=[13,14,15,16,17,18],completed_joint_steps=[13,14,17],main_only_steps=[15,16,18],operational_five_completed_count=3,ranker_worker_updates_current=4,ranker_update_skipped_no_rows=False,disk_free_bytes=x['resource_snapshot']['disk_free_bytes']);d['operational_new_five']['completed_joint_iterations']=[13,14,17];d['latest_checkpoint_inventory']=result;q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

print(json.dumps(result))
