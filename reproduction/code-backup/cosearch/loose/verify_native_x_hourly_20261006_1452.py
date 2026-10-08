import json,math,pathlib,datetime
b=pathlib.Path('outputs/evidence');tag='20261006-1452'
x=json.load(open(b/f'native-x-hourly-{tag}-minimal.json'));sr=json.load(open(b/f'native-x-saved137-current-hourly-{tag}-stream.json'));rpc=json.load(open(b/f'native-x-rpc-hourly-{tag}-stream.json'));e=rpc['events'];u=json.load(open(b/f'native-x-updates-hourly-{tag}-stream.json'))['updates'];old=json.load(open(b/'native-continue-300-x-live-latest.json'));prior=json.load(open(b/'native-x-iterations135-137-verified.json'))
assert x['queue_rc']==0 and '3211 RUNNING' in x['queue'] and not x['files']
assert len(x['live'])==8 and all(v['ticks_match'] and v['cgroup_match'] for v in x['live'])
assert x['gpus']['rc']==0 and set(l.split(',')[0] for l in x['gpus']['stdout'].splitlines())==set(v['uuid'] for v in old['acks'])
proof=[];joint=prior['completed_joint_steps'].copy();mainonly=prior['main_only_steps'].copy();saved={v['step']:v for v in sr['steps']}
steps=sorted(v['step'] for v in e if v['kind']=='outer' and v['step']>=138)
assert steps and steps==list(range(138,max(steps)+1))
for step in steps:
 o=next(v for v in e if v['kind']=='outer' and v['step']==step);assert all(math.isfinite(v) for v in o['metrics'].values())
 updates=[v for v in u if v['outer_step']==step];skip=[v for v in e if v['kind']=='ranker_update_skipped_no_rows' and v['step']==step];roles=['main'] if skip else ['main','reranker']
 assert {(v['role'],v['rank']) for v in updates}=={(r,i) for r in roles for i in range(4)}
 for v in updates:assert v['actual_adam_delta']==(4 if v['role']=='main' else 1) and v['lr_before']>0 and all(math.isfinite(g) for g in v['original_gradient_norms'])
 for role in roles:assert any(v['kind']=='global_update_completed' and v.get('step')==step and v.get('role')==role for v in e)
 if skip:
  assert all(v['Adam_steps']==v['scheduler_steps']==0 for v in skip)
  aa={(v['role'],v['rank']):v['state'] for v in saved[step-1]['full_states']};cc={(v['role'],v['rank']):v['state'] for v in saved[step]['full_states']};assert all(aa['reranker',i]==cc['reranker',i] for i in range(4));mainonly.append(step)
 else:joint.append(step)
 assert saved[step]['saved_records']==8
 start=next(v for v in e if v['kind']=='generation_start' and v['step']==step and not v.get('validation',False))
 proof.append(dict(step=step,joint=not bool(skip),full_saved_records=8,gen_to_outer_seconds=(o['monotonic_ns']-start['monotonic_ns'])/1e9,original_step_seconds=o['metrics']['timing_s/step'],save_seconds=o['metrics']['timing_s/save_checkpoint']))
retained=[v['step'] for v in sr['steps'] if v['current_checkpoint_exists']];assert len(retained)==2 and all(len(saved[s]['files'])==8 for s in retained)
result=dict(run=x['run'],job=3211,checked_at=x['checked_at'],steps=proof,completed_joint_steps=joint,main_only_steps=mainonly,retained_checkpoint_steps=retained,all8_sidecar_states_and24file_sizes_verified=True,ranker_skipped_full_model_optimizer_scheduler_unchanged=True,full_file_rehash=False,new_pureload=False,strict_accepted=False,latest_minimal_snapshot=f'native-x-hourly-{tag}-minimal.json',last_complete_detailed_snapshot_preserved=True)
last=max(steps);current=max(v['step'] for v in e if v['kind']=='generation_start' and not v.get('validation',False));assert current==last+1
cu=[v for v in u if v['outer_step']==current];rs=[v for v in cu if v['role']=='reranker'];skip=any(v['kind']=='ranker_update_skipped_no_rows' and v['step']==current for v in e);genend=any(v['kind']=='generation_end' and v['step']==current and not v.get('validation',False) for v in e)
if rs:
 assert {v['rank'] for v in rs}==set(range(4)) and all(v['actual_adam_delta']==1 and v['lr_before']>0 and all(math.isfinite(g) for g in v['original_gradient_norms']) for v in rs)
 stage='ranker_updated_main_pending'
elif skip:stage='ranker_skipped_main_pending'
else:stage='generation_returned' if genend else 'generation_started'
ms=[v for v in cu if v['role']=='main']
if ms:
 assert {v['rank'] for v in ms}==set(range(4)) and all(v['actual_adam_delta']==4 and v['lr_before']>0 and all(math.isfinite(g) for g in v['original_gradient_norms']) for v in ms)
 assert any(v['kind']=='global_update_completed' and v['step']==current and v['role']=='main' for v in e)
 stage='main_updated_save_outer_pending'
result['current_step']=current;result['current_stage']=stage;result['current_controller_stream_epoch']=rpc['checked_at']
(b/f'native-x-iterations{min(steps)}-{last}-verified.json').write_text(json.dumps(result,indent=2)+'\n')
for f in ['outputs/STATE.json','outputs/source-identity.json']:
 p=pathlib.Path(f);d=json.loads(p.read_text());status=f'X{current}_{stage}_no_outer{current}';d.update(status=status,current_status=status,active_run_status=status,last_verified_snapshot_epoch=x['checked_at'],last_supervision=datetime.datetime.fromtimestamp(x['checked_at'],datetime.timezone.utc).isoformat(),latest_current_minimal_snapshot=result['latest_minimal_snapshot'],latest_checkpoint_inventory=result)
 d['active_continuation'].update(status=status,current_step=current,completed_outer_steps=list(range(13,last+1)),completed_joint_steps=joint,main_only_steps=mainonly,operational_five_completed_count=len(joint),ranker_worker_updates_current=len(rs),last_verified_epoch=x['checked_at'],disk_free_bytes=x['disk_free_bytes'],last_completed_global_step=last,last_completed_joint_step=max(joint),current_iteration_verified_stage=stage,current_qualification=None,qualified_groups_current=None,ranker_update_skipped_no_rows=skip,latest_fully_verified_saved_record_step=last,current_controller_stream_epoch=rpc['checked_at'])
 d['operational_new_five']['completed_joint_iterations']=joint;d['ranker_last_verified_update_global_step']=current if rs else max(joint);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result))
