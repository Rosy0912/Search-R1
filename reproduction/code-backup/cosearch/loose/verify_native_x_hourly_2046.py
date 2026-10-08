import json,math,pathlib
b=pathlib.Path('outputs/evidence');x=json.load(open(b/'native-x-hourly-2046-minimal-v2.json'));s=json.load(open(b/'native-x-saved22-24-hourly-2046-stream.json'));e=json.load(open(b/'native-x-current-rpc-hourly-2046-stream.json'))['events'];u=json.load(open(b/'native-x-current-updates-hourly-2046-stream.json'))['updates'];old=json.load(open(b/'native-continue-300-x-live-latest.json'))
assert x['queue_rc']==0 and '3211 RUNNING' in x['queue'] and not x['files']
assert len(x['live'])==8 and all(v['ticks_match'] and v['cgroup_match'] for v in x['live'])
assert x['gpus']['rc']==0 and set(l.split(',')[0] for l in x['gpus']['stdout'].splitlines())==set(v['uuid'] for v in old['acks'])
proof=[]
for step in [22,23,24]:
 o=next(v for v in e if v['kind']=='outer' and v['step']==step);assert all(math.isfinite(v) for v in o['metrics'].values())
 roles=['main','reranker'] if step==22 else ['main'];updates=[v for v in u if v['outer_step']==step]
 assert {(v['role'],v['rank']) for v in updates}=={(r,i) for r in roles for i in range(4)}
 for v in updates:assert v['actual_adam_delta']==(4 if v['role']=='main' else 1) and v['lr_before']>0 and all(math.isfinite(g) for g in v['original_gradient_norms'])
 for role in roles:assert any(v['kind']=='global_update_completed' and v.get('step')==step and v.get('role')==role for v in e)
 if step!=22:assert any(v['kind']=='ranker_update_skipped_no_rows' and v['step']==step and v['Adam_steps']==v['scheduler_steps']==0 for v in e)
 assert next(v for v in s['steps'] if v['step']==step)['saved_records']==8
 start=next(v for v in e if v['kind']=='generation_start' and v['step']==step)
 proof.append(dict(step=step,joint=step==22,full_saved_records=8,gen_to_outer_seconds=(o['monotonic_ns']-start['monotonic_ns'])/1e9,original_step_seconds=o['metrics']['timing_s/step'],save_seconds=o['metrics']['timing_s/save_checkpoint']))
for a,c in zip(s['steps'],s['steps'][1:]):
 aa={(v['role'],v['rank']):v['state'] for v in a['full_states']};cc={(v['role'],v['rank']):v['state'] for v in c['full_states']};assert all(aa['reranker',i]==cc['reranker',i] for i in range(4))
assert all(v['current_checkpoint_exists'] and len(v['files'])==8 for v in s['steps'][1:])
result=dict(run=x['run'],job=3211,checked_at=x['checked_at'],steps=proof,completed_joint_steps=[13,14,17,19,20,22],main_only_steps=[15,16,18,21,23,24],retained_checkpoint_steps=[23,24],all8_sidecar_states_and24file_sizes_verified=True,ranker_skipped_full_model_optimizer_scheduler_unchanged=True,full_file_rehash=False,new_pureload=False,strict_accepted=False,latest_minimal_snapshot='native-x-hourly-2046-minimal-v2.json',last_complete_detailed_snapshot_preserved=True)
(b/'native-x-iterations22-24-verified.json').write_text(json.dumps(result,indent=2)+'\n')
for f in ['outputs/STATE.json','outputs/source-identity.json']:
 p=pathlib.Path(f);d=json.loads(p.read_text());d['last_verified_snapshot_epoch']=x['checked_at'];d['active_run_status']='X25_generation_returned_no_completed_outer25';d['active_continuation'].update(status=d['active_run_status'],current_step=25,completed_outer_steps=list(range(13,25)),completed_joint_steps=result['completed_joint_steps'],main_only_steps=result['main_only_steps'],operational_five_completed_count=6,ranker_worker_updates_current=0,last_verified_epoch=x['checked_at'],disk_free_bytes=x['disk_free_bytes']);d['operational_new_five']['completed_joint_iterations']=result['completed_joint_steps'];d['latest_checkpoint_inventory']=result;d['latest_current_minimal_snapshot']=result['latest_minimal_snapshot'];p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
note='Hourly X3211 epoch1791147182.7314842 RUNNING/noexit: finite completed22joint(all8Adam main+4/ranker+1/bothRPC/full8save),23/24main-only(explicitranker skip0Adam/scheduler/main4+4/full8save); each skippedranker fullmodel/RLAdam/scheduler equals previous saved state. Retained23,24 all8sidecar states/full24file sizes/data checked, no fullfile rehash/newpureload. Gen→outer22/23/24='+ '/'.join(f"{v['gen_to_outer_seconds']:.3f}" for v in proof)+'s including save='+ '/'.join(f"{v['save_seconds']:.3f}" for v in proof)+'s,600failed/strictfalse. Current25 generationreturned/noouter25 observed. All8livePIDticks/cgroups/leasedUUIDgood/disk6.483TB. Jointcount6, five-stability already notified/do not repeat. Three detailed read attempts timedout0B; same-route queue/bounded-tail current read and separate controller RPC/worker updates/saved streams recovered. Detailed lastgood snapshot preserved; latest minimal evidence native-x-hourly-2046-minimal-v2.json/native-x-iterations22-24-verified.json. No pause/resubmit/hotedit; continue300.'
p=pathlib.Path('REPORT.md');p.write_text(p.read_text()+'\n\n'+note+'\n')
for dst in ['outputs/REPORT.md','/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/cosearch/REPORT.md']:pathlib.Path(dst).write_text(p.read_text())
p=pathlib.Path('work/automation-cosearch-x-hourly-current.json');a=json.loads(p.read_text());a['prompt']='LATEST VERIFIED '+note+' FIRST same-route currentX poll; if detailed reads timeout, local read-only bounded-tail work/poll_native_x_minimal_current_command.txt and explicit controllerRPC/workerupdate/saved streaming reads recover current records. Bounded-tail alone can omit large records; do not infer missing update from tail. Preserve good detailed snapshots. Supersedes older current22/pending statements. Normalprogressquiet.\n\n'+a['prompt'];p.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result))
