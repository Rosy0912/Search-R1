import json,hashlib
from pathlib import Path
p=Path('outputs/evidence/native-v-resume-current-attempt.json');x=json.loads(p.read_text())
assert x['run']=='author-native-continue-300-v' and x['job']=='3205' and x['queue_query']['returncode']==0 and 'RUNNING' in x['queue']
assert 'driver-exit.json' not in x['files'] and 'startup-failure.json' not in x['files']
acks=x['acks'];assert len(acks)==8 and {(a['role'],a['rank']) for a in acks}=={(r,k) for r in ('main','reranker') for k in range(4)}
assert len({a['uuid'] for a in acks})==8 and all(a['adam_state_initial']==0 for a in acks)
leased=x['files']['launch.json']['metadata']['leased_uuids'];assert set(leased)=={a['uuid'] for a in acks}
live=x['resource_snapshot']['live_training_pids'];assert len(live)==8 and all(a['current_start_ticks_match'] and a['current_cgroup_match'] for a in live)
loads=x['loaded_states'];assert len(loads)==8 and {(a['role'],a['rank']) for a in loads}=={(r,k) for r in ('main','reranker') for k in range(4)}
assert all(a['full_model_optimizer_scheduler_equal'] and a['source_job']=='3201' and '/author-native-continue-300-u/checkpoints/global_step_12/' in a['path'] for a in loads)
events=x['controller_events'];resume=[e for e in events if e['kind']=='continuation_resume_verified'];assert len(resume)==1
v=resume[0];assert v['source_run']=='author-native-continue-300-u' and v['source_step']==12 and v['next_training_step']==13 and v['target_total_step']==300 and v['dataloader_state_equal'] and v['original_load_called_once'] and v['full_state_load_ranks']==8 and v['natural_scheduler_horizon_unchanged']
assert v['dataloader_sha256']=='e900aca11f4f1b9b36c9c82ae91b517181bca38952713caf49ea5718f19f9341'
assert any(e['kind']=='generation_start' and e['step']==13 and e['input_rows']==4096 and not e['validation'] for e in events)
assert any(e['kind']=='all_rank_ack_gate' and e['passed'] for e in events)
Path('outputs/evidence/native-continue-300-v-live-latest.json').write_bytes(p.read_bytes())
proof=dict(run=x['run'],job=x['job'],checked_at=x['checked_at'],source=v['source_run'],source_step=12,new8_liveACK=True,new8_source_fullstate_equal=True,data_state_equal=True,original_load_called_once=True,first_training_step=13,real4096_generation_started=True,completed_new_joint_iterations=0,operational_five_passed=False,strict_accepted=False,live_snapshot='native-continue-300-v-live-latest.json',snapshot_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path('outputs/evidence/native-v-resume-start-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
