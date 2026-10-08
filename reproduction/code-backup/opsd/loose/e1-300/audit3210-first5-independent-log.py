from pathlib import Path
import json,hashlib,copy,importlib.util,math
W=Path(__file__).resolve().parent;S=W/'actual3210-independent-first5-20261004T171004Z';R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210/';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
G=W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py';assert H(G)=='74c3c13929c4f5cb784b90b7fd568ee80dfe1d37c8efaa28f97aae60b3492632';sp=importlib.util.spec_from_file_location('actual3210_gate',G);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
mp=S/'full_state/global_step_5/manifest.json';assert H(mp)=='44b94a6605568ebf8fa745a41a7f1df3444deb1fec187c2a7c2cbb619643dc7b';m=g.load_json(mp);summary=json.loads((S/'collection-summary.json').read_text());assert summary['source_head']==m['source_commit']=='a0968c2d24beeaa0400f60e4e34677fe6deef9c9' and not summary['source_status'];view=copy.deepcopy(m);pins={str(mp):H(mp)}
for name,rec in m['files'].items():
 if name.endswith('.json'):
  p=S/rec['path'].removeprefix(R);assert H(p)==rec['sha256'];pins[str(p)]=H(p);view['files'][name]['path']=str(p.resolve())
# Only JSON report paths are remapped to lossless local snapshots. PT paths and
# report body paths remain actual native paths, so native report->PT pin join is
# unchanged. Actual PT full content inspection is pending separate native tool.
workers=g.manifest_worker_pids(view,S.resolve());ranks=[]
for i in range(8):
 rec=m['files'][f'actor/training_state/rank_{i}.json'];row=g.load_json(S/rec['path'].removeprefix(R));assert row['model_dtensor_entries']==435 and row['scheduler_last_epoch']==5 and row['optimizer_state_entries']>0
 t=row['teacher'];assert t['execution_device']=='cuda' and t['storage_device']=='cpu' and t['parameter_count']==3085938688 and t['counters']['cuda_update_entries']==5
 ranks.append({'rank':i,'pid':row['pid'],'status':row['status'],'state_sha256':row['state_sha256'],'object_ids':row['object_ids'],'teacher_content_digest':t['model_content_digest'],'teacher_cuda_entries':5})
log=S/'train.log';pins[str(log)]=H(log);rows=[];pause=[];released=[];evaluations=[]
for line in log.read_text().splitlines():
 for marker,dest in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_PAUSED_JSON=',pause),('OPSD_E1_STEP5_RELEASED_JSON=',released),('BASELINE_EVALUATION_METRICS_JSON=',evaluations)]:
  if marker in line:dest.append(g.parse_log_marker(line,marker,driver_pid=m['driver_pid'],job_id='3210',worker_pids=workers,checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'))
assert len(rows)==5 and [r['global_step'] for r in rows]==list(range(1,6)) and len(pause)==1 and not released
compact=[]
for i,row in enumerate(rows,1):
 assert row['actual_outer_updates']==i and row['job_id']=='3210' and row['driver_pid']==m['driver_pid'] and row['config_sha256']==m['config_sha256'] and row['source_commit']==m['source_commit'];metrics=row['metrics'];g.validate_outer(metrics)
 assert math.isclose(metrics['actor/lr'],1e-6*i/57,rel_tol=1e-12)
 for rank in range(8):assert metrics[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}']==i
 compact.append({'step':i,'optimizer_steps_per_rank':[metrics[f'actor/optimizer_steps/rank_{rank}'] for rank in range(8)],'lr':metrics['actor/lr'],'opsd_loss':metrics['actor/opsd_loss'],'finish_ratio':metrics['env/finish_ratio'],'update_seconds':metrics['timing_s/update_actor']})
assert pause[0]['manifest_sha256']==H(mp) and pause[0]['pause_nonce']==m['pause_nonce'] and pause[0]['manifest_path']==R+'full_state/global_step_5/manifest.json'
stage={k.removeprefix('opsd_stage/'):v for k,v in rows[-1]['metrics'].items() if k.startswith('opsd_stage/')};g.validate_evaluation(stage,64);assert len(evaluations)==2;g.validate_evaluation(evaluations[0],512);g.validate_evaluation(evaluations[1],64)
for k,v in evaluations[1].items():assert stage[k]==v
cp=W/'actual3210-cpu-runner/actual-small-20261004T171119Z/038-resolved-config.json';assert H(cp)==m['files']['resolved_config']['sha256'];profile=g.load_json(cp);assert hashlib.sha256(g.canonical(profile)).hexdigest()==m['config_sha256'];pins[str(cp)]=H(cp);
a=profile['actor_rollout_ref']['actor'];d=a['distillation'];assert (a['ppo_mini_batch_size'],a['ppo_micro_batch_size'],a['ppo_epochs'],a['entropy_coeff'],a['use_kl_loss'],a['optim']['lr_warmup_steps'])==(128,16,1,0,False,57);assert (d['teacher_device'],d['teacher_prompt_mode'],d['beta'],d['temperature'],d['component_clip'])==('cuda','user_context',0,1,None)
out={'scope':'ACTUAL3210_FIRST5_LOG_CONFIG_PINNED_REPORT_AUDIT_ONLY_NATIVE_HF_AND_RESOURCE_ACCEPTANCE_PENDING','checkpoint_manifest_sha256':H(mp),'source_commit':m['source_commit'],'job_id':'3210','driver_pid':m['driver_pid'],'config_sha256':m['config_sha256'],'actual_worker_pids':sorted(workers),'ranks':ranks,'outer_rows':compact,'paused':pause[0],'evaluations':evaluations,'pinned_local_files':pins,'actual_canonical_resolved_config_verified':True,'no_signature_or_release':True};(W/'ACTUAL3210_FIRST5_LOG_REVIEW.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'scope':out['scope'],'actual_worker_pids':sorted(workers),'complete_outers':len(rows),'stage64_console_json_equal':True,'source_and_log_config_identity_match':True}))
