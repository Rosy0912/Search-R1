"""Actual first5 acceptance and sixth update continuation; local originals only."""
import base64,hashlib,importlib.util,json,math
from pathlib import Path
W=Path(__file__).resolve().parent
S=W/'actual-snapshots/3261-20261005T210038278787Z';F=S/'original-files'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
peer=read(W/'ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json')
need(sha(W/'ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json')=='a9df63ada3da689069495e25eb9e8e5c993a0f6b1aad9648cfa19d822fd85a14' and peer['first5_implementation_state_acceptance'] is True,'Actual accepted first5 record')
need(sha(W/'ACTUAL3261_UNSIGNED_PREPARATION_INDEPENDENT_REVIEW.json')=='8ef6d7242084d02dbf1bab80e2fd83378088b1a559bc71e774ce5e0ec01dab89','Actual accepted unsigned record')
need(sha(W/'ACTUAL3261_FIRST5_PUBLICATION_INDEPENDENT_REVIEW.json')=='0f9ca7e33befbfa44deffbb7b0f27a2f56050256c4a4b114cc1f46521b166e4f','Actual accepted publication record')
source=peer['source_commit'];workers=tuple(x['pid'] for x in peer['actual_rank_results'])
gp=W.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
need(sha(gp)=='c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb','Frozen gate source')
sp=importlib.util.spec_from_file_location('independent_actual_outer6_gate',gp);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
original=read(S/'observed-remote-original.json');summary=read(S/'observation-summary.json')
need(sha(S/'observed-remote-original.json')==summary['transport_original_sha256'],'Actual snapshot original SHA')
streams={};outer_streams=[];release_streams=[]
for name in ['train.log','ray-stdout-pid-1533312.out']:
 p=F/name;meta=original['files'][name];raw=p.read_bytes()
 need(base64.b64decode(meta['base64'],validate=True)==raw and hashlib.sha256(raw).hexdigest()==meta['sha256'] and len(raw)==meta['bytes'] and meta['same_inode'] and meta['complete_original_at_start'] and meta['source_offset']==0 and meta['before']['uid']==meta['after']['uid']==0,'Actual exact owned log prefix '+name)
 rows=[];releases=[];pauses=[]
 for line in raw.decode().splitlines():
  for marker,dest in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_RELEASED_JSON=',releases),('OPSD_E1_STEP5_PAUSED_JSON=',pauses)]:
   if marker in line:dest.append(g.parse_log_marker(line,marker,driver_pid=1533312,job_id='3261',worker_pids=workers))
 need([r['actual_outer_updates'] for r in rows]==list(range(1,7)),'Actual completed1..6 '+name)
 need(len(pauses)==1 and pauses[0]['manifest_sha256']==peer['checkpoint_manifest_sha256'] and pauses[0]['pause_nonce']==peer['pause_nonce'],'Actual unique prior5pause')
 need(releases==[{'actual_outer_updates':5,'driver_pid':1533312,'next_global_step':6,'receipt_sha256':'812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b'}],'Actual unique released5-to6')
 for row in rows:
  need(row['actual_outer_updates']==row['global_step'] and row['source_commit']==source and row['config_sha256']=='73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60' and row['driver_pid']==1533312 and row['job_id']=='3261','Actual unchanged source/config/process')
  g.validate_outer(row['metrics'],profile='author200')
  m=row['metrics'];need(m['actor/lr']==5e-6 and m['rollout/trajectories']==512 and m['rollout/context_mismatches']==0,'Actual unchanged sample/LR/history contract')
  for rank in range(8):
   suffix='/rank_'+str(rank)
   need(m['actor/optimizer_steps'+suffix]==16 and m['actor/samples'+suffix]==64,'Actual per-rank optimizer/sample budget')
   need(m['teacher/cuda_update_entries_cumulative_per_rank'+suffix]==row['global_step'] and m['teacher/update_device_is_cuda_per_rank'+suffix]==1 and m['teacher/student_optimizer_membership_per_rank'+suffix]==0,'Actual pure fixed-teacher cumulative continuity')
 streams[name]={'sha256':sha(p),'bytes':len(raw),'unchanged_during_read':meta['unchanged_during_read']};outer_streams.append(rows);release_streams.append(releases)
need(outer_streams[0]==outer_streams[1] and release_streams[0]==release_streams[1],'Actual direct/aggregate exact complete object equality')
need(sha(F/'acceptance-5-receipt.json')=='812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b','Actual published receipt unchanged')
need(sha(F/'full_state/global_step_5/manifest.json')==peer['checkpoint_manifest_sha256'],'Actual accepted checkpoint manifest unchanged')
need((F/'train.log').read_bytes()[:405957]==(W/'actual3261-unsigned-collected-20261005T205139510221Z/immutable-first5-train-prefix.out').read_bytes(),'Actual accepted immutable first5 prefix still preserved')
proc=original['processes']['1533312'];need(proc['start_ticks']=='1829085316' and proc['in_observed_job'] and proc['uids']==[0]*4 and proc['state'] in ['R','S','D','I'],'Actual same driver lifetime')
for r in peer['actual_rank_results']:
 p=original['processes'][str(r['pid'])]
 need(p['start_ticks']==r['start_ticks'] and p['in_observed_job'] and p['uids']==[0]*4 and p['state'] in ['R','S','D','I'],'Actual original worker lifetime')
 apps=[a for a in original['unfiltered_application_csv_rows'] if int(a[1])==r['pid']]
 need(len(apps)==1 and apps[0][0].strip()==r['gpu_uuid'],'Actual original worker same physical H100')
service=original['processes']['1531763'];need(service['start_ticks']=='1829074494' and service['in_observed_job'] and service['uids']==[0]*4,'Actual original loopback service lifetime')
need({int(a[1]) for a in original['unfiltered_application_csv_rows']}==set(workers)|{1531763},'Actual current GPU applications all own job')
slurm=original['commands']['scontrol']['stdout'];need(all(x in slurm for x in ['JobId=3261','JobState=RUNNING','Restarts=0','NodeList=h100-10','TresPerNode=gres:gpu:h100:8']),'Actual unchanged same-job allocation without restart')
m=outer_streams[0][-1]['metrics']
out={'status':'PASS_ACTUAL_FIRST5_IMPLEMENTATION_STATE_ACCEPTANCE_AND_SAME_PROCESS_OUTER6','source_commit':source,'job_id':'3261','config_sha256':outer_streams[0][-1]['config_sha256'],'driver_pid':1533312,'driver_start_ticks':'1829085316','actual_completed_outer_iterations':6,'first5_independent_implementation_state_acceptance':True,'same_job_process_continuation_after_actual_five_gate':True,'receipt_sha256':sha(F/'acceptance-5-receipt.json'),'actual_release_marker':release_streams[0][0],'actual_original_streams':streams,'actual_snapshot_original_sha256':sha(S/'observed-remote-original.json'),'actual_eight_worker_lifetimes_unchanged':True,'actual_retriever_lifetime_unchanged':True,'actual_job_restarts':0,'actual_outer6':{'reward':m['critic/rewards/mean'],'finished_ratio':m['env/finish_ratio'],'legal_action_ratio':m['env/ratio_of_valid_action'],'clipped_OPSD_loss':m['actor/opsd_loss'],'logged_preclip_global_grad_norm':m['actor/grad_norm'],'lr':m['actor/lr'],'step_seconds':m['timing_s/step'],'actual_optimizer_calls_each_rank':16,'actual_teacher_cuda_update_entries_each_rank':6,'actual_sample_count_each_rank':64,'actual_trajectories':512,'finite_gate_pass':True},'actual_cumulative_optimizer_calls_from_accepted5_plus_logged6_each_rank':96,'accepted_first5_saved_data_consumed_cursor':2560,'expected_after6_data_cursor_from_executed_source':3072,'after6_data_cursor_raw_state_not_saved_or_claimed':True,'HF6_or_native6_saved_or_inspected':False,'new_process_restore_accepted':False,'final200_complete':False,'effect_improvement_proven':False,'negative_effect_review_sha256':sha(W/'ACTUAL3261_FIRST5_BEHAVIOR_INDEPENDENT_REVIEW.json'),'scope':'Exactly first5 actual implementation/state gate plus actual sixth outer after same-job/same-driver/same-worker release. Saved5 Adam/scheduler/RNG/teacher/data were checked; frozen executed same-process path and actual6 cumulative teacher/optimizer logs establish continuation. No separate raw native6 state or full trajectories are invented. Clipped negative loss is not true KL or task improvement. Later200/final51713 remain outstanding.','reviewer_GPU_signature_submission_source_mutation':False}
p=W/'ACTUAL3261_FIRST5_AND_OUTER6_FINAL_INDEPENDENT_REVIEW.json'
with p.open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'status':out['status'],'sha256':sha(p),'actual_completed_outer_iterations':6,'actual_outer6':out['actual_outer6']},sort_keys=True))
