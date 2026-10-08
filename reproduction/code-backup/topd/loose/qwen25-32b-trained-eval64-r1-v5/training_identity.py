"""Pure record contract for the standard Search-R1 actual100 export; no model load."""
SOURCE='0d8a9214dc6e19f10e21549b56f6c01b45d67100'
SOURCE_MANIFEST='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
KIND='SEARCH_R1_GRPO100_32B'
UPDATES={'actual_outer_steps':100,'optimizer_steps_per_rank':1000,'micro_forwards_per_rank':4000,'ranks':8}

def need(ok,message):
 if not ok:raise RuntimeError(message)

def check_standard100(review,result,outer,config,model_directory):
 need(review.get('status')=='INDEPENDENTLY_REVIEWED_ACTUAL_100_UPDATES' and review.get('algorithm')=='SEARCH_R1_GRPO' and review.get('actual_outer_updates')==100 and review.get('source_reference')==SOURCE,'Reviewed standard Search-R1 actual100 identity required')
 need(result.get('status')=='COMPLETE_UNREVIEWED' and result.get('scope')=='ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD' and result.get('cleanup_passed') is True and result.get('error') is None,'Successful actual standard100 runtime required')
 need(outer.get('actual_completed_outer_steps')==list(range(1,101)) and outer.get('export_step')==100 and outer.get('checkpoint_directory')==model_directory,'Actual100 sequence/export path required')
 actual=result.get('actual_update_evidence',{})
 need(all(type(actual.get(k)) is int and actual[k]==v for k,v in UPDATES.items()),'Standard100 eight-rank1000opt4000micro required')
 need(review.get('actual_update_evidence')==actual and review.get('actual_outer_evidence')==result.get('actual_outer_evidence'),'Independent review must bind same actual count receipts')
 need(result.get('export',{}).get('directory')==model_directory,'Actual trainer export directory differs')
 need(config.get('mode')=='train100' and config.get('expected_outer_steps')==100 and config.get('final_checkpoint')==model_directory,'Profile/resume/calibration recipe cannot be trained100')
 need(config.get('source_manifest',{}).get('sha256')==SOURCE_MANIFEST and review.get('source_manifest_sha256')==SOURCE_MANIFEST,'Original standard source plus reviewed engineering-patch manifest required')
 args=config.get('training_command',[])
 need(len(args)>3 and args[1:3]==['-m','verl.trainer.main_ppo'],'Original standard main_ppo entry required')
 options={}
 for arg in args[3:]:
  name,sep,value=arg.partition('=');name=name.lstrip('+')
  need(sep and name not in options,'Duplicate/malformed actual training override')
  options[name]=value
 expected={'algorithm.adv_estimator':'grpo','data.train_batch_size':'512','actor_rollout_ref.actor.ppo_mini_batch_size':'256','actor_rollout_ref.actor.ppo_micro_batch_size':'64','actor_rollout_ref.actor.ppo_epochs':'1','actor_rollout_ref.rollout.n_agent':'5','actor_rollout_ref.rollout.tensor_model_parallel_size':'8','trainer.total_training_steps':'101','trainer.save_freq':'100'}
 need(all(options.get(k)==v for k,v in expected.items()),'Actual standard100 recipe geometry differs')
 return {'scope':'BOUND_REVIEWED_STANDARD100_IDENTITY_NOT_QUALITY_OR_TOPD','teacher_kind':KIND,'source_reference':SOURCE,'actual_update_evidence':actual}
