import copy,importlib.util,pathlib,pytest
P=pathlib.Path(__file__).with_name('training_identity.py');spec=importlib.util.spec_from_file_location('standard100_contract',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def records():
 model='/dataset/shaoshitong/search-r1-rl/topd/logs/standard100/checkpoints/actor/global_step_100'
 u=dict(m.UPDATES,scope='ACTUAL_SEARCH_R1_RESOURCE_UPDATES_UNREVIEWED_NOT_100_TRAINING');ref={'path':'actual-outer.json','sha256':'a'*64}
 review=dict(status='INDEPENDENTLY_REVIEWED_ACTUAL_100_UPDATES',algorithm='SEARCH_R1_GRPO',actual_outer_updates=100,source_reference=m.SOURCE,source_manifest_sha256=m.SOURCE_MANIFEST,actual_update_evidence=u,actual_outer_evidence=ref)
 result=dict(status='COMPLETE_UNREVIEWED',scope='ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD',cleanup_passed=True,error=None,actual_update_evidence=u,actual_outer_evidence=ref)
 outer=dict(actual_completed_outer_steps=list(range(1,101)),export_step=100,export_path=model)
 opts={'algorithm.adv_estimator':'grpo','data.train_batch_size':'512','actor_rollout_ref.actor.ppo_mini_batch_size':'256','actor_rollout_ref.actor.ppo_micro_batch_size':'64','actor_rollout_ref.actor.ppo_epochs':'1','actor_rollout_ref.rollout.n_agent':'5','actor_rollout_ref.rollout.tensor_model_parallel_size':'8','trainer.total_training_steps':'101','trainer.save_freq':'100'}
 config=dict(mode='train100',expected_outer_steps=100,final_checkpoint=model,source_manifest={'sha256':m.SOURCE_MANIFEST},training_command=['env/python','-m','verl.trainer.main_ppo']+[k+'='+v for k,v in opts.items()]);return [review,result,outer,config,model]
def test_real_standard_schema_not_legacy():
 assert m.check_standard100(*records())['teacher_kind']=='SEARCH_R1_GRPO100_32B'
@pytest.mark.parametrize('case',['g1','profile2','missing_outer','bad100_counts','bool_count','cleanup','failed','source','wrong_epoch','duplicate_override','unreviewed_counts','wrong_model'])
def test_failed_or_relabelled_records_rejected(case):
 a=copy.deepcopy(records());review,result,outer,config,model=a
 if case=='g1':review['algorithm']='SEARCH_G1_GRPO'
 if case=='profile2':config.update(mode='profile2',expected_outer_steps=2)
 if case=='missing_outer':outer['actual_completed_outer_steps'].remove(47)
 if case=='bad100_counts':result['actual_update_evidence'].update(optimizer_steps_per_rank=300,micro_forwards_per_rank=1000)
 if case=='bool_count':result['actual_update_evidence']['ranks']=True
 if case=='cleanup':result['cleanup_passed']=False
 if case=='failed':result['status']='FAILED'
 if case=='source':review['source_manifest_sha256']='0'*64
 if case=='wrong_epoch':config['training_command']=[x.replace('ppo_epochs=1','ppo_epochs=2') for x in config['training_command']]
 if case=='duplicate_override':config['training_command']+=['trainer.total_training_steps=3']
 if case=='unreviewed_counts':review['actual_update_evidence']={'ranks':8}
 if case=='wrong_model':outer['export_path']=model+'/other'
 with pytest.raises(RuntimeError):m.check_standard100(*a)
