import copy,importlib.util,pathlib,pytest
P=pathlib.Path(__file__).with_name('training_identity.py');spec=importlib.util.spec_from_file_location('standard100_contract',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def original_outer(model):
 # Execute the actual unchanged producer record_outer body; publication itself
 # uses a temporary CPU path. This is not a claim of real actor updates.
 import ast,json,math,time,tempfile
 source=pathlib.Path(__file__).with_name('original_r1_observation.py')
 tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='record_outer')
 namespace={'json':json,'pathlib':pathlib,'time':time,'math':math,'atomic':lambda p,v:p.write_text(json.dumps(v))}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
 with tempfile.TemporaryDirectory() as tmp:
  output=pathlib.Path(tmp)/'actual-outer.json'
  for step in range(1,101):namespace['record_outer'](output,step,{'actor/grad_norm':0.,'actor/lr':1e-6},str(pathlib.Path(model).parents[1]))
  return json.loads(output.read_text())
def records():
 model='/dataset/shaoshitong/search-r1-rl/topd/logs/standard100/checkpoints/actor/global_step_100'
 u=dict(m.UPDATES,scope='ACTUAL_SEARCH_R1_RESOURCE_UPDATES_UNREVIEWED_NOT_100_TRAINING');ref={'path':'actual-outer.json','sha256':'a'*64}
 review=dict(status='INDEPENDENTLY_REVIEWED_ACTUAL_100_UPDATES',algorithm='SEARCH_R1_GRPO',actual_outer_updates=100,source_reference=m.SOURCE,source_manifest_sha256=m.SOURCE_MANIFEST,actual_update_evidence=u,actual_outer_evidence=ref)
 result=dict(status='COMPLETE_UNREVIEWED',scope='ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD',cleanup_passed=True,error=None,export={'directory':model},actual_update_evidence=u,actual_outer_evidence=ref)
 outer=original_outer(model)
 opts={'algorithm.adv_estimator':'grpo','data.train_batch_size':'512','actor_rollout_ref.actor.ppo_mini_batch_size':'256','actor_rollout_ref.actor.ppo_micro_batch_size':'64','actor_rollout_ref.actor.ppo_epochs':'1','actor_rollout_ref.rollout.n_agent':'5','actor_rollout_ref.rollout.tensor_model_parallel_size':'8','trainer.total_training_steps':'101','trainer.save_freq':'100'}
 config=dict(mode='train100',expected_outer_steps=100,final_checkpoint=model,source_manifest={'sha256':m.SOURCE_MANIFEST},training_command=['env/python','-m','verl.trainer.main_ppo']+[k+'='+v for k,v in opts.items()]);return [review,result,outer,config,model]
def test_real_standard_schema_not_legacy():
 assert m.check_standard100(*records())['teacher_kind']=='SEARCH_R1_GRPO100_32B'
@pytest.mark.parametrize('case',['g1','profile2','missing_outer','bad100_counts','bool_count','cleanup','failed','source','wrong_epoch','duplicate_override','unreviewed_counts','wrong_model','legacy_export_path'])
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
 if case=='legacy_export_path':outer['export_path']=outer.pop('checkpoint_directory')
 if case=='wrong_model':outer['checkpoint_directory']=model+'/other'
 with pytest.raises(RuntimeError):m.check_standard100(*a)

def test_original_observer_snapshot_exact():
 import hashlib
 assert hashlib.sha256(pathlib.Path(__file__).with_name('original_r1_observation.py').read_bytes()).hexdigest()=='3a605c3b244c6d8de9c76b25764a120125beee9e83369ef392ea3abfa6fd2099'
