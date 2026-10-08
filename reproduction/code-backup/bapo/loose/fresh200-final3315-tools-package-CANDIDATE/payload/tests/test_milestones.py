import copy,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
import milestones as m
from search_r1.baselines.full_state import state_digest
from search_r1.baselines.fresh200_contract import outer_contract
import test_final_candidate as fixture
class Milestones(unittest.TestCase):
 def setUp(self):self.t=tempfile.TemporaryDirectory();self.run=Path(self.t.name).resolve();self.attempts,_=fixture.Final().attempts()
 def tearDown(self):self.t.cleanup()
 def create(self,step):
  state=self.run/'checkpoints/fresh200_state'/('outer_%d_COMPLETE'%step);actor=self.run/'checkpoints/actor'/('global_step_%d'%step);alg=self.run/'checkpoints/algorithm'/('global_step_%d'%step)
  for p in (state/'ranks',actor,alg):p.mkdir(parents=True)
  for name in ('config.json','tokenizer.json','tokenizer_config.json'): (actor/name).write_text('{}')
  (actor/'model-1.safetensors').write_bytes(b'fixture-not-native');(alg/'state.json').write_text('{}');(state/'driver.pt').write_bytes(b'driver-fixture')
  a=m.checkpoint_prefixes(self.attempts)[step];status=dict(status='COMPLETE',actual_outer=step,completed_outer_updates=step,attempt_id=a['attempt_id'],progress=a['progress'])
  algorithm=dict(method='bapo',algorithm_source_commit='source',global_step=step,actual_outer_updates=step,bapo={'early':False},resume_supported=False,algorithm_json_resume_supported=False)
  driver=dict(protocol='BAPO_FRESH200_V1',status='COMPLETE',source_commit='source',global_step=step,next_global_step=step+1,actual_outer_updates=step,driver_pid=123,attempt_id=a['attempt_id'],attempted_outer=step,previous_completed_outer=step-1,infrastructure_fault=False,progress=a['progress'],bapo=algorithm['bapo'],outer_contract=outer_contract(step),rank_write_ledger=a['rank_write_ledger'])
  import torch
  torch.save(driver,state/'driver.pt')
  reload=dict(status='EXACT_DRIVER_RELOAD_OK'if step==5 else'DRIVER_SERIALIZED_OK',step=step,pid=123,path=str(state/'driver.pt'),sha256=m.sha(state/'driver.pt'),state_digest=state_digest(driver))
  ranks=[]
  for r in range(8):
   p=state/'ranks'/('rank_%d.pt'%r);p.write_bytes(('fake-rank%d'%r).encode());event=a['rank_write_ledger'][r]
   ranks.append(dict(rank=r,world_size=8,step=step,scheduler_last_epoch=step,native_state_version=3,attempt_id=a['attempt_id'],pid=1000+r,status='EXACT_FULL_STATE_RELOAD_OK'if step==5 else'FULL_STATE_SERIALIZED_OK',roundtrip=step==5,model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=1,optimizer_state_entries=1,latest_rank_event=event,state_digest='x',reloaded_state_digest='x',generation_rng_before_digest='y',generation_rng_after_digest='y',path=str(p),sha256=m.sha(p)))
  return [step,status,algorithm,reload,ranks,driver,a,123,'source',state]
 def test_all_five_prefixes_and_actual_retained_writes(self):
  p=m.checkpoint_prefixes(self.attempts);self.assertEqual(tuple(p),m.SAVE_OUTERS)
  self.assertEqual(p[200]['progress']['actual_adam_writes_per_rank'],[2491]*8)
  for step in m.SAVE_OUTERS:
   x=self.create(step);self.assertEqual(m.metadata(*x),list(range(1000,1008)));self.assertTrue(m.inventories(self.run,step)[1])
 def test_missing_milestone_directory_or_empty_export_rejected(self):
  with self.assertRaises(ValueError):m.inventories(self.run,50)
  self.create(50);(self.run/'checkpoints/actor/global_step_50/tokenizer.json').unlink()
  with self.assertRaisesRegex(ValueError,'Incomplete'):m.inventories(self.run,50)
  a=[x for x in self.attempts if x['candidate_outer']!=100]
  with self.assertRaisesRegex(ValueError,'Missing accepted'):m.checkpoint_prefixes(a)
 def test_zero_saved_counters_false_types_and_wrong_scheduler(self):
  x=self.create(50)
  for field,value in [('actual_outer',0),('completed_outer_updates',False),('actual_outer',50.0)]:
   y=copy.deepcopy(x);y[1][field]=value
   with self.assertRaises(ValueError):m.metadata(*y)
  for field,value in [('scheduler_last_epoch',49),('roundtrip',True),('optimizer_state_entries',0)]:
   y=copy.deepcopy(x);y[4][2][field]=value
   with self.assertRaises(ValueError):m.metadata(*y)
 def test_boundary50_not20_and_zero_write_not_accepted(self):
  x=self.create(50)
  for field,value in [('expected_optimizer_steps',20),('optimizer_steps',0)]:
   y=copy.deepcopy(x);y[5]['rank_write_ledger'][0][field]=value;y[6]['rank_write_ledger']=y[5]['rank_write_ledger'];y[4][0]['latest_rank_event']=y[5]['rank_write_ledger'][0];y[3]['state_digest']=state_digest(y[5])
   with self.assertRaises(ValueError):m.metadata(*y)
 def test_native_hash_mutation_and_export_missing_shard(self):
  x=self.create(100);(x[-1]/'ranks/rank_2.pt').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'full file hash'):m.metadata(*x)
  actor=self.run/'checkpoints/actor/global_step_100';(actor/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'q':'missing.safetensors'}}))
  with self.assertRaises(ValueError):m.inventories(self.run,100)
 def test_symlink_member_refused(self):
  self.create(150);p=self.run/'checkpoints/actor/global_step_150/alias';p.symlink_to('config.json')
  with self.assertRaisesRegex(ValueError,'symlink'):m.inventories(self.run,150)
 def test_horizon_warmup_typed_explicit_and_200required_hf(self):
  c=dict(trainer=dict(total_training_steps=201,acceptance_step=5,save_freq=50),actor_rollout_ref=dict(actor=dict(optim=dict(num_warmup_steps=57))))
  with patch('search_r1.baselines.fresh200_contract.validate_fresh_config',return_value=(200,57)):
   m.horizon(c)
   for container,key,value in [('trainer','total_training_steps',200),('trainer','save_freq',True),('optim','num_warmup_steps',0),('optim','num_warmup_steps',57.)]:
    bad=copy.deepcopy(c);target=bad['trainer']if container=='trainer'else bad['actor_rollout_ref']['actor']['optim'];target[key]=value
    with self.assertRaises(ValueError):m.horizon(bad)
   with self.assertRaisesRegex(ValueError,'Final200 CPU HF'):m.collect_milestones(self.run,self.attempts,123,'source',(5,),None,None,c)
 def test_driver_hash_and_state_progress_bind(self):
  x=self.create(200)
  for idx,field,value in [(5,'actual_outer_updates',199),(2,'algorithm_source_commit','old'),(3,'sha256','0'*64)]:
   y=copy.deepcopy(x);y[idx][field]=value
   with self.assertRaises(ValueError):m.metadata(*y)
 def test_collect_all_five_optional_middle_hf_final_mandatory_and_cost(self):
  for step in m.SAVE_OUTERS:
   x=self.create(step);state=x[-1]
   for name,obj in [('status.json',x[1]),('driver-reload.json',x[3]),('all-rank-write-ledger.json',x[6]['rank_write_ledger'])]: (state/name).write_text(json.dumps(obj))
   for r,report in enumerate(x[4]):(state/'ranks'/('rank_%d.json'%r)).write_text(json.dumps(report))
   (self.run/'checkpoints/algorithm'/('global_step_%d'%step)/'state.json').write_text(json.dumps(x[2]))
  calls=[]
  def hf(base,p):calls.append(p.name);return {'status':'SYNTHETIC_TEST_CALLBACK_NOT_HF_ACCEPTANCE'}
  with patch.object(m,'horizon'):
   out=m.collect_milestones(self.run,self.attempts,123,'source',(5,200),hf,'fixture',{})
  self.assertEqual([x['actual_outer']for x in out],list(m.SAVE_OUTERS));self.assertEqual(calls,['global_step_5','global_step_200'])
  self.assertEqual([x['native_gpu_reload_verified']for x in out],[True,False,False,False,False])
  self.assertEqual(out[1]['cpu_hf']['status'],'NOT_EXECUTED_OPTIONAL_MIDDLE_MILESTONE')
  self.assertTrue(all(x['cost']['wall_seconds']>=0 and x['cost']['driver_cpu_seconds']>=0 for x in out))
 def test_final_coverage_rejects_any_optional_unexecuted_middle_or_fake_finite(self):
  proof=dict(hf=dict(status='REAL_CPU_HF_FINITE_FORWARD',device='cpu',parameters=3085938688,nonfinite_parameters=[],tied_input_output_storage=True,loading_info=dict(missing_keys=[],unexpected_keys=[],mismatched_keys=[],error_msgs=[]),forward=dict(logits_finite=True,device='cpu',logits_shape=[1,5,151936],next_token_id=12)),tokenizer=dict(full_vocabulary_equal=True,bos_equal=True,eos_equal=True,pad_equal=True,special_tokens_equal=True,added_token_metadata_equal=True,chat_template_equal=True,real_chat_template_tokenization_equal=True),parameter_delta=dict(changed=True,comparison='base_cast_to_checkpoint_dtype_then_float64',max_abs_delta=.001))
  entries=[dict(actual_outer=s,cpu_hf=copy.deepcopy(proof))for s in m.SAVE_OUTERS]
  self.assertTrue(m.require_all_milestone_hf(entries))
  for value in (float('inf'),float('nan'),float('-inf'),True):
   bad=copy.deepcopy(entries);bad[0]['cpu_hf']['parameter_delta']['max_abs_delta']=value
   with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
  for value in (3085938688.0,True):
   bad=copy.deepcopy(entries);bad[0]['cpu_hf']['hf']['parameters']=value
   with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
  for key in ('full_vocabulary_equal','bos_equal','eos_equal','pad_equal','special_tokens_equal','added_token_metadata_equal','chat_template_equal','real_chat_template_tokenization_equal'):
   for value in (False,None,1):
    bad=copy.deepcopy(entries);bad[0]['cpu_hf']['tokenizer'][key]=value
    with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
  for key,value in [('device','cuda'),('logits_shape',[1,5.,151936]),('logits_shape',[1,0,151936]),('next_token_id',True),('next_token_id',151936),('logits_finite',False)]:
   bad=copy.deepcopy(entries);bad[0]['cpu_hf']['hf']['forward'][key]=value
   with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
  bad=copy.deepcopy(entries);bad[0]['cpu_hf']['hf']['loading_info']['unexpected_keys']=['bad']
  with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
  for i in range(5):
   bad=copy.deepcopy(entries);bad[i]['cpu_hf']={'status':'NOT_EXECUTED_OPTIONAL_MIDDLE_MILESTONE'}
   with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
  for key,value in [('nonfinite_parameters',['bad']),('tied_input_output_storage',False),('parameters',0)]:
   bad=copy.deepcopy(entries);bad[0]['cpu_hf']['hf'][key]=value
   with self.assertRaises(ValueError):m.require_all_milestone_hf(bad)
if __name__=='__main__':unittest.main()
