import copy,tempfile,unittest
from pathlib import Path
from . import contract as c

def fixture():
 jid='999992';sub=dict(job_id=jid,source=c.TRAIN_SOURCE)
 cold=dict(job_id=jid,parent_job_id='139135',source=c.TRAIN_SOURCE,local_raw_sha_verified=True,actual_loaded_model_adam_scheduler_saved_rng_exact=True,actual_first_batch_ids_and_driver_exact=True,actual_complete_first_resumed_iteration=True,rank_adam=[dict(preserved_actual_adam_calls=157) for _ in range(8)])
 workers=[dict(rank=i,scheduler=dict(last_epoch=150,_last_lr=[1e-6]),optimizer=dict(state={str(k):dict(step=167) for k in range(37)}),actual_adam_calls=167) for i in range(8)]
 files={f'{kind}_world_size_8_rank_{r}.pt':dict(sha256=f'{r:064x}') for r in range(8) for kind in ('model','optim','extra_state')}
 native=dict(job_id=jid,parent_job_id='139135',source=c.TRAIN_SOURCE,local_raw_sha_verified=True,status='FINAL250_ACTUAL150_ALL24_SHA_MODEL_MEASURED_ADAM_RNG_SCHEDULER_DATA_CPU_ACCEPTED',actual_updates=150,label=250,size='3b',native=dict(world_size=8,actual_updates=150,label=250,files=files,workers=workers,cuda_initialized=False),shapes={str(i):[1] for i in range(435)},dataloader=dict(next_label=251,next_ids=list(range(256))),scheduler=[jid,'jmzhang','name','COMPLETED','0:0'])
 hf=dict(job_id=jid,source=c.TRAIN_SOURCE,local_raw_sha_verified=True,status='OFFICIAL_CPU_MERGE_EXACT_ALL_LOGICAL_BF16_FULL_HF_RELOAD_ACCEPTED',actual_updates=150,label=250,size='3b',target=c.MODEL,files={'model.safetensors':dict(sha256='f'*64)},projection=dict(exact_native_bf16_projection_all_parameters=True,logical_parameters=435,rank_parameter_comparisons=3480,parameters_different_from_initialization=['weight']),actual_complete_hf_cpu_reload=True,no_cuda_initialized=True,semantics=dict(complete_vocabulary_special_tokens_and_chat_template_equal=True,eos_token_ids=[151645,151643]))
 intent=dict(sub=sub,actual_updates=150,label=250,native_identity={n:r['sha256'] for n,r in files.items()})
 template={k:'fixture' for k in c.INHERITED};template.update(base=c.BASE,backend='slurm_lumia',data_manifest_sha256=c.DATA_SHA,ordered_ids_sha256=c.IDS_SHA)
 plan=dict(template,protocol='branpo3b_final150_common_v1',group='BranPO',outer=150,model=c.MODEL,model_files={n:v['sha256'] for n,v in hf['files'].items()},training_job_id=jid,training_source=c.TRAIN_SOURCE,checkpoint_identity=c.identity(native,hf),template_plan=c.TEMPLATE,template_plan_sha256=c.TEMPLATE_SHA,protocol_template_config_sha256=c.CONFIG_SHA,evidence={},inference_model=c.ROOT+'/model-view',evaluation_code_commit='f'*40,evaluation_run_id='a'*32,job=c.job(1),output=str(c.root(1)/'output'),repeat_index=1,repeat_count=3,no_auto_retry=True,allocation=dict(gpus=8,cpus=64,memory_gib=600))
 return cold,native,hf,intent,template,plan
class TestContract(unittest.TestCase):
 def test_complete_lineage_real_counter_not_iteration(self):
  cold,n,h,i,t,p=fixture();self.assertEqual(c.validate_lineage(cold,n,h,i),n['job_id']);self.assertEqual(c.validate_plan(p,t,n,h),c.root(1))
 def reject(self,mutate):
  values=fixture();mutate(*values)
  with self.assertRaises(ValueError):c.validate_lineage(*values[:4]);c.validate_plan(values[5],values[4],values[1],values[2])
 def test_old200_target(self):self.reject(lambda cold,n,h,i,t,p:n.update(actual_updates=200))
 def test_label150_not_actual150(self):self.reject(lambda cold,n,h,i,t,p:n.update(label=150))
 def test_live_terminal(self):self.reject(lambda cold,n,h,i,t,p:n['scheduler'].__setitem__(3,'RUNNING'))
 def test_native_missing_rank(self):self.reject(lambda cold,n,h,i,t,p:n['native']['files'].pop(next(iter(n['native']['files']))))
 def test_adam_reset(self):self.reject(lambda cold,n,h,i,t,p:n['native']['workers'][0].update(actual_adam_calls=150))
 def test_no_cold_acceptance(self):self.reject(lambda cold,n,h,i,t,p:cold.update(actual_complete_first_resumed_iteration=False))
 def test_wrong_hf_model(self):self.reject(lambda cold,n,h,i,t,p:h.update(target='/other/model'))
 def test_no_local_backup(self):self.reject(lambda cold,n,h,i,t,p:h.update(local_raw_sha_verified=False))
 def test_incomplete_reload(self):self.reject(lambda cold,n,h,i,t,p:h.update(actual_complete_hf_cpu_reload=False))
 def test_changed_eos(self):self.reject(lambda cold,n,h,i,t,p:h['semantics'].update(eos_token_ids=[151645]))
 def test_export_different_native(self):self.reject(lambda cold,n,h,i,t,p:i['native_identity'].update(model_world_size_8_rank_0='bad'))
 def test_protocol_config_changed(self):self.reject(lambda cold,n,h,i,t,p:p.update(protocol_template_config_sha256='bad'))
 def test_fourth_repeat(self):
  with self.assertRaises(ValueError):c.job(4)
 def test_tiny_actual_hf_reload_all_tensors_and_tied_pointer(self):
  try:import torch
  except ImportError:raise unittest.SkipTest('Remote runtime owns actual torch CPU fixture')
  from transformers import AutoModelForCausalLM,Qwen2Config
  from .operations import reload_model
  with tempfile.TemporaryDirectory() as tmp:
   cfg=Qwen2Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True)
   model=AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16);model.save_pretrained(tmp)
   shapes={n:list(v.shape) for n,v in model.state_dict().items()};del model
   record=reload_model(tmp,shapes);self.assertTrue(record['tied_pointer_shared']);self.assertEqual(record['logical_parameters'],15)
 def test_tiny_hf_reload_missing_shape_rejected(self):
  try:import torch
  except ImportError:raise unittest.SkipTest('Remote runtime owns actual torch CPU fixture')
  from transformers import AutoModelForCausalLM,Qwen2Config
  from .operations import reload_model
  with tempfile.TemporaryDirectory() as tmp:
   cfg=Qwen2Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True)
   model=AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16);model.save_pretrained(tmp)
   shapes={n:list(v.shape) for n,v in model.state_dict().items()};del model;shapes.pop('lm_head.weight')
   with self.assertRaises(ValueError):reload_model(tmp,shapes)
if __name__=='__main__':unittest.main()
