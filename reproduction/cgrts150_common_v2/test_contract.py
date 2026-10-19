"""New namespace, midrun150 and actual CPU HF projection fault tests."""
import copy,json,tempfile,unittest
from pathlib import Path
from . import contract as c
from .progress import ProgressObserver

def fixture():
 f=json.loads((Path(__file__).parent/'test_synthetic_lineage.json').read_text())
 assert f['fixture_only_not_checkpoint_or_evaluation_acceptance'] is True
 n,m,t,s=f['arguments']
 model_files={name:h for name,h in n['hf_identity']['files'].items() if name.endswith('.safetensors') or name in ('model.safetensors.index.json','config.json','generation_config.json')}
 model_files.update(c.CANONICAL_FILES)
 p=dict(profile=c.PROFILE,science=c.SCIENCE,kind='trained-cgrts',repeat=1,repeat_count=1,job=c.job(1),root=str(c.root(1)),training_job_id='139134',training_source=c.TRAIN_SOURCE,actual_training_iterations=150,training_limit_remains=200,native_boundary='pre-evaluation',accepted_model=c.MODEL,hf_files=n['hf_identity']['files'],model=c.VIEW,model_revision=None,model_identity=c.identity(n),model_files=model_files,data=c.ROOT+'/final.jsonl',data_sha256=c.FINAL_JSONL,id_gold_identity=c.ID_GOLD,data_manifest_sha256=c.DATA_MANIFEST,retriever_receipt=c.BASE+'/receipts/gpu-retriever-ready.json',gpus=8,cpus=64,memory_gib=600,no_auto_retry=True,source_commit='f'*40,source=c.BASE+'/sources/'+'f'*40,source_files=c.SCIENCE_FILES)
 return p,n,m,t,s

class ContractTests(unittest.TestCase):
 def reject(self,change):
  args=copy.deepcopy(fixture());change(*args)
  with self.assertRaises((ValueError,KeyError)):c.validate_plan(*args)
 def test_running150_n1_uses_measured_counters(self):
  p,n,m,t,s=fixture();self.assertEqual(c.validate_plan(p,n,m,t,s),c.root(1));self.assertEqual(n['scheduler'][3],'RUNNING')
  record=c.validate_lineage(n,m,t,s);self.assertIsNone(record['sample_variance']);self.assertNotIn(3000,record['actual_optimizer_calls_per_rank'].values())
 def test_target200_not_changed(self):self.reject(lambda p,n,m,t,s:p.update(training_limit_remains=150))
 def test_wrong_step(self):self.reject(lambda p,n,m,t,s:p.update(actual_training_iterations=200))
 def test_posteval_boundary(self):self.reject(lambda p,n,m,t,s:p.update(native_boundary='post-evaluation'))
 def test_repeated_n1(self):self.reject(lambda p,n,m,t,s:p.update(repeat_count=3))
 def test_second_slot(self):
  with self.assertRaises(ValueError):c.job(2)
 def test_boolean_slot(self):
  with self.assertRaises(ValueError):c.job(True)
 def test_foreign_model(self):self.reject(lambda p,n,m,t,s:p.update(accepted_model=c.CANONICAL))
 def test_hf_sha(self):self.reject(lambda p,n,m,t,s:p.update(hf_files={'model.safetensors':'0'*64}))
 def test_data(self):self.reject(lambda p,n,m,t,s:p.update(data_sha256='0'*64))
 def test_science(self):self.reject(lambda p,n,m,t,s:p.update(science=dict(max_new_tokens=5)))
 def test_other_training_job(self):self.reject(lambda p,n,m,t,s:p.update(training_job_id='139135'))
 def test_failed_training(self):self.reject(lambda p,n,m,t,s:n['scheduler'].__setitem__(3,'FAILED'))
 def test_missing_adam(self):self.reject(lambda p,n,m,t,s:n['workers'][0].update(actual_adam_min=0))
 def test_missing_original_sha(self):self.reject(lambda p,n,m,t,s:n.update(local_small_evidence_sha_verified=False))
 def test_progress_transient_and_bounded_failure(self):
  with tempfile.TemporaryDirectory() as tmp:
   clock=[0.];obs=ProgressObserver(clock=lambda:clock[0]);p=Path(tmp)/'rank-0.progress.json'
   self.assertEqual(obs.read(tmp),[])
   p.write_text(json.dumps(dict(rank=0,total=len(range(0,51713,8)),completed=32)))
   self.assertEqual(obs.read(tmp)[0]['completed'],32);p.unlink();clock[0]=1
   self.assertEqual(obs.read(tmp)[0]['completed'],32);clock[0]=61
   with self.assertRaises(TimeoutError):obs.read(tmp)
 def tiny(self,dtype,missing=False):
  try:import torch
  except ImportError:raise unittest.SkipTest('Remote old inference runtime owns real Torch CPU tests')
  from transformers import AutoModelForCausalLM,Qwen2Config
  from .operations import reload_model
  with tempfile.TemporaryDirectory() as tmp:
   cfg=Qwen2Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=False)
   model=AutoModelForCausalLM.from_config(cfg).to(dtype)
   # FP32 weights include values outside the exactly BF16-representable set.
   if dtype==torch.float32:
    with torch.no_grad():next(model.parameters()).view(-1)[0]=0.123456789
   model.save_pretrained(tmp);shapes={n:list(v.shape) for n,v in model.state_dict().items()};del model
   if missing:
    shapes.pop('lm_head.weight')
    with self.assertRaises(ValueError):reload_model(tmp,shapes)
   else:
    record=reload_model(tmp,shapes);self.assertTrue(record['all_tensors_finite_bf16_and_exact_saved_weight_bf16_projection']);self.assertEqual(record['logical_parameters'],15);self.assertEqual(record['original_saved_tensor_dtypes'],{str(dtype):15})
 def test_tiny_actual_fp32_to_bf16_projection(self):
  try:import torch
  except ImportError:raise unittest.SkipTest('Remote Torch fixture')
  self.tiny(torch.float32)
 def test_tiny_actual_bf16_reload(self):
  try:import torch
  except ImportError:raise unittest.SkipTest('Remote Torch fixture')
  self.tiny(torch.bfloat16)
 def test_tiny_incomplete_reload(self):
  try:import torch
  except ImportError:raise unittest.SkipTest('Remote Torch fixture')
  self.tiny(torch.float32,missing=True)

if __name__=='__main__':unittest.main(verbosity=2)
