import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
import search_r1.baselines.recursive_b_schedule_resume as resume

class ResumeTests(unittest.TestCase):
 def test_only_schedule_changes_allowed(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'parent.yaml';old=OmegaConf.create(dict(trainer=dict(evaluation_steps=None,checkpoint_save_steps=None,total_training_steps=201),actor_rollout_ref=dict(actor=dict(optim=dict(lr=1e-6,total_training_steps=200))),critic=dict(optim=dict(total_training_steps=0))))
   p.write_text(OmegaConf.to_yaml(old));b=dict(parent_config=str(p),parent_config_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   new=copy.deepcopy(old);new.trainer.evaluation_steps=[5,50,100,150,200];new.trainer.checkpoint_save_steps=[5,50,100,150,200]
   with patch.object(resume,'binding',return_value=b),patch.dict(os.environ,RECURSIVE_B_CONTINUATION_BINDING='present'):
    self.assertTrue(resume.validate_entry(new));new.actor_rollout_ref.actor.optim.lr=2e-6
    with self.assertRaises(ValueError):resume.validate_entry(new)
    new.actor_rollout_ref.actor.optim.lr=1e-6;new.trainer.evaluation_steps=[5,10]
    with self.assertRaises(ValueError):resume.validate_entry(new)
 def stream(self,tmp):
  value=torch.arange(24,dtype=torch.uint8);digest=hashlib.sha256(value.numpy().tobytes()).hexdigest();meta=dict(protocol='recursive-b-native-boundary-sampling-rng-v1',completed_step=4,actual_optimizer_calls=80,scheduler_step=4,rank=0,world_size=4,run_id='parent',source_commit='commit',gen_dp_rank=0,state_sha256=digest);p=Path(tmp)/'rng.pt';torch.save(dict(metadata=meta,gen_random_states=value),p)
  r=dict(rank=0,path=str(p),file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),state_sha256=digest)
  b=dict(parent_run_id='parent',parent_commit='commit',sampling_streams=[r]+[dict(r,rank=i) for i in range(1,4)])
  w=SimpleNamespace(rank=0,rollout_sharding_manager=SimpleNamespace(gen_random_states=torch.zeros_like(value),device_mesh={'dp':SimpleNamespace(get_local_rank=lambda:0)}))
  return w,b,value
 def test_separate_sampling_restore_does_not_draw_training_rng(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(resume,'PARENT_COMPLETED_STEP',4,create=True),patch.object(resume,'PARENT_OPTIMIZER_CALLS',80,create=True):
   w,b,value=self.stream(tmp);before=torch.get_rng_state().clone();receipt=resume.restore_sampling_stream(w,b)
   self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(torch.equal(value,w.rollout_sharding_manager.gen_random_states));self.assertTrue(receipt['actual_separate_sampling_cuda_stream_restored']);self.assertFalse(receipt['executor_bitwise_equivalence'])
 def test_stream_bytes_and_rank_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(resume,'PARENT_COMPLETED_STEP',4,create=True),patch.object(resume,'PARENT_OPTIMIZER_CALLS',80,create=True):
   w,b,value=self.stream(tmp);b['sampling_streams'][0]['file_sha256']='0'*64
   with self.assertRaises(ValueError):resume.restore_sampling_stream(w,b)
   b['sampling_streams'][0]['file_sha256']=hashlib.sha256(Path(b['sampling_streams'][0]['path']).read_bytes()).hexdigest();b['sampling_streams'][3]['rank']=0
   with self.assertRaises(ValueError):resume.restore_sampling_stream(w,b)

if __name__=='__main__':unittest.main()
