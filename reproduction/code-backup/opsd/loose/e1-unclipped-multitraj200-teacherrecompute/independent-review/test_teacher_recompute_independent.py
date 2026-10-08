"""Local tiny-model checkpoint/input-retention proof, not nativeGPU evidence."""
import copy, pathlib, sys, unittest
from types import SimpleNamespace
ROOT=pathlib.Path(__file__).resolve().parents[2]/'opsd-unclipped-multitraj200-teacherrecompute'
sys.path.insert(0,str(ROOT))
import torch
from transformers import Qwen2ForCausalLM
from search_r1.baselines.distillation import FrozenTeacher,opsd_loss
from tests.baselines import test_models_and_driver as fixtures
from tests.baselines.test_opsd_zero_initialization import tensors_and_settings
class IndependentTeacherRecompute(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixtures.TinyModels.setUpClass()
 @classmethod
 def tearDownClass(cls):fixtures.TinyModels.tearDownClass()
 def data(self,chunk=3):
  tok=fixtures.TinyModels.tokenizer;ts,settings=tensors_and_settings(tok,chunk)
  ts={k:v.repeat((2,1)) for k,v in ts.items()}
  ts['responses'][3,-1]=tok.pad_token_id;ts['input_ids'][3,-1]=tok.pad_token_id
  ts['attention_mask'][3,-1]=0;ts['loss_mask'][3,-1]=0
  settings.component_clip=None;settings.response_only_logits=False;settings.teacher_forward_micro_batch=4
  settings.loss_checkpointing=True
  return ts,settings
 def run_variants(self,chunk=3,checkpoint=True,empty=False,zero_middle=False):
  tok=fixtures.TinyModels.tokenizer;ts,settings=self.data(chunk);settings.loss_checkpointing=checkpoint
  if empty:ts['loss_mask'].zero_()
  if zero_middle:ts['loss_mask'][:,chunk:2*chunk]=0
  teacher=FrozenTeacher(fixtures.TinyModels.tmp.name,tok);out=[]
  for recompute in (False,True):
   student=Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
   settings.teacher_logits_recompute=recompute
   head_calls=[];h=teacher.model.lm_head.register_forward_hook(lambda m,a,o:head_calls.append(tuple(o.shape)))
   loss=opsd_loss(student,teacher,ts,tok,[['Paris'],['Rome'],['Berlin'],['Paris']],settings,ts['loss_mask'].sum().clamp_min(1))
   forward_calls=len(head_calls);loss.backward();h.remove()
   gradients={n:p.grad.clone() for n,p in student.named_parameters()}
   self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(g).all() for g in gradients.values()))
   self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
   out.append((loss.detach(),gradients,forward_calls,len(head_calls)))
  torch.testing.assert_close(out[0][0],out[1][0],atol=0,rtol=0)
  for name in out[0][1]:torch.testing.assert_close(out[0][1][name],out[1][1][name],atol=0,rtol=0)
  self.assertEqual(out[0][2],out[1][2])
  self.assertEqual(out[0][3],out[0][2]);self.assertEqual(out[1][3],out[1][2]*(2 if checkpoint else 1))
  return out
 def test_chunk1_checkpoint_all_student_grads_exact(self):self.run_variants(chunk=1)
 def test_chunk3_checkpoint_all_student_grads_exact(self):self.run_variants(chunk=3)
 def test_no_checkpoint_same_math_all_grads(self):self.run_variants(checkpoint=False)
 def test_all_empty_keeps_connected_zero_and_no_projection(self):
  out=self.run_variants(empty=True)
  self.assertEqual(float(out[1][0]),0.);self.assertEqual(out[1][2],0)
  self.assertTrue(all(torch.count_nonzero(g)==0 for g in out[1][1].values()))
 def test_zero_middle_chunk_preserves_late_chunk_binding(self):self.run_variants(zero_middle=True)
 def test_bool_and_frozen_guard(self):
  tok=fixtures.TinyModels.tokenizer;ts,settings=self.data();teacher=FrozenTeacher(fixtures.TinyModels.tmp.name,tok)
  student=Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
  settings.teacher_logits_recompute=1
  with self.assertRaisesRegex(ValueError,'boolean'):opsd_loss(student,teacher,ts,tok,[['Paris']]*4,settings,1)
  settings.teacher_logits_recompute=True;next(teacher.model.parameters()).requires_grad_(True)
  with self.assertRaisesRegex(ValueError,'frozen teacher'):opsd_loss(student,teacher,ts,tok,[['Paris']]*4,settings,1)
 def test_invalid_supervised_logit_still_rejected(self):
  tok=fixtures.TinyModels.tokenizer;ts,settings=self.data();teacher=FrozenTeacher(fixtures.TinyModels.tmp.name,tok)
  values=torch.randn((*ts['input_ids'].shape,len(tok)),requires_grad=True)
  width=ts['responses'].shape[1];values.data[0,-width-1,17]=float('nan')
  class Direct:
   def __call__(self,**kwargs):return SimpleNamespace(logits=values)
  for recompute in (False,True):
   settings.teacher_logits_recompute=recompute
   with self.assertRaisesRegex(ValueError,'Nonfinite OPSD logits'):opsd_loss(Direct(),teacher,ts,tok,[['Paris']]*4,settings,1)
 def test_checkpoint_saves_features_instead_of_second_vocab_input(self):
  tok=fixtures.TinyModels.tokenizer;ts,settings=self.data();teacher=FrozenTeacher(fixtures.TinyModels.tmp.name,tok)
  counts=[]
  for recompute in (False,True):
   settings.teacher_logits_recompute=recompute;student=Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval();saved=[]
   def pack(t):saved.append(tuple(t.shape));return t
   with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
    loss=opsd_loss(student,teacher,ts,tok,[['Paris']]*4,settings,ts['loss_mask'].sum())
   vocab=sum(len(s)==3 and s[0]==4 and s[-1]==len(tok) and s[1]<=settings.chunk_size for s in saved)
   features=sum(len(s)==3 and s[0]==4 and s[-1]==student.config.hidden_size and s[1]<=settings.chunk_size for s in saved)
   counts.append((vocab,features))
  self.assertGreater(counts[0][0],counts[1][0]);self.assertGreater(counts[1][1],counts[0][1])
  # Original head/masking also retains its finite-logit check tensor;
  # compare the actual input-retention reduction rather than assuming2:1.
  self.assertGreaterEqual(counts[0][0]-counts[1][0],counts[1][1]-counts[0][1])
if __name__=='__main__':unittest.main()
