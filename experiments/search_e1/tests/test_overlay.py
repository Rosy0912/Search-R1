"""Exercise the actual runtime RewardManager and preserve archived source hashes."""
import ast,json,hashlib,runpy,unittest,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from search_r1.qa_protocol import decode_model_spans
ROOT=Path(os.environ['E1_TEST_RUNTIME'])
qa=SimpleNamespace(**runpy.run_path(str(ROOT/'upstream/Search-R1/verl/utils/reward_score/qa_em.py')))
history=SimpleNamespace(**runpy.run_path(str(ROOT/'search_e1/historical_qa_em.py')))
source=ROOT/'upstream/Search-R1/verl/trainer/main_ppo.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['RewardManager','_select_rm_score_fn']]
namespace=dict(torch=torch,np=np,qa_em=qa,historical_qa_em=history,decode_model_spans=decode_model_spans,DataProto=object)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
class Tokenizer:
 def decode(self,ids):return ''.join(chr(int(i)) for i in ids)
class Batch:
 def __init__(self,prompt,response,model_mask):
  ids=lambda s:torch.tensor([ord(c) for c in s],dtype=torch.long)
  self.item=SimpleNamespace(batch=dict(prompts=ids(prompt),responses=ids(response),attention_mask=torch.ones(len(prompt+response),dtype=torch.long),info_mask=torch.tensor([1]*len(prompt)+model_mask)),non_tensor_batch={'data_source':'nq','reward_model':{'ground_truth':{'target':['Paris']}}})
  self.batch={'responses':self.item.batch['responses'].unsqueeze(0)}
 def __len__(self):return 1
 def __getitem__(self,i):return self.item
class OverlayTests(unittest.TestCase):
 def score(self,response,mask=None):
  data=Batch('<answer>example</answer>',response,mask if mask is not None else [1]*len(response))
  rm=namespace['RewardManager'](Tokenizer(),0)
  value=rm(data).sum().item();return value,rm.last_metrics
 def test_search_nested_rejected_and_diagnostic_preserved(self):
  score,metrics=self.score('<search><answer>Paris</answer></search>')
  self.assertEqual(score,0);self.assertEqual(metrics['audit/legacy_em'],1);self.assertEqual(metrics['audit/score_disagreement'],1)
 def test_observation_answer_excluded(self):
  search='<search>capital</search>';obs='<information><answer>Paris</answer></information>'
  score,metrics=self.score(search+obs,[1]*len(search)+[0]*len(obs))
  self.assertEqual(score,0);self.assertEqual(metrics['audit/legacy_em'],1)
 def test_normal_final_answer(self):self.assertEqual(self.score('<think>Found it</think><answer>Paris</answer>')[0],1)
 def test_later_invalid_span_cannot_reuse_answer(self):
  a='<answer>Paris</answer>';obs='<information>x</information>';end='invalid'
  self.assertEqual(self.score(a+obs+end,[1]*len(a)+[0]*len(obs)+[1]*len(end))[0],0)
 def test_zero_response(self):self.assertEqual(self.score('')[0],0)
 def test_rollout_core_uses_provenance(self):
  core=runpy.run_path(str(ROOT/'search_e1/core.py'))
  response='<search>x</search><information><answer>Paris</answer></information>'
  self.assertEqual(core['reward'](response,['Paris'],[(0,len('<search>x</search>'))]),0)
  response='<answer>Paris</answer>'
  self.assertEqual(core['reward'](response,['Paris'],[(0,len(response))]),1)
 def test_immutable_archive(self):
  archive=Path(os.environ['E1_TEST_ARCHIVE'])
  for record in json.loads((archive/'PACKAGE_MANIFEST.json').read_text()):
   self.assertEqual(hashlib.sha256((archive/record['path']).read_bytes()).hexdigest(),record['sha256'],record['path'])
if __name__=='__main__':unittest.main()
