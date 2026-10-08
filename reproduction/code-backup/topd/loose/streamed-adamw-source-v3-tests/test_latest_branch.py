import os,pathlib,runpy,sys,types,ast
SOURCE=pathlib.Path(os.environ['R1_STREAM_SOURCE'])
sys.path.insert(0,str(SOURCE))
QA=runpy.run_path(str(SOURCE/'verl/utils/reward_score/qa_em.py'))
WC=runpy.run_path(str(SOURCE/'verl/utils/weight_continuation.py'))
class Config(dict):
 __getattr__=dict.__getitem__
def test_terminal_reward_rejects_nested_and_later_action():
 for spans in [['<search><answer>Paris</answer></search>'],['<answer>Paris</answer>','<search>next</search>'],['<answer>Paris</answer> trailing']]:
  assert QA['compute_score_generated_em'](spans,{'target':['Paris']},format_score=.5)==0

def test_terminal_reward_preserves_valid_em_and_normalization():
 for s in ['<answer>The Paris.</answer>','<think>reason</think><answer>Paris</answer><|im_end|>']:
  assert QA['compute_score_generated_em']([s],{'target':['Paris']})==1

def test_observation_separation_cannot_manufacture_answer():
 from search_r1.qa_protocol import decode_model_spans,extract_final_answer
 class Tokenizer:
  def decode(self,ids,**kw):return ''.join({1:'<answer>Par',2:'is</answer>',3:'observation'}[i] for i in ids)
 spans=decode_model_spans(Tokenizer(),[1,3,2],[1,0,1])
 assert len(spans)==2 and extract_final_answer(spans) is None

def test_fresh_base_disables_weight_continuation_and_reference_split():
 model=Config(path='/own/base');c=Config(trainer=Config(),actor_rollout_ref=Config(model=model))
 assert WC['completed_updates'](c,101)==0
 assert WC['model_initialization_path'](model,True)==WC['model_initialization_path'](model,False)=='/own/base'

def test_weight_continuation_requires_explicit_paired_count():
 import pytest
 c=Config(trainer=Config(weight_continuation_step=2),actor_rollout_ref=Config(model=Config(path='/own/base')))
 with pytest.raises(ValueError,match='supplied together'):WC['completed_updates'](c,101)
