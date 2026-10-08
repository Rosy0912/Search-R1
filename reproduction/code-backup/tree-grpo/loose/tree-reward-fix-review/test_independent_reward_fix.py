import ast,pathlib,runpy,types,unittest,sys,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]/'tree-grpo';sys.path.insert(0,str(ROOT))
from search_r1.qa_protocol import decode_model_spans,extract_final_answer,parse_model_action
QA=runpy.run_path(str(ROOT/'verl/utils/reward_score/qa_em.py'))
FMT=runpy.run_path(str(ROOT/'verl/utils/reward_score/qa_em_format.py'))
class Tok:
 def decode(self,ids):return ''.join(chr(int(x)) for x in ids)
class Data:
 def __init__(self,response,mask):
  ids=torch.tensor([ord(c) for c in response]);prompt=torch.tensor([ord('P')]);self.batch={'responses':ids[None,:],'prompts':prompt[None,:],'attention_mask':torch.ones(1,1+len(ids),dtype=torch.long),'info_mask':torch.tensor([[1]+mask])};self.non_tensor_batch={'reward_model':{'ground_truth':{'target':['Paris']}},'data_source':'nq'}
 def __len__(self):return 1
 def __getitem__(self,i):return types.SimpleNamespace(batch={k:v[i] for k,v in self.batch.items()},non_tensor_batch=self.non_tensor_batch)
class Test(unittest.TestCase):
 def test_final_boundary(self):
  for spans in [['<search><answer>Paris</answer></search>'],['<answer>Paris</answer>','<search>again</search>'],['<answer>Par','is</answer>'],['<answer>Paris</answer>','']]:self.assertEqual(QA['compute_score_generated_em'](spans,{'target':['Paris']},format_score=.3),0)
  self.assertEqual(QA['compute_score_generated_em'](['<search>q</search>','<think>found</think><answer>Paris</answer>'],{'target':['Paris']}),1)
 def test_observation_mask_no_join(self):
  text='<answer>Par'+'ENV'+'is</answer>';spans=decode_model_spans(Tok(),[ord(c) for c in text],[1]*11+[0]*3+[1]*11)
  self.assertIsNone(extract_final_answer(spans))
 def test_optional_real_manager_info_mask(self):
  path=ROOT/'verl/trainer/main_ppo_format.py';tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('_select_rm_score_fn','RewardManager')];ns=dict(torch=torch,DataProto=Data,qa_em=types.SimpleNamespace(**QA),qa_em_format=types.SimpleNamespace(**FMT),decode_model_spans=decode_model_spans)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);manager=ns['RewardManager'](Tok(),0)
  observation='<information><answer>Paris</answer></information>';search='<search>q</search>'
  d=Data(search+observation,[1]*len(search)+[0]*len(observation));self.assertEqual(manager(d).sum().item(),0)
  answer='<answer>Paris</answer>';d=Data(search+observation+answer,[1]*len(search)+[0]*len(observation)+[1]*len(answer));self.assertEqual(manager(d).sum().item(),1)
  d=Data('<answer>Par'+'ENV'+'is</answer>',[1]*11+[0]*3+[1]*11);self.assertEqual(manager(d).sum().item(),0)
 def test_earlier_answer_not_reused(self):
  self.assertIsNone(extract_final_answer(['<answer>Paris</answer>','<search>q</search>']))
 def test_reserved_and_eos(self):
  for t in ['<search><answer>Paris</answer></search>','<answer>Paris</answer><search>tail','<think><search>q</search></think><answer>Paris</answer>']:self.assertIsNone(parse_model_action(t))
  self.assertEqual(parse_model_action('<answer>Paris</answer><|im_end|>'),('answer','Paris'))
if __name__=='__main__':unittest.main(verbosity=2)
