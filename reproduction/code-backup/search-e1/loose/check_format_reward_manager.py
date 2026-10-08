import ast, runpy, types, torch
from pathlib import Path
from search_r1.qa_protocol import decode_model_spans
root=Path.cwd();qa=types.SimpleNamespace(**runpy.run_path(str(root/'verl/utils/reward_score/qa_em_format.py')))
p=root/'verl/trainer/main_ppo_format.py';tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('RewardManager','_select_rm_score_fn')]
ns=dict(DataProto=object,torch=torch,qa_em_format=qa,decode_model_spans=decode_model_spans)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
class Tokenizer:
 def decode(self,ids):return ''.join(chr(int(i)) for i in ids)
class Batch:
 def __init__(self,prompt,parts):
  response=''.join(t for t,_ in parts); mask=[v for t,v in parts for _ in t]
  self.item=types.SimpleNamespace(batch={'prompts':torch.tensor([ord(c) for c in prompt]),'responses':torch.tensor([ord(c) for c in response]),'attention_mask':torch.ones(len(prompt)+len(response),dtype=torch.long),'info_mask':torch.tensor([1]*len(prompt)+mask)},non_tensor_batch={'reward_model':{'ground_truth':{'target':['Paris']}},'data_source':'nq'})
  self.batch={'responses':self.item.batch['responses'].unsqueeze(0)}
 def __len__(self):return 1
 def __getitem__(self,i):return self.item
manager=ns['RewardManager'](Tokenizer(),0)
prompt='<answer>example</answer><|im_start|>assistant\n'
cases=[([('<search><answer>Paris</answer></search>',1)],0),([('<search>capital</search>',1),('<information><answer>Paris</answer></information>',0)],0),([('<search>capital</search>',1),('<information>Paris</information>',0),('<answer>Paris</answer>',1)],1),([],0)]
for parts,expected in cases:
 result=manager(Batch(prompt,parts));assert float(result.sum())==expected,(parts,result)
print('Actual format RewardManager: 4 cases passed')
