import asyncio,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from common_em import *
assert score('<answer>The U2!</answer>',{'target':['u2','other']})['em']==1
for text in ['<search><answer>U2</answer></search>','<answer><answer>U2</answer></answer>','<answer>U2</answer> more','<answer>U2','<answer>U2</answer><answer>U2</answer>','<reason><answer>U2</answer>']:
 assert score(text,['U2'])['em']==0,text
assert score('<reason>bad format?</reason><answer>wrong</answer>',['U2'])['em']==0
assert score('<answer>U2</answer>',['U2'],False)['em']==0
assert score('<answer>red blue</answer>',['red green'])['f1']==.5
rows=[dict(score('<answer>yes</answer>',['yes']),data_source='nq')]+[dict(score('<answer>no</answer>',['yes']),data_source='nq') for _ in range(2559)]
assert summarize(rows)['em']==1/2560 and summarize(rows)['trajectories']==2560
assert mixed_first(rows*2,rows)['candidate_pool']['trajectories']==5120
# Actual boundary hook excludes prompt/tool answers and chooses final model action.
from common_em_runtime import install_loop
class Tokenizer:
 def decode(self,tokens,skip_special_tokens=True):return ''.join(tokens)
class Loop:
 response_length=1000;tokenizer=Tokenizer()
 async def _handle_generating_state(self,data,*args):
  data.response_ids=list(data.next);data.response_mask.extend([1]*len(data.response_ids));return 'end'
 async def run(self,params,**kwargs):
  class Data:pass
  d=Data();d.extra_fields={};d.response_mask=[];d.next='<search>U2</search>'
  await self._handle_generating_state(d,{})
  d.response_mask.extend([0]*len('<answer>U2</answer>'))
  d.next='<answer>wrong</answer>';await self._handle_generating_state(d,{})
  return d,[]
install_loop(Loop)
x=asyncio.run(Loop().run({},raw_prompt=[{'role':'user','content':'Answer the given question.\nQuestion: q'}],extra_info={},reward_model={'ground_truth':{'target':['U2']}},data_source='nq'))[0]
assert score(x.extra_fields['common_model_actions'][-1]['text'],['U2'])['em']==0
assert any(v==0 for v in x.extra_fields['common_response_mask'])
from prepare_source import patch_trainer
p=Path('work/cosearch-author-review/verl/verl/trainer/ppo/search_r1_reranker_ray_trainer.py')
s=patch_trainer(p.read_text());assert 'metrics.update(completed_outer(self.global_steps))' in s
assert '(self.global_steps in (5, 50, 100, 150, 200))' in s
print('PASS adversarial extraction, aliases, EM/F1,2560 denominator,mixed pool,boundary hook,exact validation milestones')
