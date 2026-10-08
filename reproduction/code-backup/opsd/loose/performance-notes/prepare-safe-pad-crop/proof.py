"""CPU semantic reference only. No production edits or real HF/FSDP speed claim."""
import copy,json,hashlib,ast
from pathlib import Path
import torch

def crop_outer_padding(t,pad,chunk=16):
 """Preserve internal holes, observations, positions, original nonempty chunk widths."""
 p,r=t['prompts'],t['responses'];P,R=p.shape[1],r.shape[1]
 assert torch.equal(t['input_ids'],torch.cat((p,r),1))
 a=t['attention_mask'];m=t['loss_mask'];assert a.shape==t['input_ids'].shape and m.shape==r.shape
 assert torch.all((a==0)|(a==1)) and torch.all((m==0)|(m==1)) and torch.all(m<=a[:,P:])
 # Only all-row leading prompt columns and strictly all-row inactive right tail.
 used=a[:,:P].bool().any(0);first=int(used.nonzero()[0]) if used.any() else 0
 assert torch.all(p[:,:first]==pad) and not a[:,:first].any()
 response_used=a[:,P:].bool().any(0)
 last=int(response_used.nonzero()[-1])+1 if response_used.any() else 0
 # Never change the width of a formerly nonempty head/JSD chunk.
 kept=min(R,max(1,((last+chunk-1)//chunk)*chunk))
 assert not a[:,P+kept:].any() and not m[:,kept:].any()
 assert torch.all(r[:,kept:]==pad)
 newP=P-first
 out={**t,'prompts':p[:,first:],'responses':r[:,:kept],
      'input_ids':t['input_ids'][:,first:P+kept],
      'attention_mask':a[:,first:P+kept],
      'position_ids':t['position_ids'][:,first:P+kept],
      'loss_mask':m[:,:kept]}
 assert newP>0 and R>0  # A zero-width input needs an explicit caller contract; do not silently repair it.
 return out,dict(original_prompt_width=P,original_response_width=R,prompt_left_removed=first,response_tail_removed=R-kept,new_prompt_width=newP,new_response_width=kept)

def teacher_fn():
 source=Path(__file__).resolve().parents[2]/'opsd/search_r1/baselines/distillation.py'
 node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='teacher_inputs')
 node.body=[n for n in node.body if not isinstance(n,ast.ImportFrom)]
 def padded(rows,pad):
  x=torch.full((len(rows),max(map(len,rows))),pad,dtype=torch.long)
  for i,row in enumerate(rows):x[i,:len(row)]=torch.tensor(row)
  return x
 ns={'torch':torch,'padded':padded};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns)
 return ns['teacher_inputs'],source
class Tokenizer:
 pad_token_id=0
 chat_template='Qwen ChatML fixture'
 all_special_ids=[0]
 def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
  return '<|im_start|>user\n'+messages[0]['content']+'<|im_end|>\n'+('<|im_start|>assistant\n' if add_generation_prompt else '')
 def encode(self,text,add_special_tokens=False):return [10+ord(c)%70 for c in text]
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.embed=torch.nn.Embedding(100,8);self.head=torch.nn.Linear(8,11,bias=False)
 def forward(self,t):
  # Explicit per-query causal reference: identical active K order after boundary crop.
  x=self.embed(t['input_ids'])+t['position_ids'][...,None]*.01
  rows=[]
  for b in range(x.shape[0]):
   values=[]
   for i in range(x.shape[1]):
    keys=x[b,:i+1][t['attention_mask'][b,:i+1].bool()]
    h=x[b,i]*0 if not len(keys) else (torch.softmax((keys*x[b,i]).sum(-1),0)[:,None]*keys).sum(0)
    values.append(self.head(h))
   rows.append(torch.stack(values))
  return torch.stack(rows)
def loss(model,t):
 R=t['responses'].shape[1];student=model(t)[:,-R-1:-1]
 target=torch.arange(11,dtype=torch.float64)[None,None,:]+t['responses'][...,None]*.017
 p=torch.log_softmax(target,-1).exp();value=(p*(torch.log_softmax(target,-1)-torch.log_softmax(student,-1))).sum(-1)
 return (value*t['loss_mask']).sum()/t['loss_mask'].sum().clamp_min(1)
def main():
 torch.manual_seed(42)
 prompts=torch.tensor([[0,0,0,4,5,6,7],[0,0,0,0,4,6,7]])
 response=torch.zeros(2,48,dtype=torch.long);response[:,:19]=torch.arange(1,20);response[1,4:7]=0
 # Real observation context and internal attention hole stay present.
 ids=torch.cat((prompts,response),1);attention=(ids!=0).long();mask=attention[:,7:].clone();mask[:,8:12]=0
 t=dict(prompts=prompts,responses=response,input_ids=ids,attention_mask=attention,position_ids=(attention.cumsum(-1)-1).clamp_min(0),loss_mask=mask)
 cropped,shape=crop_outer_padding(t,0)
 f,source=teacher_fn();oldids,olda,oldpos=f(Tokenizer(),prompts,response,attention[:,7:],[['gold'],['gold']])
 newids,newa,newpos=f(Tokenizer(),cropped['prompts'],cropped['responses'],cropped['attention_mask'][:,cropped['prompts'].shape[1]:],[['gold'],['gold']])
 assert torch.equal(oldids,newids) and torch.equal(olda,newa) and torch.equal(oldpos[:,:32],newpos)
 # Exercise current structured teacher adapter too, using verified suffix token IDs.
 suffix=torch.tensor(Tokenizer().encode('<|im_end|>\n<|im_start|>assistant\n'))
 structured=torch.cat((prompts,suffix.expand(2,-1)),1)
 structured_crop=structured[:,3:]
 si,sa,sp=f(Tokenizer(),structured,response,attention[:,7:],[['gold'],['gold']],prompt_mode='user_context')
 ni,na,np=f(Tokenizer(),structured_crop,cropped['responses'],cropped['attention_mask'][:,4:],[['gold'],['gold']],prompt_mode='user_context')
 assert torch.equal(si,ni) and torch.equal(sa,na) and torch.equal(sp[:,:32],np)
 a=Tiny().double();b=copy.deepcopy(a);oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4)
 comparisons=[]
 for step in range(2):
  oa.zero_grad();ob.zero_grad();la=loss(a,t);lb=loss(b,cropped);la.backward();lb.backward()
  gd=max(float((x.grad-y.grad).abs().max()) for x,y in zip(a.parameters(),b.parameters()));ld=float((la-lb).abs())
  assert gd<=1e-12 and ld<=1e-12
  oa.step();ob.step();pd=max(float((x-y).abs().max()) for x,y in zip(a.parameters(),b.parameters()));assert pd<=1e-12
  comparisons.append(dict(step=step+1,loss_max_abs=ld,gradient_max_abs=gd,parameter_max_abs=pd))
 assert torch.equal(cropped['responses'][:,8:12],response[:,8:12])
 assert torch.equal(cropped['attention_mask'][:,cropped['prompts'].shape[1]+4:cropped['prompts'].shape[1]+7],attention[:,7+4:7+7])
 result=dict(scope='CPU tiny float64 semantic reference tolerance1e-12; NOT HF BF16 exact/FSDP/GPU speed or completed performance fix',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),shape=shape,teacher_context_and_target_positions_exact=True,teacher_prompt_modes=['legacy','user_context'],observations_and_internal_holes_preserved=True,two_step_comparisons=comparisons)
 Path(__file__).with_name('cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
