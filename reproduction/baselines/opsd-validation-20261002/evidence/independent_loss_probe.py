import ast, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests/baselines'))
from search_r1.baselines.distillation import generalized_jsd, teacher_inputs, FrozenTeacher, opsd_loss
from test_models_and_driver import TinyModels
source=Path(__file__).with_name('author_opsd_trainer.py').read_text()
tree=ast.parse(source)
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generalized_jsd_loss')
fn.decorator_list=[]
namespace={'torch':torch,'F':F}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'pinned_author_function','exec'),namespace)
author=namespace['generalized_jsd_loss']
torch.manual_seed(51)
results=[]
for beta in (0.,.2,.5,1.):
 for temperature in (1.,1.1):
  for clip in (None,.05):
   raw_s=torch.randn(2,4,11)*3
   raw_t=torch.randn(2,4,11)*3
   s=raw_s.clone().requires_grad_(); t=raw_t.clone().requires_grad_()
   loss=generalized_jsd(s,t,beta,temperature,clip).sum()
   loss.backward()
   a=raw_s.clone().requires_grad_(); b=raw_t.clone()
   expected=author(a,b,beta=beta,temperature=temperature,token_clip=clip,reduction='sum')
   expected.backward()
   torch.testing.assert_close(loss,expected,atol=2e-6,rtol=2e-6)
   torch.testing.assert_close(s.grad,a.grad,atol=2e-6,rtol=2e-6)
   assert t.grad is None
   results.append({'beta':beta,'temperature':temperature,'clip':clip,'loss':float(loss.detach()),'max_gradient_error':float((s.grad-a.grad).abs().max())})
TinyModels.setUpClass()
try:
 tok=TinyModels.tokenizer
 teacher=FrozenTeacher(TinyModels.tmp.name,tok)
 from transformers import Qwen2ForCausalLM
 student=Qwen2ForCausalLM.from_pretrained(TinyModels.tmp.name).eval()
 from search_r1.baselines.rollout import padded
 prompts=padded([tok.encode('Question?'),tok.encode('Question capital Paris?')],tok.pad_token_id,left=True)
 rows=[tok.encode('Rome Paris Berlin'),tok.encode('Paris')]
 response=padded(rows,tok.pad_token_id)
 att=response.ne(tok.pad_token_id).long()
 p_att=prompts.ne(tok.pad_token_id).long()
 ids=torch.cat([prompts,response],1); attention=torch.cat([p_att,att],1)
 positions=(attention.cumsum(-1)-1).clamp_min(0)
 loss_mask=att.clone(); loss_mask[0,1::2]=0
 targets=[['Rome','Paris'],['Paris']]
 tid,ta,tp=teacher_inputs(tok,prompts,response,att,targets,teacher.model.config.max_position_embeddings)
 with torch.no_grad():
  tl=teacher.model(tid,attention_mask=ta).logits
  sl=student(ids,attention_mask=attention,position_ids=positions).logits[:,-response.shape[1]-1:-1]
  selected=tl.gather(1,tp.clamp(0,tl.shape[1]-1)[...,None].expand(-1,-1,tl.shape[-1]))
  expected=(generalized_jsd(sl,selected,0.,1.,.05)*loss_mask).sum()/loss_mask.sum()
 from types import SimpleNamespace
 tensors=dict(prompts=prompts,responses=response,input_ids=ids,attention_mask=attention,position_ids=positions,loss_mask=loss_mask)
 settings=SimpleNamespace(beta=0.,temperature=1.,component_clip=.05,chunk_size=1,max_student_logits_gib=4.)
 actual=opsd_loss(student,teacher,tensors,tok,targets,settings,loss_mask.sum())
 torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
 actual.backward()
 assert all(p.grad is None for p in teacher.model.parameters())
 assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
 assert sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None)>0
 alignment={'batch':2,'student_prompt_lengths':p_att.sum(-1).tolist(),'teacher_prompt_lengths':(tp[:,0]+1).tolist(),'response_lengths':att.sum(-1).tolist(),'supervised_tokens':int(loss_mask.sum()),'chunk1_loss':float(actual.detach()),'full_teacher_expected':float(expected),'teacher_gradient_absent':True,'student_gradient_finite_nonzero':True}
finally:
 TinyModels.tearDownClass()
print(json.dumps({'author_loss_cases':results,'tiny_alignment_and_gradient':alignment,'scope':'CPU tiny random Qwen; not 3B/FSDP/GPU/search effectiveness'},indent=2))
