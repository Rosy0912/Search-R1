import ast, math, pathlib, types, sys, torch
import torch.nn.functional as F
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-R1-RL-Baselines')
from search_r1.baselines.distillation import generalized_jsd, teacher_inputs, opsd_loss, global_token_denominator
from transformers import Qwen2Config, Qwen2ForCausalLM
root=pathlib.Path('/tmp/topd_opsd_source_audit/OPSD/opsd_trainer.py')
tree=ast.parse(root.read_text()); node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='generalized_jsd_loss'); node.decorator_list=[]
ns={'torch':torch,'F':F}; exec(compile(ast.Module(body=[node],type_ignores=[]),str(root),'exec'),ns); author=ns[node.name]
torch.manual_seed(42)
for beta in [0.,.3,1.]:
 for temp in [1.,1.7]:
  for clip in [None,.05]:
   s=torch.randn(2,3,7,requires_grad=True); t=torch.randn(2,3,7,requires_grad=True)
   a=generalized_jsd(s,t,beta,temp,clip).sum(); b=author(s,t.detach(),beta=beta,temperature=temp,token_clip=clip,reduction='sum')
   ga=torch.autograd.grad(a,(s,t),allow_unused=True,retain_graph=True); gb=torch.autograd.grad(b,s)[0]
   torch.testing.assert_close(a,b); torch.testing.assert_close(ga[0],gb); assert ga[1] is None
print('author FKL/RKL/JSD values+student gradients parity: 12 cases PASS; teacher grad None')
# Real tiny Qwen eager model; variable prompt lengths, response padding, observations.
class Tok:
 pad_token_id=0
 def encode(self,s,add_special_tokens=False): return [8,9]
tok=Tok(); prompts=torch.tensor([[0,0,2,3],[0,4,5,6]]); resp=torch.tensor([[10,11,12,0],[13,14,0,0]]); att=(resp!=0).long()
ids,tm,pos=teacher_inputs(tok,prompts,resp,att,[['Paris'],['Rome']])
assert pos.tolist()==[[3,4,5,5],[4,5,5,5]]
cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.,pad_token_id=0)
student=Qwen2ForCausalLM(cfg).eval(); teacher_model=Qwen2ForCausalLM(cfg).eval().requires_grad_(False); teacher_model.load_state_dict(student.state_dict())
fullids=torch.cat([prompts,resp],1); fullatt=(fullids!=0).long(); tensors=dict(prompts=prompts,responses=resp,input_ids=fullids,attention_mask=fullatt,position_ids=(fullatt.cumsum(-1)-1).clamp_min(0),loss_mask=torch.tensor([[1.,0.,1.,0.],[1.,1.,0.,0.]]))
teacher=types.SimpleNamespace(model=teacher_model,device=torch.device('cpu'),scored_tokens=0); settings=types.SimpleNamespace(beta=0.,temperature=1.,component_clip=.05,chunk_size=1,max_student_logits_gib=4.)
loss=opsd_loss(student,teacher,tensors,tok,[['Paris'],['Rome']],settings,global_token_denominator(tensors['loss_mask']))
with torch.no_grad():
 sl=student(input_ids=fullids,attention_mask=fullatt,position_ids=tensors['position_ids']).logits[:,-5:-1]; tl=teacher_model(input_ids=ids,attention_mask=tm).logits
 selected=tl.gather(1,pos[...,None].expand(-1,-1,32)); expected=(generalized_jsd(sl,selected)*tensors['loss_mask']).sum()/4
 torch.testing.assert_close(loss,expected)
loss.backward(); assert all(p.grad is None for p in teacher_model.parameters()); assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
print('real tiny Qwen positions/left+right padding/observation mask/chunk=1 fullforward parity PASS; loss',float(loss))
student.zero_grad(); tensors['loss_mask'].zero_(); zl=opsd_loss(student,teacher,tensors,tok,[['Paris'],['Rome']],settings,global_token_denominator(tensors['loss_mask'])); zl.backward(); assert zl.item()==0 and all(p.grad is None or p.grad.eq(0).all() for p in student.parameters()); print('zero global loss mask: finite zero loss+all zero gradients PASS')
# Show advertised allowed global micro < ranks turns into invalid local micro.
print('global micro/ranks -> worker floor local:',[(g,w,g//w) for g,w in [(1,2),(2,8),(8,8)]])
# Frozen teacher cannot track student update.
p=next(student.parameters()); before=p.detach().clone(); opt=torch.optim.AdamW(student.parameters(),lr=.1,weight_decay=.01); opt.step(); print('zero loss still AdamW parameter max delta',float((p-before).abs().max()))
