import copy,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tests/baselines'))
from test_models_and_driver import TinyModels
from search_r1.baselines import distillation as new
old_source=subprocess.check_output(['git','show','1620c2edb13809d19cc3f5af4a298ccd75488104:search_r1/baselines/distillation.py'],text=True)
ns={'__name__':'search_r1.baselines.review_old_distillation','__package__':'search_r1.baselines'}
exec(compile(old_source,'frozen_pre_instrumentation','exec'),ns)
TinyModels.setUpClass()
try:
 from transformers import Qwen2ForCausalLM
 tok=TinyModels.tokenizer
 baseline=Qwen2ForCausalLM.from_pretrained(TinyModels.tmp.name).eval()
 instrumented=copy.deepcopy(baseline)
 teacher_old=ns['FrozenTeacher'](TinyModels.tmp.name,tok)
 teacher_new=new.FrozenTeacher(TinyModels.tmp.name,tok)
 prompt=torch.tensor([tok.encode('Question capital?')]); response=torch.tensor([tok.encode('Paris Rome Berlin')])
 ids=torch.cat((prompt,response),1); mask=torch.ones_like(response); mask[:,1::2]=0
 tensors=dict(prompts=prompt,responses=response,input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(ids.shape[1])[None,:],loss_mask=mask)
 settings=SimpleNamespace(chunk_size=1,beta=0.,temperature=1.,component_clip=.05,max_student_logits_gib=4.)
 a=ns['opsd_loss'](baseline,teacher_old,tensors,tok,[['Paris']],settings,mask.sum())
 b=new.opsd_loss(instrumented,teacher_new,tensors,tok,[['Paris']],settings,mask.sum())
 assert torch.equal(a,b)
 a.backward();b.backward()
 count=0
 for p,q in zip(baseline.parameters(),instrumented.parameters()):
  if p.grad is not None:
   assert torch.equal(p.grad,q.grad); count+=1
 assert all(p.grad is None for p in teacher_new.model.parameters())
 costs=teacher_new.cost_metrics()
 assert costs['teacher/model_bytes_per_rank']==sum(p.numel()*p.element_size() for p in teacher_new.model.parameters())+sum(p.numel()*p.element_size() for p in teacher_new.model.buffers())
 before=(teacher_new.cpu_seconds,teacher_new.wall_seconds)
 with patch.object(new.time,'perf_counter',side_effect=[10.,12.]),patch.object(new.time,'process_time',side_effect=[20.,23.]):
  with teacher_new.measure(): pass
 assert teacher_new.wall_seconds-before[1]==2.
 assert teacher_new.cpu_seconds-before[0]==3.
 # Fixture/RSS unit proof uses synthetic process high-water data only.
 with patch.object(new.resource,'getrusage',return_value=SimpleNamespace(ru_maxrss=123)),patch.object(new.sys,'platform','darwin'):
  assert teacher_new.cost_metrics()['teacher_worker/peak_rss_bytes_per_rank']==123
 with patch.object(new.resource,'getrusage',return_value=SimpleNamespace(ru_maxrss=123)),patch.object(new.sys,'platform','linux'):
  assert teacher_new.cost_metrics()['teacher_worker/peak_rss_bytes_per_rank']==123*1024
 print(json.dumps({'loss_bitwise_equal':True,'loss':float(a.detach()),'gradient_tensors_bitwise_equal':count,'teacher_has_no_gradient':True,'cost_metrics_before_synthetic_clock':costs,'clock_delta_accounting_passed':True,'darwin_linux_rss_units_passed':True,'scope':'CPU tiny; not 3B or FSDP or isolated teacher memory attribution'},indent=2))
finally:TinyModels.tearDownClass()
