import ast, pathlib, typing
from collections import Counter
import re, string
import torch
from search_r1.baselines.bapo import *
from search_r1.baselines.advantages import grouped_center
source=ast.parse(pathlib.Path('/tmp/bapo-audit-20261002/re_search_faith.py').read_text())
selected=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('normalize_answer','get_f1_score')],type_ignores=[])
ns=dict(re=re,string=string,Counter=Counter,Union=typing.Union,List=typing.List)
exec(compile(selected,'author-scorer','exec'),ns)
for pred,gt in [('yes indeed','yes'),('no answer','no'),('the','a')]:
 print('F1',repr(pred),repr(gt),'local=',token_f1(pred,gt),'author=',ns['get_f1_score'](pred,gt))
answers=['yes indeed','I do not know','wrong','wrong','wrong']; targets=[['yes']]*5
print('partial-yes-gate=',boundary_scores(answers,targets,['q']*5,BoundaryStage(early=False)))
print('partial-yes-resample=',resample_rows(['yes indeed']+['wrong']*4,targets,['q']*5,BoundaryStage(early=False)))
stage=BoundaryStage()
for step in range(50,351,50):
 stage.observe(.5)
 print('validation-phase',step,stage.early,stage.stalled)
print('uid-collision=',boundary_scores(['I do not know','wrong','Paris','Paris'],[['Mars'],['Mars'],['Paris'],['Paris']],[0,0,0,0],BoundaryStage(early=False)))
print('distinct-question=',boundary_scores(['I do not know','wrong','Paris','Paris'],[['Mars'],['Mars'],['Paris'],['Paris']],['a','a','b','b'],BoundaryStage(early=False)))
print('permutation=',grouped_center(torch.tensor([0.,1.,0.,.5]),['a','a','b','b']).tolist(),grouped_center(torch.tensor([.5,0.,1.,0.]),['b','a','a','b']).tolist())
from verl.trainer.ppo.core_algos import compute_policy_loss
z=torch.zeros(2,3)
print('zero-mask-PPO=',[float(x) for x in compute_policy_loss(z,z,z,z,.2)])
a=['I do not know']+['wrong']*4+['Paris']*5
t=[['Mars']]*5+[['Paris']]*5
print('uid-collision-10=',boundary_scores(a,t,[0]*10,BoundaryStage(early=False)))
print('distinct-question-10=',boundary_scores(a,t,['a']*5+['b']*5,BoundaryStage(early=False)))
