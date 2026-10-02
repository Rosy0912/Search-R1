import sys, importlib.util, types, json
from pathlib import Path
import torch
from types import SimpleNamespace
from unittest.mock import patch
root=Path(__file__).resolve().parents[1]/'cosearch'
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tests/baselines'))
from test_models_and_driver import TinyModels, config
from search_r1.baselines.cosearch import GenerativeRanker, contains_answer
from omegaconf import open_dict
author=Path(__file__).resolve().parents[1]/'cosearch-author-review/verl/verl/tools/utils/answer_match_reward.py'
sys.modules['pytrec_eval']=types.ModuleType('pytrec_eval')
spec=importlib.util.spec_from_file_location('author_match',author);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
pairs=[('I live in Hague',['The Hague']),('THE Hague.',['The Hague']),('foobar',['foo-bar']),('foo-bar',['foo-bar']),('A cafe\u0301 in Paris',['CAFÉ']),('Roman',['Rome']),('a',['']),('yes',['yes'])]
print(json.dumps({'matcher':[[d,a,contains_answer(d,a),m.has_answer_string(a,d,m.SimpleTokenizer())] for d,a in pairs]}))
TinyModels.setUpClass()
try:
 c=config('cosearch').baselines.ranker;c.model_path=TinyModels.tmp.name
 with open_dict(c):c.weight_decay=0.
 r=GenerativeRanker(c);r.question_uids=['q']*3
 docs=[{'document':{'contents':d}} for d in ['Paris','Rome','Berlin']]
 def generated(input_ids,**kw):return torch.cat([input_ids,torch.tensor([r.tokenizer.encode('<ranking>[1,2,3]</ranking>')])],1)
 with patch.object(r.model,'generate',side_effect=generated):
  for i in range(3):r.rank('Question capital?','capital',docs,SimpleNamespace(source=i,trajectory=str(i)))
 before=[p.detach().clone() for p in r.model.parameters()]
 metrics=r.update({str(i):(float(i==0),['Paris'],True) for i in range(3)})
 grad=(sum(float(p.grad.double().square().sum()) for p in r.model.parameters() if p.grad is not None))**.5
 delta=sum(float((p.detach()-b).double().abs().sum()) for p,b in zip(r.model.parameters(),before))
 print(json.dumps({'identical_response_weight_decay_zero':{'grad_l2':grad,'parameter_delta_l1':delta,'events':metrics['ranker/training_events']}}))
 r=GenerativeRanker(c);r.question_uids=['q']*3
 outputs=iter(['<ranking>[1,2,3]</ranking>','<ranking>[2,1,3]</ranking>','<ranking>[3,2,1]</ranking>'])
 def varying(input_ids,**kw):return torch.cat([input_ids,torch.tensor([r.tokenizer.encode(next(outputs))])],1)
 with patch.object(r.model,'generate',side_effect=varying):
  for i in range(3):r.rank('Question capital?','capital',docs,SimpleNamespace(source=i,trajectory=str(i)))
 before=[p.detach().clone() for p in r.model.parameters()]
 metrics=r.update({str(i):(0.,['Paris'],True) for i in range(3)})
 grad=(sum(float(p.grad.double().square().sum()) for p in r.model.parameters() if p.grad is not None))**.5
 delta=sum(float((p.detach()-b).double().abs().sum()) for p,b in zip(r.model.parameters(),before))
 print(json.dumps({'different_response_weight_decay_zero':{'grad_l2':grad,'parameter_delta_l1':delta,'events':metrics['ranker/training_events']}}))
finally:TinyModels.tearDownClass()
