import os,sys,pathlib,json,torch
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path('work/tree-recovery-runtime/source/repo').resolve()))
from tests.baselines.test_actual_update_budget import real_iteration_tail,FakeTrainer
results=[]
for calls in (10,9):
 t=FakeTrainer(300,301,save_freq=10,test_freq=10);t.config.baselines=SimpleNamespace(method='tree_grpo');t.actual_outer_updates=299
 t.config.trainer.recovery_min_train_reward=.1;t.config.trainer.recovery_min_f1=.34242827316810346
 t.actor_rollout_wg.update_actor=lambda batch:SimpleNamespace(meta_info={'metrics':{'actor/optimizer_steps':calls}})
 def validation():
  t.events.append(('validate',t.global_steps));return {'val/f1/all':.48918324738300495}
 t._validate=validation
 run=real_iteration_tail();run.__globals__['os']=os
 run.__globals__['object']=lambda:SimpleNamespace(batch={'token_level_rewards':torch.full((3840,1),.2)})
 with patch.dict(os.environ,{'TREE_RECOVERY_MANIFEST':'mock-parent'},clear=False):
  try:run(t)
  except RuntimeError as e:
   assert calls==9 and 'required10' in str(e);assert t.global_steps==300 and not any(x[0]=='save' for x in t.events)
   results.append({'case':'9calls','rejected':True,'scope':'Calls already happened before guard; no saved accepted update'})
  else:
   assert calls==10 and t.actual_outer_updates==300 and t.global_steps==301
   assert any(x[0]=='save' and x[1]==300 for x in t.events)
   assert ('validate',301) in t.events
   results.append({'case':'10calls','saved300':True,'final301':True,'no301optimizer':True})
pathlib.Path('work/tree-recovery-review/TERMINAL300_CPU_PROOF.json').write_text(json.dumps(dict(status='REAL_RECOVERY_DRIVER_TAIL_CPU_PASS',cases=results,limits='AST real trainer tail with fake RPC metrics and validation; not actual model results',fixture_namespace_note='Existing test lacks os in AST namespace; independent probe supplies actual os and fake batch for recovery branch'),indent=2))
print('TERMINAL300 CPU PASS')
