import sys,importlib.util,copy,shutil,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch
chat=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2');sys.path.insert(0,str(chat/'work/tree-diagnostic-development'))
from tests.baselines.test_algorithms import manager,batch
from search_r1.baselines.controller import Controller
from search_r1.baselines.objectives import attach_policy_weights
sp=importlib.util.spec_from_file_location('validator',chat/'work/validate_tree_stage.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
results=[]
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);run=root/'real-cpu-fixture';run.mkdir();directory=run/'output/tree-audit';settings=SimpleNamespace(method='tree_grpo',seed=42,context_alignment='report',base_commit='base',algorithm_source_commit='algo',gigpo=SimpleNamespace(anchor_mode='observation'),bapo=SimpleNamespace(patience=5),tree=SimpleNamespace(rounds=1,expansions=2,leaves=3));trainer=SimpleNamespace(global_steps=1,config=SimpleNamespace(baselines=settings));controller=Controller(trainer)
 with patch.dict(__import__('os').environ,{'SEARCH_R1_TREE_AUDIT_DIR':str(directory),'SEARCH_R1_BASELINE_OUTPUT':str(run/'output')}):
  for step in (1,2):
   trainer.global_steps=step;obj,_,_=manager(gpus=8);data,sources=controller.generate(obj,batch(obj.tokenizer,4),SimpleNamespace(non_tensor_batch={'index':np.array(['q1','q1','q2','q2'],dtype=object)}),{});width=data.batch['responses'].shape[1];data.batch['loss_mask']=data.batch['info_mask'][:,-width:];data.batch['old_log_probs']=torch.zeros_like(data.batch['loss_mask']).float();data.batch['advantages']=torch.ones_like(data.batch['loss_mask']).float();data.non_tensor_batch['uid']=np.array(['q1' if x<2 else 'q2' for x in sources],dtype=object);attach_policy_weights(data,'tree_grpo');data.reorder(torch.arange(len(data)-1,-1,-1));controller.audit_actor_batch(data)
 cfg={'baselines':{'base_commit':'base','algorithm_source_commit':'algo'}};assert len(v.inspect_audits(run,2,cfg))==2;results.append({'case':'real_cpu_rollout_policy_two_steps','passed':True,'fixture_only':True})
 for kind in ['missing','failed','extra','binding','trajectory','mass','group','branch','step','algorithm','pending','five_from_two','missing_directory']:
  case=root/kind;shutil.copytree(run,case);d=case/'output/tree-audit';rp=d/'step_1_rollout_1.json';pp=d/'step_1_rollout_1_policy_batch.json';r=json.loads(rp.read_text());p=json.loads(pp.read_text())
  if kind=='missing':rp.unlink()
  if kind=='failed':r['status']='FAILED'
  if kind=='extra':(d/'extra.json').write_text('{}')
  if kind=='binding':p['rollout_file_sha256']='0'*64
  if kind=='trajectory':p['rows'][0]['trajectory_hash']='0'*64
  if kind=='mass':p['rows'][0]['policy_weight_sum']=0.5
  if kind=='group':r['question_groups'][0]['leaves']=5
  if kind=='branch':r['branches'][0]['inherited']=False
  if kind=='step':p['global_step']=5
  if kind=='algorithm':r['algorithm_source_commit']='wrong'
  if kind=='pending':(d/'a.pending.99').write_text('{}')
  if kind=='missing_directory':shutil.rmtree(d)
  if kind in ('failed','group','branch','algorithm'):rp.write_text(json.dumps(r));p['rollout_file_sha256']=v.digest(rp)
  if kind not in ('missing_directory',):pp.write_text(json.dumps(p))
  try:v.inspect_audits(case,5 if kind=='five_from_two' else 2,cfg);raise RuntimeError('invalid audit passed '+kind)
  except (AssertionError,KeyError):results.append({'case':kind,'rejected':True,'fixture_only':True})
(chat/'work/aux-gate-review/stage-audit-validator-recheck.json').write_text(json.dumps(results,indent=2));print(len(results),'fixture checks passed, no real stage accepted')
