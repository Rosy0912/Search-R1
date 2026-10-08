import ast,copy,json,hashlib,pathlib,tempfile,types,os,inspect,sys,subprocess
from unittest.mock import patch
ROOT=pathlib.Path.cwd();TOOLS=ROOT/'work/tree-fixed-formal-296217a';OUT=ROOT/'work/tree-fixed-formal-review';REMOTE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((TOOLS/'manifest.json').read_text())
def require(v,msg):
 if not v:raise AssertionError(msg)
tree=ast.parse((TOOLS/'supervisor.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Expr) and 'sha(a.actual_five_review)' in ast.unparse(n));end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lock' for t in n.targets));code=compile(ast.Module(body=main.body[start:end],type_ignores=[]),str(TOOLS/'supervisor.py'),'exec');receipt=json.loads((TOOLS/'ACTUAL3103_RECEIPT.json').read_text())
def proof_sha(p):
 p=pathlib.Path(p)
 if str(p).startswith(str(REMOTE/'logs/tree-fixed-five-3103-review')+'/'):return sha(TOOLS/p.name)
 if str(p).startswith(str(REMOTE)+'/'):return sha(ROOT/'work/tree-fixed-five-review/actual3103-files'/p.relative_to(REMOTE))
 return sha(p)
mutations=[('valid',None,None),('status','status','FAILED'),('stage','stage','two'),('commit','candidate_commit','old'),('actual4','actual_outer_updates',4),('science','scientific_five_accepted',False),('operations','operational_five_accepted',False),('HF','hf5_full_cpu_reload',False),('terminal','original_slurm_exit_code','1:0'),('body','training_body_exit_code',1),('entry','entry_config_verified',False),('cleanup','cleanup_residual',{'12':'R'}),('ranks','direct_rank_outer_records',39),('tree','tree_audit_files',9),('profiles','actual_memory_profile_records',39)]
results=[]
with tempfile.TemporaryDirectory() as d:
 for label,key,value in mutations:
  x=copy.deepcopy(receipt)
  if key:x[key]=value
  p=pathlib.Path(d)/'receipt.json';p.write_text(json.dumps(x));a=types.SimpleNamespace(actual_five_review=p,actual_five_review_sha=sha(p));error=None
  try:exec(code,dict(a=a,ROOT=REMOTE,COMMIT=m['code_commit'],Path=pathlib.Path,sha=proof_sha,require=require,json=json))
  except (AssertionError,FileNotFoundError) as e:error=str(e)
  results.append({'case':label,'error':error})
 for label,key in [('proofsha','evidence'),('reviewsha','small_review_bundle_mapping')]:
  x=copy.deepcopy(receipt);x[key][0]['sha256']='0'*64;p=pathlib.Path(d)/'receipt.json';p.write_text(json.dumps(x));a=types.SimpleNamespace(actual_five_review=p,actual_five_review_sha=sha(p));error=None
  try:exec(code,dict(a=a,ROOT=REMOTE,COMMIT=m['code_commit'],Path=pathlib.Path,sha=proof_sha,require=require,json=json))
  except (AssertionError,FileNotFoundError) as e:error=str(e)
  results.append({'case':label,'error':error})
(OUT/'FIVE_GATE_CASES.json').write_text(json.dumps({'source_sha256':sha(TOOLS/'supervisor.py'),'cases':results,'valid_case_passed':results[0]['error'] is None,'scope':'Exact AST priorfive block; fixture digest recomputed to reach semantic cases; heavy parts not executed'},indent=2));print('valid five prior:',results[0])
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configuration');ns=dict(CAND=ROOT/'work/tree-native-numeric-296217a',ROOT=REMOTE,require=require,CONTENT='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(TOOLS/'supervisor.py'),'exec'),ns)
with tempfile.TemporaryDirectory() as d,patch('search_r1.baselines.holdout.prepare_common_holdout',return_value={'content_identity':{'sha256':ns['CONTENT']},'holdout_rows':3393}),patch.dict(os.environ,{'SEARCH_R1_ASSETS':str(REMOTE/'assets-root'),'SEARCH_R1_BASELINE_OUTPUT':str(REMOTE/'runs/tree-fixed-formal-3110/output')}):
 c,steps=ns['configuration'](pathlib.Path(d),'formal');assert steps==500 and c.trainer.total_training_steps==501 and not c.trainer.smoke_only and c.trainer.val_before_train and c.trainer.test_freq==50 and c.data.val_batch_size==64
 from omegaconf import OmegaConf
 (OUT/'ACTUAL_CONFIGURATION.yaml').write_text(OmegaConf.to_yaml(c,resolve=True));budget={'actual':steps,'total':c.trainer.total_training_steps,'initial64':bool(c.trainer.val_before_train),'period':c.trainer.test_freq,'validation_batch':c.data.val_batch_size,'train':c.data.train_batch_size,'mini':c.actor_rollout_ref.actor.ppo_mini_batch_size,'micro':c.actor_rollout_ref.actor.ppo_micro_batch_size,'save_freq':c.trainer.save_freq,'base':str(c.actor_rollout_ref.model.path)}
import validation_cost_hooks_fixed_formal as hooks
class Trainer:pass
class Manager:pass
mod=types.ModuleType('verl.trainer.ppo.ray_trainer');mod.RayPPOTrainer=Trainer;manager=types.ModuleType('search_r1.llm_agent.generation');manager.LLMGenerationManager=Manager;trainer=ROOT/'work/tree-numeric-candidate/verl/trainer/ppo/ray_trainer.py';cases=[]
for name,job,out,valid in [('valid','3110',str(REMOTE/'runs/tree-fixed-formal-3110/output'),True),('wrongjob','3111',str(REMOTE/'runs/tree-fixed-formal-3110/output'),False),('oldfive','3110',str(REMOTE/'runs/tree-fixed-five-3110/output'),False),('oldformal','3110',str(REMOTE/'runs/tree-formal-3110/output'),False),('otherroot','3110','/tmp/tree-fixed-formal-3110/output',False),('escape','3110',str(REMOTE/'runs/tree-fixed-formal-3110/output/../../outside'),False),('zero','0',str(REMOTE/'runs/tree-fixed-formal-0/output'),False)]:
 calls=[]
 with patch.dict(sys.modules,{'verl.trainer.ppo.ray_trainer':mod,'search_r1.llm_agent.generation':manager}),patch.object(inspect,'getsourcefile',return_value=str(trainer)),patch.object(hooks,'wrap_trainer',side_effect=lambda *a,**k:calls.append(k)),patch.dict(os.environ,{'TREE_VALIDATION_TRAINER_SHA256':sha(trainer),'SLURM_JOB_ID':job,'SEARCH_R1_BASELINE_OUTPUT':out}):
  try:hooks.install();accepted=True
  except AssertionError:accepted=False
 assert accepted==valid and bool(calls)==valid
 if valid:assert calls[0]['final_expected_rows']==51713
 cases.append({'case':name,'accepted':accepted})
(OUT/'CONFIG_OBSERVER_CASES.json').write_text(json.dumps({'configuration':budget,'observer_cases':cases,'scope':'Actual Hydra configuration and hook.install, only assets/heavy Trainer+Manager/wrap mocked; no Slurm/GPU/remote side effects'},indent=2));print(budget)
