import ast,copy,json,hashlib,pathlib,tempfile,types,os,inspect,sys,subprocess,socket
from unittest.mock import patch
ROOT=pathlib.Path.cwd();TOOLS=ROOT/'work/tree-fixed-five-296217a';OUT=ROOT/'work/tree-fixed-five-review';REMOTE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((TOOLS/'manifest.json').read_text());checks={}
checks['manifest']=sha(TOOLS/'manifest.json')=='8f3fb72fc21037d4d629026be9db0cac00c24b00df21a45982d4a63a0fd6710b';checks['wrapper']=sha(ROOT/'work/start_tree_fixed_five_296217a.sh')=='b5f2932c507d75defd83b9b84e203e3634f0f0b1d94daf120377e9c59b0e624a';checks['afterany']=sha(ROOT/'work/check_fixed_five_afterany.py')=='a6a73cf43a659d55402d56005d975749a5766536ceec54bffa8e0f364f3b96d3'
for n,v in m['files'].items():checks['package:'+n]=sha(TOOLS/n)==v
for n,v in m['external_helpers'].items():checks['helper:'+n]=sha(ROOT/'work'/n)==v
checks['retriever']=sha(ROOT/'work/tree-grpo/reproduction/h20/retriever_local.py')==m['retriever_script_sha256']
for n in ['gpu-kl-proof.py','sitecustomize.py']:checks['unchanged:'+n]=sha(TOOLS/n)==sha(ROOT/'work/tree-fixed-two-296217a-v3'/n)
assert all(checks.values()),checks
for n in ['supervisor.py','validation_cost_entry.py','validation_cost_hooks_fixed_five.py','gpu-kl-proof.py','sitecustomize.py']:compile((TOOLS/n).read_text(),str(TOOLS/n),'exec')
for p in [ROOT/'work/start_tree_fixed_five_296217a.sh',TOOLS/'run_tree_with_validation_cost_fixed_short.sh']:subprocess.run(['bash','-n',str(p)],check=True)
def require(value,message):
 if not value:raise AssertionError(message)
# Execute exact new prior-check statements, using real locally captured original proof bytes.
tree=ast.parse((TOOLS/'supervisor.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');start=next(i for i,n in enumerate(main.body) if n.lineno==65);end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lock' for t in n.targets));prior_code=compile(ast.Module(body=main.body[start:end],type_ignores=[]),str(TOOLS/'supervisor.py'),'exec');receipt=json.loads((TOOLS/'ACTUAL3095_RECEIPT.json').read_text());prior_cases=[]
def proof_sha(p):
 p=pathlib.Path(p)
 if str(p).startswith(str(REMOTE/'logs/tree-fixed-two-3095-review')+'/'):return sha(TOOLS/p.name)
 if str(p).startswith(str(REMOTE)+'/'):return sha(ROOT/'work/tree-fixed-two-review/actual3095-files'/p.relative_to(REMOTE))
 return sha(p)
mutations=[('valid',None,None),('wrong_status','status','FAILED'),('wrong_stage','stage','formal'),('wrong_commit','candidate_commit','old'),('updates1','actual_outer_updates',1),('no_science','scientific_two_accepted',False),('no_operations','operational_two_accepted',False),('no_hf','hf2_full_cpu_reload',False),('failed_exit','original_slurm_exit_code','1:0'),('bodyfail','training_body_exit_code',1),('entryfail','entry_config_verified',False),('cleanup','cleanup_residual',{'12':'R'}),('rankmissing','direct_rank_outer_records',15),('treemissing','tree_audit_files',3),('profilemissing','actual_memory_profile_records',15)]
with tempfile.TemporaryDirectory() as d:
 for label,key,value in mutations:
  data=copy.deepcopy(receipt)
  if key:data[key]=value
  path=pathlib.Path(d)/'receipt.json';path.write_text(json.dumps(data));a=types.SimpleNamespace(actual_two_review=path,actual_two_review_sha=sha(path));rejected=False
  try:exec(prior_code,dict(a=a,ROOT=REMOTE,COMMIT=m['code_commit'],Path=pathlib.Path,sha=proof_sha,require=require,json=json))
  except (AssertionError,FileNotFoundError):rejected=True
  assert rejected==(label!='valid'),label
  prior_cases.append({'case':label,'rejected':rejected,'scope':'Semantic conditions isolated with fixture digest; actual wrapper independently requires original signed e79 hash'})
 for label,mutator,wrong_sha in [('proofsha',lambda data:data['evidence'][0].update(sha256='0'*64),False),('outsideproof',lambda data:data['evidence'][0].update(path=str(REMOTE/'runs/other-run/file.json')),False),('reviewsha',lambda data:data['small_review_bundle_mapping'][0].update(sha256='0'*64),False),('outsidereview',lambda data:data['small_review_bundle_mapping'][0].update(path=str(REMOTE/'logs/other/file.md')),False),('wrong_receipt_sha',lambda data:None,True)]:
  data=copy.deepcopy(receipt);mutator(data);path=pathlib.Path(d)/'receipt.json';path.write_text(json.dumps(data));a=types.SimpleNamespace(actual_two_review=path,actual_two_review_sha='0'*64 if wrong_sha else sha(path))
  try:exec(prior_code,dict(a=a,ROOT=REMOTE,COMMIT=m['code_commit'],Path=pathlib.Path,sha=proof_sha,require=require,json=json))
  except (AssertionError,FileNotFoundError):rejected=True
  else:raise AssertionError(label+' accepted')
  prior_cases.append({'case':label,'rejected':True})
# Budget uses actual configuration() body; only asset-reading holdout substituted.
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configuration');namespace=dict(CAND=ROOT/'work/tree-native-numeric-296217a',ROOT=REMOTE,require=require,CONTENT='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(TOOLS/'supervisor.py'),'exec'),namespace)
with tempfile.TemporaryDirectory() as d,patch('search_r1.baselines.holdout.prepare_common_holdout',return_value={'content_identity':{'sha256':namespace['CONTENT']},'holdout_rows':3393}),patch.dict(os.environ,{'SEARCH_R1_ASSETS':str(REMOTE/'assets-root'),'SEARCH_R1_BASELINE_OUTPUT':str(REMOTE/'runs/tree-fixed-five-3110/output')}):
 config,steps=namespace['configuration'](pathlib.Path(d),'five');assert steps==5 and config.trainer.total_training_steps==6 and config.trainer.smoke_only and not config.trainer.val_before_train
 budget={'actual_updates':steps,'total_steps':config.trainer.total_training_steps,'train':config.data.train_batch_size,'mini':config.actor_rollout_ref.actor.ppo_mini_batch_size,'micro':config.actor_rollout_ref.actor.ppo_micro_batch_size,'val_before':config.trainer.val_before_train,'base_model':str(config.actor_rollout_ref.model.path)}
# Actual fixed-five install; only heavy classes and wrap side effect mocked.
import validation_cost_hooks_fixed_five as hooks
class Trainer:pass
class Manager:pass
module=types.ModuleType('verl.trainer.ppo.ray_trainer');module.RayPPOTrainer=Trainer;manager=types.ModuleType('search_r1.llm_agent.generation');manager.LLMGenerationManager=Manager;trainer=ROOT/'work/tree-numeric-candidate/verl/trainer/ppo/ray_trainer.py';pathcases=[]
for label,job,output,valid in [('valid','3110',str(REMOTE/'runs/tree-fixed-five-3110/output'),True),('wrongjob','3111',str(REMOTE/'runs/tree-fixed-five-3110/output'),False),('oldtwo','3110',str(REMOTE/'runs/tree-fixed-two-3110/output'),False),('oldformal','3110',str(REMOTE/'runs/tree-formal-3110/output'),False),('otherroot','3110','/tmp/tree-fixed-five-3110/output',False),('escape','3110',str(REMOTE/'runs/tree-fixed-five-3110/output/../../outside'),False),('zero','0',str(REMOTE/'runs/tree-fixed-five-0/output'),False)]:
 calls=[]
 with patch.dict(sys.modules,{'verl.trainer.ppo.ray_trainer':module,'search_r1.llm_agent.generation':manager}),patch.object(inspect,'getsourcefile',return_value=str(trainer)),patch.object(hooks,'wrap_trainer',side_effect=lambda *a,**k:calls.append((a,k))),patch.dict(os.environ,{'TREE_VALIDATION_TRAINER_SHA256':sha(trainer),'SLURM_JOB_ID':job,'SEARCH_R1_BASELINE_OUTPUT':output}):
  try:hooks.install();accepted=True
  except AssertionError:accepted=False
 assert accepted==valid and bool(calls)==valid
 pathcases.append({'case':label,'accepted':accepted})
result={'status':'PASS_LOCAL_CPU_ONLY','pins':checks,'prior_cases':prior_cases,'prior_original_proof_count':len(receipt['evidence']),'prior_original_independent_small_reviews':len(receipt['small_review_bundle_mapping']),'actual_configuration_budget':budget,'observer_install_path_cases':pathcases,'scope':'Local actual source/AST with heavy assets/classes mocked; no remote writes/Slurm/GPU. Fixed wrapper cryptographically pins e79 signedtwo; direct supervisor CLI hash is caller-supplied and outside supported standalone launch scope.'}
(OUT/'CPU_GATE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pins':len(checks),'prior_cases':len(prior_cases),'path_cases':len(pathcases),'budget':budget}))
