from pathlib import Path
import hashlib,json,re,ast
from omegaconf import OmegaConf
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006');LOCAL=Path('work/bapo-e1-300-freeze-scalability').resolve();RAW=Path('work/evidence/freeze-scalability-b711-independent-cpu-raw');S='b7119791b86951727cea36a1881e8a525c932494'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory=json.loads((RAW/'inventory.json').read_text());assert inventory['head']==S and inventory['status']==''
r=json.loads((RAW/'logs__cpu-gate.json').read_text());assert r==json.loads(Path('work/evidence/freeze-scalability-b711-actual-source-linux-record.json').read_text())
assert r['source_commit']==S and r['sys_prefix']==str(ROOT/'env')and r['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'
assert all(type(r[k])is int for k in ('passed','skipped','failed','exit_code'));assert (r['passed'],r['skipped'],r['failed'],r['exit_code'])==(395,0,0,0)
assert r['log_sha256']==sha(RAW/'logs__cpu-gate.log');assert r['source_marker_sha256']==sha(RAW/'logs__cpu-gate-source.txt')==sha(RAW/'source-expected.txt');assert (RAW/'logs__cpu-gate-source.txt').read_text()==S+'\n'
assert r['test_launcher_sha256']==sha('work/e1-300-freeze-scalability-20261006/deployment-tools/cpu-tests.py')
log=(RAW/'logs__cpu-gate.log').read_text();summaries=re.findall(r'(?m)^395 passed(?:,.*)?$',log);assert len(summaries)==1
critical={'verl','search_r1','verl.protocol','verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos','search_r1.baselines.distillation','search_r1.baselines.objectives','search_r1.baselines.full_state','search_r1.baselines.checkpoint_gate','search_r1.baselines.gradient_norm','search_r1.baselines.actor_metrics','search_r1.baselines.recovery'}
for label in ('before','after'):
 e=r['import_origins'][label];assert e['repo']==str(ROOT/'repo') and e['source_commit']==S;assert critical<=set(e['modules'])
 marker='CPU_IMPORT_ORIGINS_'+label.upper()+'_JSON=';events=[json.loads(x.split(marker,1)[1])for x in log.splitlines()if marker in x];assert events==[e]
 for name,d in e['modules'].items():
  local=LOCAL.joinpath(*name.split('.'));local=local.with_suffix('.py')if local.with_suffix('.py').is_file()else local/'__init__.py';rel=local.relative_to(LOCAL);assert d['path']==str(ROOT/'repo'/rel)and d['sha256']==sha(local)
 for name,d in e['functions'].items():
  module=d['function_module'];local=LOCAL.joinpath(*module.split('.')).with_suffix('.py');assert d['path']==d['co_filename']==str(ROOT/'repo'/local.relative_to(LOCAL))and d['sha256']==sha(local)
  dotted=name[len(module)+1:].split('.');tree=ast.parse(local.read_text());node=tree
  for q in dotted:node=next(x for x in node.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name==q)
  assert d['co_firstlineno']==node.lineno and d['qualname']=='.'.join(dotted)and re.fullmatch('[0-9a-f]{64}',d['code_sha256'])
b=r['import_origins']['before'];a=r['import_origins']['after'];assert b['functions']==a['functions'];assert all(a['modules'][name]==d for name,d in b['modules'].items())
sm='STABILITY_TEST_FILES_JSON=';st=[json.loads(x.split(sm,1)[1])for x in log.splitlines()if sm in x];assert len(st)==2 and st[0]==st[1]and all(sha(LOCAL/n)==h for n,h in st[0].items())
config=RAW/'repo__verl__trainer__config__ppo_bapo_e1_300.yaml';assert sha(config)==sha(LOCAL/'verl/trainer/config/ppo_bapo_e1_300.yaml')
c=OmegaConf.to_container(OmegaConf.load(config),resolve=False)
checks=dict(horizon=c['trainer']['total_training_steps']-1,total_steps=c['trainer']['total_training_steps'],warmup=c['actor_rollout_ref']['actor']['optim']['num_warmup_steps'],stage_size=c['baselines']['bapo']['stage_validation_size'],mini=c['actor_rollout_ref']['actor']['ppo_mini_batch_size'],micro=c['actor_rollout_ref']['actor']['ppo_micro_batch_size'],logprob=c['actor_rollout_ref']['rollout']['log_prob_micro_batch_size'],loss_reduction=c['actor_rollout_ref']['actor']['loss_reduction'])
assert checks==dict(horizon=300,total_steps=301,warmup=114,stage_size=64,mini=128,micro=16,logprob=32,loss_reduction='sequence_mean')
result=dict(status='PASS_INDEPENDENT_ACTUAL_SOURCE_LINUX_CPU',scope='ACTUAL_SOURCE_CPU_ONLY_NOT_GPU_HARDWARE_PERMISSION_OR_LAUNCH',source_commit=S,record_raw_sha256=sha(RAW/'logs__cpu-gate.json'),log_raw_sha256=sha(RAW/'logs__cpu-gate.log'),before_modules=len(b['modules']),after_modules=len(a['modules']),functions=len(b['functions']),critical_modules=len(critical),stability_tests=st[0],config_yaml_sha256=sha(config),config_checks=checks,passed=395,skipped=0,failed=0,wall_seconds=r['wall_seconds'])
Path('work/e1-300-freeze-scalability-20261006/actual-source-cpu-independent-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
