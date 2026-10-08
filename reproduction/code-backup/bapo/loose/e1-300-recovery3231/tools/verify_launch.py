import pathlib,json,hashlib,subprocess,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_cpu_source_marker(root, source, record):
 marker=root/'logs/cpu-gate-source.txt'
 assert marker.is_file()and not marker.is_symlink()and marker.resolve()==marker
 assert marker.read_text().strip()==source=='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
 assert record['source_commit']==source and record['source_marker_sha256']==sha(marker)
 return sha(marker)
def validate_import_origins(repo,source,proof):
 critical={'verl','search_r1','verl.protocol','verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos','search_r1.baselines.distillation','search_r1.baselines.objectives','search_r1.baselines.full_state','search_r1.baselines.checkpoint_gate','search_r1.baselines.gradient_norm','search_r1.baselines.actor_metrics','search_r1.baselines.recovery'}
 for label in ('before','after'):
  event=proof[label];assert event['repo']==str(repo)and event['source_commit']==source
  modules=event['modules'];assert critical<=set(modules)
  for name,item in modules.items():
   assert name in ('verl','search_r1')or name.startswith(('verl.','search_r1.'))
   expected=repo.joinpath(*name.split('.'));file=expected.with_suffix('.py')
   if not file.is_file():file=expected/'__init__.py'
   p=pathlib.Path(item['path']);assert p==file and p.resolve()==p and p.is_file()and repo in p.parents and sha(p)==item['sha256']
  expected_functions={
   'verl.workers.actor.dp_actor':('DataParallelPPOActor.update_policy','DataParallelPPOActor.compute_log_prob','DataParallelPPOActor._optimizer_step'),
   'verl.trainer.ppo.core_algos':('kl_penalty','compute_policy_loss'),
   'search_r1.baselines.distillation':('global_sum_count','global_token_count','global_token_denominator'),
   'search_r1.baselines.objectives':('row_denominator','weighted_policy_loss'),
   'search_r1.baselines.actor_metrics':('validate_actor_metric_topology','merge_actor_metric_records','aggregate_sequence_mean_actor_metrics'),
   'search_r1.baselines.full_state':('state_digest','atomic_save'),
   'search_r1.baselines.checkpoint_gate':('validate_release',),
 'search_r1.baselines.recovery':('validate_manifest','validate_native','validate_driver'),}
  functions=event['functions'];assert set(functions)=={module+'.'+name for module,names in expected_functions.items()for name in names}
  for module,names in expected_functions.items():
   expected=repo.joinpath(*module.split('.')).with_suffix('.py')
   for name in names:
    item=functions[module+'.'+name];assert pathlib.Path(item['co_filename']).resolve()==expected==pathlib.Path(item['path'])
    assert item['function_module']==module and type(item['co_firstlineno'])is int and item['co_firstlineno']>0
    assert sha(expected)==item['sha256']and len(item['code_sha256'])==64
 assert proof['after']['functions']==proof['before']['functions']
 assert all(proof['after']['modules'].get(name)==value for name,value in proof['before']['modules'].items())
phase=sys.argv[1];assert phase in ('hardware','formal300')
assert root.is_dir()and not root.is_symlink()and (root/'tools').is_dir()and not (root/'tools').is_symlink()
source=(root/'source-expected.txt').read_text().strip();assert source=='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e';r=json.loads((root/'logs/launch-approval.json').read_text())
assert r['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'and r['source_commit']==source and r['root']==str(root)
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True)
assert sha(root/'logs/tools-manifest.json')==r['tools_manifest_sha256']
m=json.loads((root/'logs/tools-manifest.json').read_text());assert m['source_commit']==source
files=[p for p in (root/'tools').iterdir()if p.is_file()]
assert type(r.get('tools_file_count'))is int and r['tools_file_count']>=22 and len(files)==r['tools_file_count']
assert all(not p.is_symlink()and p.resolve().parent==root/'tools'for p in files)
actual={p.name:sha(p)for p in files};assert actual==m['files']
assert sha(root/'repo/verl/trainer/config/ppo_bapo_e1_300.yaml')==r['config_yaml_sha256']
assert sha(root/'common-data/manifest.json')==r['data_manifest_sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert sha(root/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json')==r['model_manifest_sha256']
assert sha(root/'logs/cpu-gate.json')==r['cpu_gate_record_sha256']
c=json.loads((root/'logs/cpu-gate.json').read_text());assert c['source_commit']==source and c['exit_code']==0 and c['failed']==0 and type(r['cpu_expected_passed'])is int and r['cpu_expected_passed']>=327 and c['passed']==r['cpu_expected_passed']
validate_cpu_source_marker(root,source,c)
validate_import_origins(root/'repo',source,c['import_origins'])
assert c['skipped']==0 and c['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'
assert c['sys_prefix']==str(root/'env')
assert c['test_launcher_sha256']==sha(root/'tools/cpu-tests.py')
assert sha(root/'logs/cpu-gate.log')==c['log_sha256']
text=(root/'logs/cpu-gate.log').read_text()
for label,marker in [('before','CPU_IMPORT_ORIGINS_BEFORE_JSON='),('after','CPU_IMPORT_ORIGINS_AFTER_JSON=')]:
 events=[json.loads(line.split(marker,1)[1])for line in text.splitlines()if marker in line]
 assert len(events)==1 and events[0]==c['import_origins'][label]
stability_files=('tests/baselines/test_actor_metrics.py','tests/baselines/test_bapo_stability_patch.py','tests/baselines/test_e1_checkpoint_gate.py')
file_events=[json.loads(line.split('STABILITY_TEST_FILES_JSON=',1)[1])for line in text.splitlines()if 'STABILITY_TEST_FILES_JSON='in line]
assert len(file_events)==2 and file_events[0]==file_events[1]=={name:sha(root/'repo'/name)for name in stability_files}
from omegaconf import OmegaConf
configuration=OmegaConf.load(root/'repo/verl/trainer/config/ppo_bapo_e1_300.yaml')
assert configuration.actor_rollout_ref.actor.optim.num_warmup_steps==114
assert configuration.actor_rollout_ref.actor.loss_reduction=='sequence_mean'
assert configuration.actor_rollout_ref.actor.ppo_mini_batch_size==128 and configuration.actor_rollout_ref.actor.ppo_micro_batch_size==16
review=pathlib.Path(r['independent_review']['path']);assert review.is_file()and root in review.resolve().parents and sha(review)==r['independent_review']['sha256']
print('NEW_INSTRUCT300_FIXED_LAUNCH_AND_ACTUAL_CPU_GATE_PASS',phase,flush=True)
