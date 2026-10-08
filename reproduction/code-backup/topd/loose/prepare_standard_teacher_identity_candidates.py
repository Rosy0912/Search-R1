import pathlib,shutil,re,json,hashlib
W=pathlib.Path('work')
names={
'qwen25-32b-trained-eval64-candidate-v2':'qwen25-32b-trained-eval64-r1-v1',
 'topd-32b-native-score-gate-v7':'topd-32b-native-score-gate-r1-v1',
 'topd-32b-shared-owner-candidate-v3':'topd-32b-shared-owner-r1-v1',
 'topd-32b-pipeline-ipc-candidate-v2':'topd-32b-pipeline-ipc-r1-v1',
 'topd-shared32b-supervisor-v5':'topd-shared32b-supervisor-r1-v1'}
for a,b in names.items():
 if (W/b).exists():raise RuntimeError('Candidate exists; do not overwrite')
 shutil.copytree(W/a,W/b,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
for b in names.values():
 for p in (W/b).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.md','.json','.sbatch'):
   s=p.read_text()
   for a,d in names.items():s=s.replace(a,d)
   s=s.replace('SEARCH_G1_GRPO100_32B','SEARCH_R1_GRPO100_32B').replace('SEARCH_G1_GRPO','SEARCH_R1_GRPO').replace('G1_100_EXPORT_BOUND_FOR_SAME64_UNREVIEWED_QUALITY','SEARCH_R1_100_EXPORT_BOUND_FOR_SAME64_UNREVIEWED_QUALITY').replace('ACTUAL_G1_100_OUTER_UNREVIEWED_NOT_TOPD_ACCEPTANCE','ACTUAL_SEARCH_R1_GRPO100_UNREVIEWED_NOT_TEACHER_NOT_TOPD').replace('4e7448d66a78996467134441571fda0a6844d180','0d8a9214dc6e19f10e21549b56f6c01b45d67100')
   p.write_text(s)
# Old IPC readiness canonical gate now agrees with all other components.
p=W/names['topd-32b-pipeline-ipc-candidate-v2']/'topd32b_shared_pipeline/readiness.py';s=p.read_text().replace('topd-32b-native-score-gate-v2','topd-32b-native-score-gate-r1-v1');p.write_text(s)
p=W/names['qwen25-32b-trained-eval64-candidate-v2']/'trained_assets.py';s=p.read_text();needle="result=read_ref(binding['train_result'],root);outer=read_ref(result['actual_outer_evidence'],root);check_outer(outer,model)";replacement=needle+"\n from training_identity import check_standard100\n cfg_ref=binding['training_effective_config'];cfg=read_ref(cfg_ref,root)\n need(cfg_ref['sha256']==result['effective_config_sha256'] and pathlib.Path(cfg_ref['path']).resolve()==pathlib.Path(binding['train_result']['path']).resolve().parent/'effective-runtime-config.json','Actual standard config artifact differs')\n check_standard100(training,result,outer,cfg,str(model))";assert needle in s;s=s.replace(needle,replacement);p.write_text(s)
