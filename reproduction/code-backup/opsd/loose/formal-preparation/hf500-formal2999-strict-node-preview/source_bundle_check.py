import hashlib,json,subprocess,os,re
from pathlib import Path
def unique(items):
 d={}
 for k,v in items:
  assert k not in d;d[k]=v
 return d
def check_manifest_pin(directory,environment):
 expected=environment.get('OPSD_STAGE_MANIFEST_SHA256','')
 assert re.fullmatch('[0-9a-f]{64}',expected), 'Reviewed external manifest SHA256 required'
 raw=(directory/'source-manifest.json').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==expected, 'Stage manifest changed from reviewed external pin'
 return json.loads(raw,object_pairs_hook=unique)
def check_sources(root,directory,manifest):
 assert set(manifest['production_math_sources'])=={'search_r1/baselines/actor.py','search_r1/baselines/distillation.py'}
 if manifest['stage']=='pilot_only_no_future_ids_or_picked':
  assert set(manifest['runtime_sources'])==set(['capacity_supervisor.py', 'checkpoint_reload.py', 'gpu_job_entry.sh', 'gpu_loss_checkpoint_equivalence.py', 'gpu_retrieval_bulk.py', 'gpu_teacher_probe-factor-dtype.py', 'http_retrieval_check.py', 'job.sh', 'microbatch_probe.py', 'model_preflight.py', 'resource_sample.py', 'retrieval_compare.py', 'semantic_device_gate.py', 'train_supervisor.py'])
  assert set(manifest['file_sources'])==set(manifest['runtime_sources'])|{'pilot_audit.py','prior_cleanup.py','source_bundle_check.py'}
 else:
  assert manifest['stage']=='hf500_only_actual_formal_no_future_reload'
  assert set(manifest['file_sources'])=={'reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py','formal_completion_check.py','training_evidence.py'}
 assert directory.name==manifest['directory_name']
 assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
 subprocess.run(['git','-C',str(root/'repo'),'diff','--quiet','HEAD'],check=True)
 for name,expected in manifest['production_math_sources'].items():
  assert name in {'search_r1/baselines/actor.py','search_r1/baselines/distillation.py'}
  assert hashlib.sha256((root/'repo'/name).read_bytes()).hexdigest()==expected
 for name,expected in manifest['file_sources'].items():
  assert Path(name).name==name
  p=directory/name;assert p.is_file() and p.resolve().is_relative_to(directory)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 print('CURRENT SOURCE IMMUTABLE STAGE BUNDLE VERIFIED',flush=True)
if __name__=='__main__':
 directory=Path(__file__).resolve().parent
 manifest=check_manifest_pin(directory,os.environ)
 check_sources(Path('/mnt/data/shaoshitong/search-r1-rl/opsd'),directory,manifest)
