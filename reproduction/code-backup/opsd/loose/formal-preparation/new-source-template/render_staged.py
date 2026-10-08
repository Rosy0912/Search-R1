"""Three immutable local preview stages. No remote I/O, submission, or invented acceptance."""
import hashlib,json,re,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from verify_formal_prerequisites import loads,RUNTIME_NAMES,COMMIT as OLD

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(s,target,value):
 assert s.count(target)==1, ('Expected exactly one critical template target',target,s.count(target))
 return s.replace(target,value,1)
def guard(s):return replace_once(s,'set -euo pipefail','set -euo pipefail\necho "STAGED PREVIEW: independent review required" >&2\nexit 78')
def fresh_output(output):
 output=Path(output);assert not output.exists(),'Never overwrite immutable stage';output.mkdir(parents=True);return output
def source_profile(p):
 p=loads(Path(p).read_text());commit=p['source_commit']
 assert re.fullmatch('[0-9a-f]{40}',commit) and commit!=OLD
 assert set(p['production_math_sources'])=={'search_r1/baselines/actor.py','search_r1/baselines/distillation.py'}
 assert all(re.fullmatch('[0-9a-f]{64}',v) for v in p['production_math_sources'].values())
 return p
def checker():return '''import hashlib,json,subprocess,os,re
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
  assert manifest['stage']=='hf_only_actual_pilot_no_future_reload'
  assert set(manifest['file_sources'])=={'reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py'}
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
'''
def finish(out,p,directory,stage,metadata):
 (out/'source_bundle_check.py').write_text(checker())
 manifest=dict(source_commit=p['source_commit'],production_math_sources=p['production_math_sources'],directory_name=directory,stage=stage,preview_blocked=True,file_sources={x.name:sha(x) for x in out.iterdir() if x.is_file()},**metadata)
 (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def pilot(profile,output):
 p=source_profile(profile);snap=Path(p['runtime_snapshot_directory'])
 assert snap.is_absolute() and set(p['runtime_sources'])==RUNTIME_NAMES
 assert all(sha(snap/n)==v for n,v in p['runtime_sources'].items())
 assert p['runtime_sources']['gpu_teacher_probe-factor-dtype.py']=='e98dbb92850d4e5b89a17209e209f1c7bcbc9dcf3712981d2b1b6e98ddf044dc'
 prior=p['completed_prior_jobs'];assert prior and all(re.fullmatch('[0-9]+',j) for j in prior)
 out=fresh_output(output);directory='pilot-runtime-'+p['source_commit'][:12]
 for name in RUNTIME_NAMES:
  text=(snap/name).read_text()
  if name!='gpu_teacher_probe-factor-dtype.py':text=text.replace('tools/runtime/',f'tools/{directory}/').replace(OLD,p['source_commit'])
  if name=='gpu_job_entry.sh':
   text=replace_once(text,': "${OPSD_EXPECTED_COMMIT:?Pinned source commit required}"',f'export OPSD_EXPECTED_COMMIT={p["source_commit"]}')
   text=replace_once(text,'exec python -m',f'python "$root/tools/{directory}/source_bundle_check.py"\nexec python -m');text=guard(text)
  if name=='job.sh':
   text=replace_once(text,'opsd-acceptance-$SLURM_JOB_ID','opsd-new-acceptance-$SLURM_JOB_ID')
   text=replace_once(text,'# Slurm owns cleanup',f'python "$root/tools/{directory}/prior_cleanup.py"\n# Slurm owns cleanup')
  if name=='train_supervisor.py':
   text=replace_once(text,'local_micro in (1,2,4,8)','local_micro in (2,4,8)')
   text=replace_once(text,'opsd-gpu-acceptance-5','opsd-new-gpu-acceptance-5')
   text=replace_once(text,'raise SystemExit(result.returncode)',"if result.returncode:raise SystemExit(result.returncode)\nfrom pilot_audit import verify_outers,verify_pilot_validations\nraw=(out.parent/'training.log').read_text()\nverify_outers(raw,5);verify_pilot_validations(raw)\n(out/'actual-fresh5-audit.json').write_text(json.dumps(dict(source_commit=os.environ['OPSD_EXPECTED_COMMIT'],job_id=os.environ['SLURM_JOB_ID'],actual_outer_updates=5,picked_local_micro=local_micro,picked_global_micro=8*local_micro),indent=2))")
  (out/name).write_text(text)
 raw=(HERE.parent/'verify_formal_prerequisites.py').read_text();audit=raw[raw.index('def pairs('):raw.index('def verify_reload(')]
 (out/'pilot_audit.py').write_text('import hashlib,json,math,re\nfrom pathlib import Path\n'+audit)
 cleanup=(HERE.parents[1]/'checkpoint-eval-preparation/cleanup.py').read_text()
 cleanup=replace_once(cleanup,"r'job_(?:2960|2962)(?:/|$)'", "r'job_(?:"+'|'.join(prior)+")(?:/|$)'")
 cleanup=replace_once(cleanup,"['2960','2962']",repr(prior))
 # Guard cannot accidentally include itself in caller-provided old-job list.
 cleanup=replace_once(cleanup,"job=os.environ['SLURM_JOB_ID'];",f"job=os.environ['SLURM_JOB_ID'];assert job not in {prior!r};")
 (out/'prior_cleanup.py').write_text(cleanup)
 return finish(out,p,directory,'pilot_only_no_future_ids_or_picked',dict(runtime_sources={n:sha(out/n) for n in RUNTIME_NAMES},pilot_run_pattern='opsd-new-acceptance-${SLURM_JOB_ID}',checkpoint_experiment='opsd-new-gpu-acceptance-5'))

def hf(profile,output):
 p=source_profile(profile);job=p['pilot_job'];assert re.fullmatch('[0-9]+',job)
 manifest=loads(Path(p['pilot_runtime_manifest']).read_text());assert manifest['source_commit']==p['source_commit'] and manifest['stage']=='pilot_only_no_future_ids_or_picked'
 assert manifest['production_math_sources']==p['production_math_sources']
 pilotdir=Path(p['pilot_runtime_manifest']).parent
 for name,expected in manifest['file_sources'].items():assert sha(pilotdir/name)==expected
 out=fresh_output(output);directory='hf-runtime-'+p['source_commit'][:12]+'-pilot-'+job
 runtime=manifest['directory_name'];wrapper=Path(p['runtime_snapshot_directory'])/'reload_job.sh'
 assert sha(wrapper)==p['reload_wrapper_sha256']
 text=wrapper.read_text()
 text=text.replace(OLD,p['source_commit']).replace('opsd-acceptance-2960',f'opsd-new-acceptance-{job}').replace('opsd-gpu-acceptance-5',manifest['checkpoint_experiment']).replace('2960',job)
 text=replace_once(text,'opsd-hf-reload-$SLURM_JOB_ID','opsd-new-hf-reload-$SLURM_JOB_ID')
 text=replace_once(text,'tools/runtime/checkpoint_reload.py',f'tools/{directory}/checkpoint_reload.py')
 text=replace_once(text,'tools/runtime/resource_sample.py',f'tools/{directory}/resource_sample.py')
 text=replace_once(text,'cd "$root/repo"',f'cd "$root/repo"\npython "$root/tools/{directory}/source_bundle_check.py"')
 (out/'reload_job.sh').write_text(guard(text))
 for name in ('checkpoint_reload.py','resource_sample.py'):shutil.copy2(pilotdir/name,out/name)
 return finish(out,p,directory,'hf_only_actual_pilot_no_future_reload',dict(pilot_job=job,pilot_runtime_manifest_sha256=sha(Path(p['pilot_runtime_manifest'])),pilot_runtime_directory_name=runtime))

if __name__=='__main__':
 assert len(sys.argv)==4,'pilot|hf profile.json NEW_LOCAL_DIR'
 assert sys.argv[1] in ('pilot','hf')
 (pilot if sys.argv[1]=='pilot' else hf)(sys.argv[2],sys.argv[3])
