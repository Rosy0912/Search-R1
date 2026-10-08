"""CPU-only pre-generation verification; no future corrected evaluation terminal needed."""
import datetime,hashlib,re,subprocess,math
from pathlib import Path
from training_evidence import loads,digest,COMMIT,OWN
SOURCE_NAMES={'search_r1/baselines/corrected_evaluation.py','search_r1/baselines/evaluation.py','search_r1/baselines/bapo.py','search_r1/qa_protocol.py','verl/trainer/main_ppo.py','verl/utils/reward_score/qa_em.py','search_r1/baselines/main.py','verl/trainer/ppo/ray_trainer.py','search_r1/llm_agent/generation.py','search_r1/baselines/actor.py','search_r1/baselines/distillation.py','verl/workers/rollout/vllm_rollout/vllm_rollout.py','verl/utils/dataset/rl_dataset.py','search_r1/baselines/corrected_runtime_capture.py','verl/workers/fsdp_workers.py','verl/single_controller/base/worker.py','verl/single_controller/base/decorator.py','verl/single_controller/ray/base.py'}
PRESCRIPTION_FIELDS={'version','phase','training_commit','eval_commit','source_files','input_manifest_sha256','model_role','model_directory','model_files','generation_budget','generation_budget_sha256'}
HF_TEMPLATE_SHA='ff617647f4e986920c6c27867fdf60e70ae7b43b8f7adf8fda62441abfae5850'

def verify_hf_activation(manifest_path,script_path,checker_path,external_sha):
 """NEW reviewed final bundle derived only from exact v5 template activation."""
 folder=Path(manifest_path).parent;directory=Path(__file__).parent
 if digest(manifest_path)!=external_sha:raise ValueError('Actual HF final manifest external pin mismatch')
 actual=loads(Path(manifest_path).read_text());template=loads((directory/'source-manifest.json').read_text())
 if actual.get('prior_hf_preview_manifest_sha256')!=HF_TEMPLATE_SHA or actual.get('activation_template_manifest_sha256')!=digest(directory/'source-manifest.json'):raise ValueError('Actual HF manifest lacks exact v5/upstream ancestry')
 if actual.get('preview_blocked') is not False or actual.get('source_commit')!=COMMIT or actual.get('formal_job')!='3007':raise ValueError('HF final source/job/activation mismatch')
 if actual.get('directory_name')!=folder.name or not re.fullmatch('[a-z0-9-]+',folder.name):raise ValueError('HF actual final directory mismatch')
 for name in ('production_math_sources','stage','formal_manifest_sha256','original_hf_manifest_sha256'):
  if actual.get(name)!=template[name]:raise ValueError('HF final immutable protocol identity mismatch')
 if set(actual['file_sources'])!=set(template['file_sources']):raise ValueError('Actual HF final sidecar/helper set differs')
 for name,sha in actual['file_sources'].items():
  if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('Actual HF final helper bytes unbound')
  source=(directory/name).read_text()
  if name=='reload_job.sh':
   guard='echo "STAGED PREVIEW: independent review required" >&2\nexit 78\n'
   if source.count(guard)!=1:raise ValueError('Unknown HF template script guard')
   source=source.replace(guard,'').replace(template['directory_name'],folder.name)
  elif name=='hf_launcher.py':
   guard=" print('BLOCKED HF LAUNCHER CANDIDATE: requires independently reviewed NEW activated HF bundle',file=sys.stderr)\n return 78\n"
   if source.count(guard)!=1:raise ValueError('Unknown HF launcher template guard')
   source=source.replace(guard,'')
  if (folder/name).read_text()!=source:raise ValueError('Actual HF helper/script differs from exact v5 reviewed activation')
 if Path(checker_path).resolve()!=(folder/'checkpoint_reload.py').resolve() or digest(checker_path)!=template['file_sources']['checkpoint_reload.py']:raise ValueError('HF checker is not immutable5586 source')
 if Path(script_path).resolve()!=(folder/'reload_job.sh').resolve():raise ValueError('HF activated script role outside bundle')
 return actual

def utc(value):
 time=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
 if time.tzinfo is None or time.utcoffset()!=datetime.timedelta(0):raise ValueError('Resource timestamp must be actual UTC')
 return time

def verify_resource_rows(rows,job,selected_uuid,uid,checker_pid,checker_start):
 """Original resource_sample.py rows: strict ownership and at least real HF activity."""
 if not rows:raise ValueError('Missing actual resource samples')
 previous=None;seen_checker=False
 for row in rows:
  time=utc(row['time'])
  if previous is not None and time<=previous:raise ValueError('Resource times not strictly ordered')
  previous=time
  if row['job_id']!=job or row.get('query_error') or row['nominal_sleep_seconds']!=2:raise ValueError('Resource job/query mismatch')
  gpus=row['gpus'];ids=[g['uuid'] for g in gpus]
  if len(ids)!=len(set(ids)) or ids.count(selected_uuid)!=1:raise ValueError('Resource GPU UUID missing/duplicate')
  for gpu in gpus:
   if not all(isinstance(gpu[k],(int,float)) and math.isfinite(gpu[k]) and 0<=gpu[k] for k in ('util','used_mib','total_mib')) or gpu['util']>100 or gpu['total_mib']<=0 or gpu['used_mib']>gpu['total_mib']:raise ValueError('Invalid GPU resource values')
  processes=row['job_processes']
  known={}
  for process in processes:
   if process.get('unavailable') or process.get('in_slurm_job') is not True or process.get('uid')!=str(uid):raise ValueError('Resource job process not owned/readable')
   if process['pid'] in known or not str(process['process_start_ticks']).isdigit():raise ValueError('Resource PID/start identity invalid')
   known[process['pid']]=process
  for process in row['gpu_processes']:
   if process.get('uuid')!=selected_uuid:continue # Other-node GPUs are observed, outside this own1GPU lease.
   if process.get('unavailable') or process.get('in_slurm_job') is not True or process.get('uid')!=str(uid):raise ValueError('Foreign/unreadable GPU process')
   try:gpu_memory=float(process['gpu_used_mib'])
   except (TypeError,ValueError):raise ValueError('Invalid process GPU memory')
   if not math.isfinite(gpu_memory) or gpu_memory<0:raise ValueError('Nonfinite/negative process GPU memory')
   if process['uuid']!=selected_uuid or process['pid'] not in known or known[process['pid']]['process_start_ticks']!=process['process_start_ticks']:raise ValueError('GPU process outside own selected UUID/cgroup identity')
   if process['pid']==checker_pid and process['process_start_ticks']==str(checker_start):seen_checker=True
  # Validate finite JSON before entry; require cgroup actual ownjob memory readings.
  memory=row['cgroup']
  if not memory or any(not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',name) for name in memory):raise ValueError('Resource cgroup outside own Slurm job')
  for name,value in memory.items():
   field=Path(name).name
   if field=='memory.events':
    pairs=[line.split() for line in value.splitlines()]
    if not pairs or any(len(pair)!=2 or not pair[1].isdigit() for pair in pairs) or len({pair[0] for pair in pairs})!=len(pairs):raise ValueError('Invalid cgroup memory event counters')
   elif value!='max' or field!='memory.max':
    if not isinstance(value,str) or not value.isdigit():raise ValueError('Invalid cgroup memory counter')
 if not seen_checker:raise ValueError('No actual checker process GPU sample retained')
 return dict(samples=len(rows),checker_gpu_activity_retained=True)

def verify_prescription(stage,final):
 if set(stage)!=PRESCRIPTION_FIELDS or stage['version']!=1 or stage['phase']!='prescribed-pre-generation':raise ValueError('Exact prescription source/input/model/budget schema required; no receipt/future/back-reference')
 if any(final.get(key)!=value for key,value in stage.items() if key not in ('version','phase')):raise ValueError('Actual final stage differs from prescribed source/input/budget/model')

def verify_isolated_sources(repo,stage_path):
 import os,sys
 code="from search_r1.baselines import corrected_evaluation as ce; import json,sys,pathlib; assert pathlib.Path(ce.__file__).resolve()==pathlib.Path(sys.argv[2])/'search_r1/baselines/corrected_evaluation.py'; ce.verify_runtime_sources(json.load(open(sys.argv[1])))"
 subprocess.run([sys.executable,'-c',code,str(stage_path),str(repo)],cwd=repo,env=dict(os.environ,PYTHONPATH=str(repo)),check=True,capture_output=True,text=True)

def verify_pre_stage(supplement,receipt,base=OWN,actual_stage_path=None,actual_stage_sha=None,model_role=None,receipt_path=None,receipt_sha=None):
 from strict_stage_prerequisites import bound_file
 required={'version','phase','eval_repo','eval_commit','model_role','stage_prescription','resolved_config','input_manifest','source_files','hf_activation_audit'}
 if set(supplement)!=required or supplement['version']!=1 or supplement['phase']!='pre-generation':raise ValueError('Exact pre-generation supplement schema required')
 repo=Path(supplement['eval_repo'])
 if not repo.is_absolute() or not repo.resolve().is_relative_to(base.resolve()):raise ValueError('Evaluation repo escapes own root')
 commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
 if commit!=receipt['eval_commit'] or commit!=supplement['eval_commit'] or commit==COMMIT:raise ValueError('Actual corrected source mismatch')
 subprocess.run(['git','-C',str(repo),'diff','--quiet','HEAD'],check=True)
 if not SOURCE_NAMES<=set(supplement['source_files']):raise ValueError('Exact native source set missing')
 for name,sha in supplement['source_files'].items():bound_file(dict(path=str(repo/name),sha256=sha),repo)
 stage_path=bound_file(supplement['stage_prescription'],base);stage=loads(stage_path.read_text())
 if not actual_stage_path or not actual_stage_sha:raise ValueError('BLOCKED: actual final stage path/SHA integration not supplied')
 final_path=bound_file(dict(path=str(actual_stage_path),sha256=actual_stage_sha),base);final=loads(final_path.read_text())
 verify_prescription(stage,final)
 if not receipt_path or final.get('training_hf_receipt_path')!=str(Path(receipt_path).resolve()) or final.get('training_hf_receipt_sha256')!=receipt_sha:raise ValueError('Actual final stage is not bound to this exact layered receipt')
 role=supplement['model_role']
 if role!=model_role:raise ValueError('Requested model role differs from prescribed/actual stage')
 if role not in receipt['model_roles'] or stage['model_role']!=role or stage['model_directory']!=receipt['model_roles'][role]['directory'] or stage['model_files']!=receipt['model_roles'][role]['files']:raise ValueError('Pre-stage actual model role/content mismatch')
 if stage['eval_commit']!=commit or stage['training_commit']!=COMMIT or stage['source_files']!=supplement['source_files']:raise ValueError('Pre-stage source identity mismatch')
 inputs=bound_file(supplement['input_manifest'],base)
 if stage['input_manifest_sha256']!=digest(inputs):raise ValueError('Pregeneration input pin mismatch')
 # Invoke actual source-pinned CPU functions in an isolated process, never substitute labels.
 config=bound_file(supplement['resolved_config'],base)
 code="from search_r1.baselines import corrected_evaluation as ce; from omegaconf import OmegaConf; import json,sys,pathlib; stage=json.load(open(sys.argv[4])); assert pathlib.Path(ce.__file__).resolve()==pathlib.Path(sys.argv[5])/'search_r1/baselines/corrected_evaluation.py'; ce.verify_runtime_sources(stage); ce.load_inputs(sys.argv[1],sys.argv[2]); ce.verify_budget(OmegaConf.load(sys.argv[3]),stage)"
 import os,sys
 subprocess.run([sys.executable,'-c',code,str(inputs),digest(inputs),str(config),str(stage_path),str(repo)],cwd=repo,env=dict(os.environ,PYTHONPATH=str(repo)),check=True,capture_output=True,text=True)
 # HF historical proof was fully cross-checked before this pre-stage layer.
 audit_path=bound_file(supplement['hf_activation_audit'],base)
 if digest(audit_path)!=receipt['hf500_evidence']['activation_result']['sha256']:raise ValueError('Pre-stage HF audit does not match completed HF receipt')
 return {'pre_generation_prescription_verified':True}
