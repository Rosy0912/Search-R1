"""CPU-only pre-generation verification; no future corrected evaluation terminal needed."""
import datetime,hashlib,re,subprocess,math
from pathlib import Path
from training_evidence import loads,digest,COMMIT,OWN
SOURCE_NAMES={'search_r1/baselines/corrected_evaluation.py','search_r1/baselines/evaluation.py','search_r1/baselines/bapo.py','search_r1/qa_protocol.py','verl/trainer/main_ppo.py','verl/utils/reward_score/qa_em.py','search_r1/baselines/main.py','verl/trainer/ppo/ray_trainer.py','search_r1/llm_agent/generation.py','search_r1/baselines/actor.py','search_r1/baselines/distillation.py','verl/workers/rollout/vllm_rollout/vllm_rollout.py','verl/utils/dataset/rl_dataset.py','search_r1/baselines/corrected_runtime_capture.py','verl/workers/fsdp_workers.py','verl/single_controller/base/worker.py','verl/single_controller/base/decorator.py','verl/single_controller/ray/base.py'}
HF_TEMPLATE_SHA='ff617647f4e986920c6c27867fdf60e70ae7b43b8f7adf8fda62441abfae5850'

def verify_hf_activation(manifest_path,script_path,checker_path,external_sha):
 """Verify reviewed actual-final bundle ancestry, not equality with blocked preview."""
 folder=Path(manifest_path).parent
 if digest(manifest_path)!=external_sha:raise ValueError('Actual HF final manifest external pin mismatch')
 actual=loads(Path(manifest_path).read_text());original=loads((Path(__file__).parent/'source-manifest.json').read_text())
 if actual.get('prior_hf_preview_manifest_sha256')!=HF_TEMPLATE_SHA:raise ValueError('Actual HF manifest lacks exact template ancestry')
 if actual.get('preview_blocked') is not False or actual.get('source_commit')!=COMMIT or actual.get('formal_job')!='3007':raise ValueError('HF final source/job/activation mismatch')
 if actual.get('directory_name')!=folder.name or not re.fullmatch('[a-z0-9-]+',folder.name):raise ValueError('HF actual final directory mismatch')
 if set(actual['file_sources'])!=set(original['file_sources']):raise ValueError('Actual HF final helper set differs')
 for name,sha in actual['file_sources'].items():
  if digest(folder/name)!=sha:raise ValueError('Actual HF final helper bytes unbound')
  if name!='reload_job.sh' and sha!=original['file_sources'][name]:raise ValueError('HF helper changed from fixed upstream checker/source gate')
 if Path(checker_path).resolve()!=(folder/'checkpoint_reload.py').resolve() or digest(checker_path)!=original['file_sources']['checkpoint_reload.py']:raise ValueError('HF actual checker is not immutable5586 source')
 if Path(script_path).resolve()!=(folder/'reload_job.sh').resolve():raise ValueError('HF activated script role is outside exact bundle')
 script=(Path(__file__).parent/'reload_job.sh').read_text();guard='echo "STAGED PREVIEW: independent review required" >&2\nexit 78\n'
 if script.count(guard)!=1:raise ValueError('Unknown original preview guard')
 expected=script.replace(guard,'').replace(original['directory_name'],folder.name)
 if Path(script_path).read_text()!=expected:raise ValueError('HF activated script not exact guard removal and directory substitution')
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
   if process['uuid']!=selected_uuid or process['pid'] not in known or known[process['pid']]['process_start_ticks']!=process['process_start_ticks']:raise ValueError('GPU process outside own selected UUID/cgroup identity')
   if process['pid']==checker_pid and process['process_start_ticks']==str(checker_start):seen_checker=True
  # Validate finite JSON before entry; require cgroup actual ownjob memory readings.
  memory=row['cgroup']
  if not memory or any(not re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',name) for name in memory):raise ValueError('Resource cgroup outside own Slurm job')
 if not seen_checker:raise ValueError('No actual checker process GPU sample retained')
 return dict(samples=len(rows),checker_gpu_activity_retained=True)

def verify_pre_stage(supplement,receipt,base=OWN,actual_stage_path=None,actual_stage_sha=None):
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
 # Prescription excludes receipt references: no receipt→stage→receipt hash cycle.
 if {'training_hf_receipt_path','training_hf_receipt_sha256','strict_training_hf_verifier'} & set(stage):raise ValueError('Stage prescription contains circular receipt references')
 if not actual_stage_path or not actual_stage_sha:raise ValueError('BLOCKED: actual final stage path/SHA integration not supplied')
 final_path=bound_file(dict(path=str(actual_stage_path),sha256=actual_stage_sha),base);final=loads(final_path.read_text())
 if any(final.get(key)!=value for key,value in stage.items()):raise ValueError('Actual final stage differs from prescribed source/input/budget/model')
 role=supplement['model_role']
 if role not in receipt['model_roles'] or stage['model_role']!=role or stage['model_directory']!=receipt['model_roles'][role]['directory'] or stage['model_files']!=receipt['model_roles'][role]['files']:raise ValueError('Pre-stage actual model role/content mismatch')
 if stage['eval_commit']!=commit or stage['training_commit']!=COMMIT or stage['source_files']!=supplement['source_files']:raise ValueError('Pre-stage source identity mismatch')
 inputs=bound_file(supplement['input_manifest'],base)
 if stage['input_manifest_sha256']!=digest(inputs):raise ValueError('Pregeneration input pin mismatch')
 # Invoke actual source-pinned CPU functions in an isolated process, never substitute labels.
 config=bound_file(supplement['resolved_config'],base)
 code="from search_r1.baselines.corrected_evaluation import load_inputs,verify_budget; from omegaconf import OmegaConf; import json,sys; load_inputs(sys.argv[1],sys.argv[2]); verify_budget(OmegaConf.load(sys.argv[3]),json.load(open(sys.argv[4])))"
 import os,sys
 subprocess.run([sys.executable,'-c',code,str(inputs),digest(inputs),str(config),str(stage_path)],env=dict(os.environ,PYTHONPATH=str(repo)),check=True,capture_output=True,text=True)
 # Only HF historical audit is allowed here. Future corrected-job runtime is a separate layer.
 audit_path=bound_file(supplement['hf_activation_audit'],base);audit=loads(audit_path.read_text())
 if audit['schema']!='actual-hf500-activation-v1' or audit['job_id']!=receipt['hf500_reload_job'] or audit['node']!='h100-10':raise ValueError('HF activation audit identity mismatch')
 if not re.search(r'(?:^|/)job_'+re.escape(audit['job_id'])+r'(?:/|$)',audit['own_cgroup']):raise ValueError('HF own cgroup missing')
 resources=bound_file(receipt['hf500_evidence']['resources'],base)
 return verify_resource_rows([loads(line) for line in resources.read_text().splitlines()],audit['job_id'],audit['gpu_uuid'],audit['uid'],audit['checker_pid'],audit['checker_start_ticks'])
