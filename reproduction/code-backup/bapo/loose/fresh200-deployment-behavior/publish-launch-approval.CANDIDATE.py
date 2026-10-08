"""Publish only the source-bound formal200 first-five evidence-capture permission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006')
SOURCE='90755a561abd239fbb23882939410e643bb913ae'
TOOLS_SHA='ec1e92369e03b141dc6a528e795a9459bc331e4296cfc65f42d895c2db0f0256'
RUNTIME_SHA='e46391c5bd939afda0e23bf9854c48114a7297eaa2d89bf79ae2a39116893618'
CPU_RAW_SHA='ba24810ff7e5dec6f278e539946b189c7f5758a2ce10b5ceec5ad822e50726d6'
CONTEXT_SHA='56ce6cacb48cf94a395d61c388e433a218d687b7d07aae5da9d51647a904917f'
RESOURCE_RAW_SHA='fc85a3cdd3555995be0f4610c44bd8fc7b60b4ab9c950123f00cc663d5a241e5'
REVIEW_HASHES={'INDEPENDENT-OFFLOAD-ROOT-CAUSE-REVIEW.json': '73acc9a35456907e78549ce1ffba529971caa47b98b2614a368d12d061460a24', 'INDEPENDENT-OFFLOAD-ROOT-CAUSE-REVIEW.md': '9c8146fb5021da3745c66529f3d9bef2f0b1df2b7b06d2ede3e50b9ee816d73d', 'INDEPENDENT-SKIP-FINAL-CODE-REVIEW.json': 'e12bcd1b3380da4e296f14b754931ec7e183b3278e760bdc0715211a00edd41a', 'INDEPENDENT-SKIP-FINAL-CODE-REVIEW.md': '7cdcd938383917b6a471d940ad4fa87a6dfb93e5fce62b8332892da97c20e011', 'independent-skip-final-critical-tests-v3.raw.log': 'e8986fb15e520214e2a30c33c53fbd7e137dd91749ab176dca80b7823358e36a', 'candidate-tests-v2.raw.log': '16b324c7a49cc6430ea5ed210bd75d9abe5afe92ae05260ef44c9bed8536426f', 'final-critical-binding-test.raw.log': '9493e042e277bb82f1f31730ffb058b10c5af58d5075b73df453e6cfd22477a9', 'INDEPENDENT-TOKEN-BOUNDARY-CODE-REVIEW.json': 'f4b0ac44a75ba99ea12904638f2e01b155d81ba6e9687e50bca75d574af16854', 'INDEPENDENT-TOKEN-BOUNDARY-CODE-REVIEW.md': 'bf9a8b10965725d220886e73f05bb4172b804e491a82da576b08f84c5e3fc441', 'independent-tokenboundary-tests.raw.log': '31136b02a29a9b230f00e34d99c4bff646fa1f77cf53202b264e1a2da8a1c2f0', 'BEHAVIOR-DEPLOYMENT-INDEPENDENT-REVIEW.json': 'a5a7f9d6d0d38fb2e2ca70b07497b5d6c46636d5c99e48ee7a42f8ede64627ff'}
RESOURCE_TEST_HASHES={'test_freeze_scalability.py': '8d71d683b17d2242f600193ba5332a0e9f4d19b04db5b7f037ef5574a7fe9e40', 'test_launch_transaction.py': '8b9385ce8c510f5af5a6cc60ff6cec8ade6de63406aeb43ee2864e04ba4a9fb7', 'test_resource_watch_recovery.py': '5acb3fb153e4d8a96b7ef0af7433d501a1786b4989abfd639d55deb79ea51ef0', 'test_runtime_anchor.py': 'bfa31421fa7289a608f6663d580e65375c48c04fcb6bb1e8d197d42d44e4ef30'}
AEC_SOURCE='aec5e057093b4c491bfb0a47abd7605d98981b56'
BEHAVIOR_REVIEW_SHA='53493fa4beca0fafa49ff92b442eff7d334c748defa064fc477405ab98191a7b'  # Await exact complete12-file independent overlay, never self-sign.
NUMERIC_REVIEW_SHA='fbfd83e9af4a59811d9bb2cc342158cabd54ed3fc0774d9555f37b2bde4f899f'
NUMERIC_MANIFEST_SHA='a52b9046bff7c933d9c68e3a88cd314204b1767f27b0523e48e60f2b353d31a2'
NUMERIC_RESULT_SHA='fc27e978fdc596c6411faea6523e406a5f780a5944e074bc418b6480d795e46f'
EXPECTED_RUNTIME_BYTES=530
EXPECTED_CPU_PASSED=186  # Parent independently executed full new selected suite:178PASS. Target must repeat.

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw(path):
 path=Path(path)
 require(path.is_absolute() and ROOT in path.parents and path.resolve()==path and path.is_file() and not path.is_symlink(),'Ordinary ownROOT artifact required: '+str(path))
 return path.read_bytes()
def parse_json(value):
 def pairs(items):
  value={}
  for key,item in items:
   require(key not in value,'Duplicate JSON key');value[key]=item
  return value
 return json.loads(value,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def document(path):return parse_json(raw(path))
def binding(path,expected=None):
 value=raw(path);digest=hashlib.sha256(value).hexdigest()
 require(expected is None or digest==expected,'Artifact bytes changed: '+str(path))
 return dict(path=str(path),bytes=len(value),sha256=digest)

def validate_runtime(runtime,watcher):
 require(runtime.get('source_commit')==SOURCE and runtime.get('root')==str(ROOT)
  and runtime.get('sys_prefix')==str(ROOT/'env') and runtime.get('host')=='h100-5'
  and runtime.get('scope')=='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE','Actual runtime capture identity differs')
 exe=watcher.validate_executable(runtime['executable'])
 require(watcher.executable_identity(exe['resolved_path'])==exe and sha(exe['resolved_path'])==runtime['executable_sha256'],'Actual executable bytes differ')
 require(watcher.executable_identity(ROOT/'env/bin/python')==exe
  and watcher.executable_identity(sys.executable)==exe and watcher.executable_identity('/proc/self/exe')==exe,'Actual ownenv/current/proc interpreter differs')

def validate_cpu(cpu,context,critical):
 require(cpu.get('source_commit')==SOURCE and type(cpu.get('exit_code')) is int and cpu['exit_code']==0
  and cpu.get('platform')=='linux' and cpu.get('host')=='h100-5'
  and cpu.get('scope')=='ACTUAL_CPU_IMPORT_TEST_CONTEXT_GATE_NOT_NATIVE_GPU'
  and cpu.get('tools_manifest_sha256')==TOOLS_SHA and cpu.get('critical_source_sha256')==critical,'Actual CPU gate not passed/source bound')
 require(cpu.get('log_path')==str(ROOT/'logs/target-cpu-gate.raw.log') and cpu.get('log_sha256')==CPU_RAW_SHA,'Actual CPU raw log identity differs')
 cpu_text=raw(ROOT/'logs/target-cpu-gate.raw.log').decode()
 require(re.search(r'^'+str(EXPECTED_CPU_PASSED)+r' passed(?:, \d+ warnings?)? in [\d.]+s',cpu_text,re.M) is not None
  and re.search(r'^\d+ (?:failed|skipped|deselected)',cpu_text,re.M) is None,'Actual CPU186 pass summary absent')
 binding(ROOT/'logs/target-cpu-gate.raw.log',CPU_RAW_SHA)
 require(cpu.get('context_audit_sha256')==CONTEXT_SHA,'Actual context hash differs')
 binding(ROOT/'logs/context-budget-audit.json',CONTEXT_SHA)
 require(context.get('status')=='ACTUAL_TARGET_CPU_ALL_SHARED_CONTEXT_BUDGET_OK_NOT_GPU_ACCEPTANCE'
  and context.get('source_commit')==SOURCE and context.get('source_sha256')==critical,'Actual allrows context identity differs')
 require({k:v['rows'] for k,v in context['records'].items()}=={'train':169028,'validation':512,'stage':64,'final':51713},'Actual allrows inventory differs')
 require(all(v['maximum_initial_tokens']<=2048 and v['worst_before_fourth_request_tokens']<=4096 for v in context['records'].values()),'Context budget failed')
 require({k:context[k] for k in ('force_final_answer','max_turns','max_start_length','max_rolling_context','max_response_tokens','max_observation_tokens')}==dict(force_final_answer=False,max_turns=4,max_start_length=2048,max_rolling_context=4096,max_response_tokens=500,max_observation_tokens=500),'Context generation contract differs')

def validate_resource(record):
 require(record.get('source_commit')==SOURCE and record.get('host')=='h100-5'
  and type(record.get('exit_code')) is int and record['exit_code']==0
  and record.get('scope')=='ACTUAL_LINUX_OWN_CHILD_RESOURCE_TESTS_NOT_GPU'
  and record.get('files')==RESOURCE_TEST_HASHES and record.get('raw_log_sha256')==RESOURCE_RAW_SHA,'Actual Linux resource gate differs')
 for name,digest in RESOURCE_TEST_HASHES.items():binding(ROOT/'tests'/name,digest)
 text=raw(ROOT/'logs/runtime-tests.actual.log').decode()
 binding(ROOT/'logs/runtime-tests.actual.log',RESOURCE_RAW_SHA)
 require(re.search(r'^Ran 80 tests in [\d.]+s$',text,re.M) is not None
  and re.search(r'^OK$',text,re.M) is not None and not re.search(r'^OK \(|^FAILED|skipped=',text,re.M),'Actual Linux80 resource tests incomplete/skipped')

def merge_source_overlay(base_snapshot,overlay,changed,boundary,boundary_changed):
 require(overlay.get('status')=='PASS_CODE_ONLY_OFFLOAD_FIX_INDEPENDENT_REVIEW'
  and overlay.get('base_source')=='379b14c65e127cd3a71524f7783deaebafbbdb41'
  and overlay.get('must_fix')==[] and overlay.get('native_acceptance') is False,'Independent offload overlay unresolved')
 expected_overlay={'verl/utils/fsdp_utils.py','search_r1/baselines/fresh200_contract.py','tests/utils/test_fsdp_optimizer_transfer.py'}
 require(set(overlay['files_sha256'])==expected_overlay,'Exact three-file offload overlay required')
 require(changed==expected_overlay,'New source has unreviewed changes beyond offload fix')
 expected_boundary={'search_r1/llm_agent/sampling_trace.py','tests/test_fresh200_sampling_trace.py','tests/test_fresh200_sampling_token_boundary.py','tests/baselines/test_fresh200_bapo_sampling_chain.py'}
 require(boundary.get('status')=='PASS_CODE_ONLY_TOKEN_BOUNDARY_FIX_INDEPENDENT_REVIEW'
  and boundary.get('base_source_commit')=='5121a8d9349eadb5a81119cc4573165338e041bd'
  and boundary.get('mustfix')==[] and boundary.get('native_acceptance') is False,'Independent token-boundary overlay unresolved')
 require(set(boundary['files_sha256'])==expected_boundary and boundary_changed==expected_boundary,'Exact four-file token-boundary overlay required; unreviewed changes refused')
 reviewed_source=dict(base_snapshot);reviewed_source.update(overlay['files_sha256']);reviewed_source.update(boundary['files_sha256'])
 return reviewed_source

EXPECTED_BEHAVIOR_FILES=frozenset(('search_r1/baselines/behavior_policy.py','search_r1/baselines/fresh200_contract.py','search_r1/baselines/fresh200_driver.py','search_r1/baselines/fresh200_gate.py','search_r1/baselines/fresh200_native_evidence.py','tests/baselines/test_fresh200_behavior_policy.py','tests/baselines/test_fresh200_native_evidence.py','tests/baselines/test_fresh200_policy_safety.py','tests/baselines/test_fresh200_sampling_evidence.py','tests/baselines/test_nonfinite_sampling_actor.py','verl/trainer/ppo/ray_trainer.py','verl/workers/actor/dp_actor.py'))

def merge_behavior_overlay(reviewed,overlay,changed):
 require(overlay.get('status')=='PASS_CODE_ONLY_BEHAVIOR_POLICY_SOURCE_OVERLAY_NOT_NATIVE'and overlay.get('base_commit')==AEC_SOURCE and overlay.get('clean_frozen_source')is True and overlay.get('source_commit')==SOURCE and overlay.get('mustfix')==[]and overlay.get('native_acceptance')is False,'Independent complete behavior overlay unresolved')
 require(set(overlay.get('files',{}))==EXPECTED_BEHAVIOR_FILES and changed==EXPECTED_BEHAVIOR_FILES,'Exact aec5-to907 twelve-file behavior overlay required')
 for item in overlay['files'].values():
  digest=item.get('source_sha256');require(type(digest)is str and re.fullmatch('[0-9a-f]{64}',digest)is not None and item.get('status')in('A','M'),'Typed overlay source/status required')
  base=item.get('base_sha256');require((item['status']=='A'and base is None)or(item['status']=='M'and type(base)is str and re.fullmatch('[0-9a-f]{64}',base)is not None),'Exact added/modified base binding required')
 result=dict(reviewed);result.update({name:item['source_sha256']for name,item in overlay['files'].items()});return result

def validate_numeric_interpretation(manifest,result,review):
 root=ROOT/'numeric-diagnostic-3310-20261006'
 require(manifest.get('schema')=='ACTUAL_ZERO_ADAM_NUMERIC_TERMINAL_EVIDENCE_V1'and manifest.get('root')==str(root)and manifest.get('source_commit')==SOURCE,'New-source numeric terminal manifest required')
 require(type(manifest.get('job_id'))is int and manifest['job_id']>3310 and manifest.get('scheduler_state')=='COMPLETED'and type(manifest.get('exit_code'))is int and manifest['exit_code']==0 and type(manifest.get('restarts'))is int and manifest['restarts']==0 and type(manifest.get('requeue'))is int and manifest['requeue']==0,'Actual unique numeric completed terminal required')
 require(manifest.get('protocol')=='BAPO_ZERO_ADAM_NUMERIC_DIAGNOSTIC_V1'and manifest.get('phase')=='numeric-zero-adam-replay'and manifest.get('node')=='h100-5','Actual numeric scope/node required')
 nonce=manifest.get('nonce');require(type(nonce)is str and re.fullmatch('[0-9a-f]{32}',nonce)is not None,'Actual numeric nonce required')
 run=root/'runs'/f"numeric-zero-adam-replay-{nonce}-{manifest['job_id']}"
 require(manifest.get('run_root')==str(run)and type(manifest.get('allocated_gpu_count'))is int and manifest['allocated_gpu_count']==8 and type(manifest.get('computed_gpu_count'))is int and manifest['computed_gpu_count']==1,'Actual owned numeric allocation required')
 require(result.get('status')=='ACTUAL_ZERO_ADAM_NUMERIC_DIAGNOSTIC_NOT_ADMISSION'and result.get('source_commit')==SOURCE and result.get('raw_source_commit')==AEC_SOURCE and result.get('numeric_probe_sha256')==manifest.get('numeric_probe_sha256'),'Actual numeric result/source/code binding required')
 for value in (manifest,result):
  require(value.get('native_acceptance')is False and value.get('production_release')is False and type(value.get('optimizer_writes'))is int and value['optimizer_writes']==0,'Diagnostic cannot claim native release/Adam writes')
 for field in ('formal_outer_updates','scheduler_updates'):require(type(manifest.get(field))is int and manifest[field]==0,'Zero diagnostic formal/scheduler updates required')
 require(review.get('status')=='PASS_ACTUAL_ZERO_ADAM_NUMERIC_INTERPRETATION_CONDITIONAL_FIRST5'and review.get('source_commit')==SOURCE and review.get('mustfix')==[]and review.get('native_acceptance')is False and review.get('whole_production_accepted')is False and review.get('whole_200_accepted')is False and review.get('production_release')is False and review.get('first5_capture_eligible')is True,'New actual numerical diagnostic independent interpretation unresolved')
 require(review.get('terminal_manifest_sha256')==NUMERIC_MANIFEST_SHA and review.get('bindings',{}).get('numeric_result_sha256')==NUMERIC_RESULT_SHA and review.get('job_id')==manifest['job_id'],'Independent interpretation exact actual artifacts/job binding required')
 require(review.get('root')==str(root)and review.get('production_root')==str(ROOT)and review.get('run_root')==str(run)and review.get('nonce')==nonce and review.get('reviewer_task_id')=='/root/fresh200_safety_math_review'and review.get('producer_task_id')=='/root/fresh200_actor_safety','Actual independent numerical review identity/scope differs')
 for review_key,role in {'selected_logit_sidecar_sha256':'selected_logit_sidecar','resource_replay_sha256':'resource_replay','own_cleanup_sha256':'own_cleanup','head_receipt_origin_sha256':'head_receipt_origin','slurm_json_sha256':'slurm_json'}.items():
  require(review.get('bindings',{}).get(review_key)==manifest.get('artifacts',{}).get(role,{}).get('sha256'),'Independent actual artifact binding differs '+role)
 required={'numeric_result','selected_logit_sidecar','phase_log','resource_journal','resource_replay','own_cleanup','worker_receipt','head_receipt','approval','slurm_terminal','head_receipt_origin','slurm_json'}
 require(set(manifest.get('artifacts',{}))==required,'Complete numeric runtime/sidecar/owncleanup/scheduler evidence required')
 checked=[]
 for role,item in manifest['artifacts'].items():
  path=Path(item['path']);require(root in path.parents and type(item.get('bytes'))is int and item['bytes']>0,'Only actual owned numeric evidence')
  current=binding(path,item['sha256']);require(current['bytes']==item['bytes'],'Actual numeric artifact changed');checked.append(current)
 receipt=document(Path(manifest['artifacts']['worker_receipt']['path']));head=document(Path(manifest['artifacts']['head_receipt']['path']))
 origin=document(Path(manifest['artifacts']['head_receipt_origin']['path']))
 head_raw=raw(Path(manifest['artifacts']['head_receipt']['path']))
 require(origin.get('origin')=='ACTUAL_H10016_HEAD_READ'and origin.get('path')==str(root/'scheduler-head/numeric-zero-adam-replay-receipt.json')and type(origin.get('text'))is str and origin['text'].encode()==head_raw and origin.get('sha256')==hashlib.sha256(head_raw).hexdigest(),'Actual separate head-origin bytes required')
 slurm=document(Path(manifest['artifacts']['slurm_json']['path']))
 require(type(slurm.get('job_id'))is int and slurm['job_id']==manifest['job_id']and type(slurm.get('returncode'))is int and slurm['returncode']==0 and slurm.get('stdout')==raw(Path(manifest['artifacts']['slurm_terminal']['path'])).decode(),'Actual successful terminal command raw bytes required')

 require(receipt==head and receipt.get('job_id')==manifest['job_id']and receipt.get('source')==SOURCE and receipt.get('nonce')==nonce and receipt.get('phase')==manifest['phase'],'Actual head/worker exact receipt required')
 terminal=raw(Path(manifest['artifacts']['slurm_terminal']['path'])).decode()
 for key,value in {'JobId':str(manifest['job_id']),'JobState':'COMPLETED','ExitCode':'0:0','Restarts':'0','Requeue':'0','NodeList':'h100-5','NumNodes':'1','Comment':'BAPO_ZERO_ADAM:'+manifest['phase']+':'+SOURCE+':'+nonce,'Command':str(root/'tools/numeric-zero-adam-replay.sbatch')}.items():
  require(re.search(r'(?:^|\s)'+re.escape(key)+'='+re.escape(value)+r'(?:\s|$)',terminal) is not None,'Actual raw Slurm terminal identity differs: '+key)
 require(re.search(r'(?:^|\s)UserId='+re.escape(__import__('getpass').getuser())+r'\(',terminal)is not None,'Actual diagnostic user owner differs')
 approval=document(Path(manifest['artifacts']['approval']['path']))
 require(approval.get('protocol')==manifest['protocol']and approval.get('source_commit')==SOURCE and approval.get('root')==str(root)and approval.get('nonce')==nonce and approval.get('numeric_probe_sha256')==manifest['numeric_probe_sha256'],'Actual diagnostic approval differs')
 require(result.get('approval',{}).get('job_id')==manifest['job_id']and result['approval'].get('sha256')==manifest['artifacts']['approval']['sha256'],'Actual numeric result allocation/approval differs')
 models=result.get('model_states',[]);require(len(models)==2 and all(x.get('before')==x.get('after')and x.get('zero_adam')is True and x.get('model_eval')is True and x.get('gradients_absent')is True and x.get('optimizer_constructed')is False and x.get('scheduler_constructed')is False for x in models),'Actual numerical model before/after zero-write evidence missing')
 cleanup=document(Path(manifest['artifacts']['own_cleanup']['path']));require(cleanup.get('own_marker')==str(run)and cleanup.get('remaining')==[]and isinstance(cleanup.get('stopped_pids'),list),'Owncleanup terminal proof must show exact run/no retained owned processes')
 replay=document(Path(manifest['artifacts']['resource_replay']['path']));require(replay.get('status')=='PASS_ACTUAL_GUARD_JOURNAL_REPLAY'and replay.get('job_id')==manifest['job_id']and replay.get('source_commit')==SOURCE and replay.get('run_root')==str(run)and replay.get('unfinished_recovery')is False,'Actual resource replay required')
 require(binding(Path(manifest['artifacts']['numeric_result']['path']))['sha256']==NUMERIC_RESULT_SHA,'Actual result sha differs')
 side=result.get('selected_logit_sidecar');require(isinstance(side,dict)and side.get('path')==manifest['artifacts']['selected_logit_sidecar']['path']and side.get('sha256')==manifest['artifacts']['selected_logit_sidecar']['sha256']and side.get('bytes')==manifest['artifacts']['selected_logit_sidecar']['bytes'],'Exact actual selected-logit sidecar binding required')
 return checked

def build_approval(expected_goal_sha256):
 require(not sys.flags.optimize,'Optimized Python forbidden')
 require(all(type(v)is str and re.fullmatch('[0-9a-f]{64}',v)is not None for v in (BEHAVIOR_REVIEW_SHA,NUMERIC_REVIEW_SHA,NUMERIC_MANIFEST_SHA,NUMERIC_RESULT_SHA)),'UNBOUND: complete907 independent overlay and actual new numeric terminal/interpretation not ready')
 require(all(isinstance(value,str)and re.fullmatch('[0-9a-f]{64}',value)for value in (CPU_RAW_SHA,CONTEXT_SHA,RUNTIME_SHA,RESOURCE_RAW_SHA))and len(REVIEW_HASHES)>=5 and type(EXPECTED_RUNTIME_BYTES)is int and EXPECTED_RUNTIME_BYTES>0,'New actual R5 CPU/context binding not ready')
 require(sys.platform=='linux' and os.uname().nodename=='h100-5' and sys.prefix==str(ROOT/'env')
  and ROOT.resolve()==ROOT and not ROOT.is_symlink(),'Actual target ownROOT environment required')
 require(raw(ROOT/'source-expected.txt').decode().strip()==SOURCE,'Source marker differs')
 require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
  and not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Source HEAD changed/dirty')
 manifest=document(ROOT/'logs/tools-manifest.json');binding(ROOT/'logs/tools-manifest.json',TOOLS_SHA)
 require(manifest['source_commit']==SOURCE and len(manifest['files'])==21
  and set(manifest['files'])=={p.name for p in (ROOT/'tools').iterdir() if p.is_file()},'Exact21 tool inventory required')
 for name,digest in manifest['files'].items():binding(ROOT/'tools'/name,digest)
 sys.path.insert(0,str(ROOT/'repo'))
 import search_r1.baselines.fresh200_contract as contract
 require(Path(contract.__file__).resolve()==ROOT/'repo/search_r1/baselines/fresh200_contract.py','Contract module origin differs')
 critical=contract.critical_source_hashes()
 cpu=document(ROOT/'logs/target-cpu-gate.json');context=document(ROOT/'logs/context-budget-audit.json')
 validate_cpu(cpu,context,critical)
 common=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data/manifest.json')
 data_sha=sha(common)
 require(data_sha=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24' and context['data_manifest_sha256']==data_sha,'Actual shared data manifest differs')
 model_sha=contract.verify_official_model_assets(ROOT/'assets/qwen2.5-3b-instruct-aa8e725')
 require(context['official_model_manifest_sha256']==model_sha,'Actual official model context binding differs')
 validate_resource(document(ROOT/'logs/runtime-tests.actual.json'))
 reviews=ROOT/'logs/reviews'
 evidence=[binding(reviews/name,digest) for name,digest in REVIEW_HASHES.items()]
 source_review=document(reviews/'INDEPENDENT-SKIP-FINAL-CODE-REVIEW.json')
 require(source_review['status']=='PASS_CODE_ONLY_INDEPENDENT_REVIEW' and source_review['must_fix']==[] and source_review['native_acceptance'] is False,'Independent source review unresolved')
 overlay=document(reviews/'INDEPENDENT-OFFLOAD-ROOT-CAUSE-REVIEW.json')
 changed=set(subprocess.check_output(['git','-C',str(ROOT/'repo'),'diff','--name-only','379b14c65e127cd3a71524f7783deaebafbbdb41','5121a8d9349eadb5a81119cc4573165338e041bd'],text=True).splitlines())
 boundary=document(reviews/'INDEPENDENT-TOKEN-BOUNDARY-CODE-REVIEW.json')
 boundary_changed=set(subprocess.check_output(['git','-C',str(ROOT/'repo'),'diff','--name-only','5121a8d9349eadb5a81119cc4573165338e041bd',AEC_SOURCE],text=True).splitlines())
 reviewed_source=merge_source_overlay(source_review['source_snapshot_sha256'],overlay,changed,boundary,boundary_changed)
 behavior_path=reviews/'BEHAVIOR-POLICY-12FILE-OVERLAY-INDEPENDENT-REVIEW.json';evidence.append(binding(behavior_path,BEHAVIOR_REVIEW_SHA))
 behavior_changed=set(subprocess.check_output(['git','-C',str(ROOT/'repo'),'diff','--name-only',AEC_SOURCE,SOURCE],text=True).splitlines())
 behavior=document(behavior_path);reviewed_source=merge_behavior_overlay(reviewed_source,behavior,behavior_changed)
 statuses=dict(line.split('\t',1)[::-1] for line in subprocess.check_output(['git','-C',str(ROOT/'repo'),'diff','--name-status',AEC_SOURCE,SOURCE],text=True).splitlines())
 require(statuses=={name:item['status']for name,item in behavior['files'].items()},'Actual twelve git status entries differ')
 for name,item in behavior['files'].items():
  if item['status']=='M':require(hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT/'repo'),'show',AEC_SOURCE+':'+name])).hexdigest()==item['base_sha256'],'Exact reviewed base source bytes differ')
 numeric_manifest_path=ROOT/'logs/numeric-terminal-evidence.json';numeric_review_path=reviews/'NUMERIC-ACTUAL-INDEPENDENT-INTERPRETATION.json'
 evidence.extend([binding(numeric_manifest_path,NUMERIC_MANIFEST_SHA),binding(numeric_review_path,NUMERIC_REVIEW_SHA)])
 numeric_manifest=document(numeric_manifest_path);numeric_result=document(Path(numeric_manifest['artifacts']['numeric_result']['path']))
 evidence.extend(validate_numeric_interpretation(numeric_manifest,numeric_result,document(numeric_review_path)))
 for name,digest in reviewed_source.items():require(sha(ROOT/'repo'/name)==digest,'Independently reviewed source/overlay bytes changed')
 deployment=document(reviews/'BEHAVIOR-DEPLOYMENT-INDEPENDENT-REVIEW.json')
 require(deployment['status']=='PASS_CODE_ONLY_NEW_BEHAVIOR_DEPLOYMENT_NOT_NATIVE'
  and deployment['root']==str(ROOT) and deployment['source_commit']==SOURCE
  and deployment['tools_manifest_sha256']==TOOLS_SHA and deployment['files']==manifest['files']
  and deployment['mustfix']==[]
  and deployment['native_acceptance'] is False,'Independent R5 deployment review differs')
 require(re.fullmatch('[0-9a-f]{64}',expected_goal_sha256) is not None,'Trusted actual human-goal SHA required')
 goal_path=ROOT/'logs/latest-human-goal-behavior.json';goal_raw=raw(goal_path)
 require(hashlib.sha256(goal_raw).hexdigest()==expected_goal_sha256,'Trusted latest human-goal bytes differ')
 goal=parse_json(goal_raw)
 require(goal.get('threadId')=='01a0fcbd-41d8-7910-a410-372ce5c8870c' and goal.get('status')=='active'
  and type(goal.get('updatedAt')) is int and goal['updatedAt']>=1791287881
  and isinstance(goal.get('objective'),str) and goal['objective'].strip(),'Actual trusted active human goal differs')
 fixed=document(ROOT/'logs/frozen-policy-safety-config.json');contract.verify_fixed_policy_safety_artifact(fixed)
 fixed_path=Path(fixed['fixed_threshold_artifact']);artifact=document(fixed_path)
 require(fixed.get('source_commit')==SOURCE and artifact.get('human_goal_sha256')==expected_goal_sha256,'User-fixed artifact not bound latest trusted goal')
 fixed_binding=binding(fixed_path,fixed['fixed_threshold_artifact_sha256'])
 spec=importlib.util.spec_from_file_location('reviewed_fresh200_watch',ROOT/'tools/resource_watch.py')
 watcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(watcher)
 runtime_binding=binding(ROOT/'logs/runtime-executable.json',RUNTIME_SHA)
 require(runtime_binding['bytes']==EXPECTED_RUNTIME_BYTES,'Actual original runtime bytes differ')
 validate_runtime(document(ROOT/'logs/runtime-executable.json'),watcher)
 evidence.extend(binding(ROOT/'logs'/name) for name in ('target-cpu-gate.json','context-budget-audit.json','runtime-tests.actual.json','runtime-tests.actual.log','target-cpu-gate.raw.log','runtime-executable.json','frozen-policy-safety-config.json'))
 evidence.extend(binding(ROOT/'tests'/name,digest) for name,digest in RESOURCE_TEST_HASHES.items())
 evidence.append(fixed_binding)
 evidence.append(dict(path=str(goal_path),bytes=len(goal_raw),sha256=expected_goal_sha256))
 review_path=reviews/'BEHAVIOR-DEPLOYMENT-INDEPENDENT-REVIEW.json'
 value=dict(protocol='BAPO_FRESH200_LAUNCH_V1',root=str(ROOT),source_commit=SOURCE,tools_manifest_sha256=TOOLS_SHA,
  authorized_phases=['formal200'],probability_contract='ACTUAL_NATIVE_BEHAVIOR_V2_HF_DIAGNOSTIC_SEPARATE',behavior_policy_role='actual_native_stochastic_sampling_policy',hf_native_drift_is_identity_gate=False,scope='FORMAL200_FIRST_FIVE_SAME_PROCESS_EVIDENCE_CAPTURE_NOT_NATIVE_ACCEPTANCE',
  runtime_executable=runtime_binding,independent_review_path=str(review_path),independent_review_sha256=REVIEW_HASHES[review_path.name],
  human_goal_sha256=expected_goal_sha256,pause_after_accepted_outer=5,full_horizon=200,warmup_completed_outers=57,
  native_gpu_acceptance=False,whole_production_pass=False,fifth_outer_independent_release_required=True,evidence=evidence)
 watcher.validate_fresh200_approval(value,SOURCE,ROOT)
 return value

def publish(expected_goal_sha256):
 value=build_approval(expected_goal_sha256)
 for item in value['evidence']:
  current=binding(Path(item['path']),item['sha256'])
  require(current['bytes']==item['bytes'],'Validated artifact changed before publication')
 output=ROOT/'logs/launch-approval.json'
 require(not output.exists() and not output.is_symlink(),'Existing approval cannot be replaced')
 temporary=output.with_name('.launch-approval-'+secrets.token_hex(16)+'.tmp')
 encoded=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
 with temporary.open('xb') as handle:handle.write(encoded);handle.flush();os.fsync(handle.fileno())
 try:os.link(temporary,output)
 finally:temporary.unlink()
 descriptor=os.open(output.parent,os.O_RDONLY)
 try:os.fsync(descriptor)
 finally:os.close(descriptor)
 print(json.dumps(dict(path=str(output),sha256=sha(output),protocol=value['protocol'],scope=value['scope'])))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--expected-human-goal-sha',required=True);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
 if args.publish:publish(args.expected_human_goal_sha)
 else:print(json.dumps(build_approval(args.expected_human_goal_sha),sort_keys=True,indent=2))
