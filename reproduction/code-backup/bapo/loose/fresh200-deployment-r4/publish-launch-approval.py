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

ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r4-20261006')
SOURCE='5121a8d9349eadb5a81119cc4573165338e041bd'
TOOLS_SHA='96a55be243f07691df1988d2df194cb5754c50cc3a5b6115dc0788b61d5eacc8'
RUNTIME_SHA='17783395a338b2f6cfe1274b030549bde40762acb2f9b23c3a0690fecd193724'
CPU_RAW_SHA='dd05eff053f30dba467a2986778eab9b27924ab8fb62671920d1bc0fb6eceb28'
CONTEXT_SHA='e254381699d4d5715aa85e88ca36b2eece87402041ef9898a39a1075128efb0e'
RESOURCE_RAW_SHA='ed85a53ed70c3cd58d737d18440d0c647a1bc03cdd6c7e9874afda8070bd0cbb'
REVIEW_HASHES={'INDEPENDENT-R4-DEPLOYMENT-CODE-REVIEW.json': '6ee346d0cb2fe184c42592968f6754697b5af8a0fe9804a8d21794c188f5e9f7', 'INDEPENDENT-R4-DEPLOYMENT-CODE-REVIEW.md': 'd2b84d40c11529df1083449098f2ae7313b5eb39681e336d32975c5499807263', 'INDEPENDENT-SKIP-FINAL-CODE-REVIEW.json': 'e12bcd1b3380da4e296f14b754931ec7e183b3278e760bdc0715211a00edd41a', 'INDEPENDENT-SKIP-FINAL-CODE-REVIEW.md': '7cdcd938383917b6a471d940ad4fa87a6dfb93e5fce62b8332892da97c20e011', 'independent-skip-final-critical-tests-v3.raw.log': 'e8986fb15e520214e2a30c33c53fbd7e137dd91749ab176dca80b7823358e36a', 'INDEPENDENT-OFFLOAD-ROOT-CAUSE-REVIEW.json': '73acc9a35456907e78549ce1ffba529971caa47b98b2614a368d12d061460a24', 'INDEPENDENT-OFFLOAD-ROOT-CAUSE-REVIEW.md': '9c8146fb5021da3745c66529f3d9bef2f0b1df2b7b06d2ede3e50b9ee816d73d', 'candidate-tests-v2.raw.log': '16b324c7a49cc6430ea5ed210bd75d9abe5afe92ae05260ef44c9bed8536426f', 'final-critical-binding-test.raw.log': '9493e042e277bb82f1f31730ffb058b10c5af58d5075b73df453e6cfd22477a9'}
RESOURCE_TEST_HASHES={
 'test_freeze_scalability.py':'8d71d683b17d2242f600193ba5332a0e9f4d19b04db5b7f037ef5574a7fe9e40',
 'test_launch_transaction.py':'8b9385ce8c510f5af5a6cc60ff6cec8ade6de63406aeb43ee2864e04ba4a9fb7',
 'test_resource_watch_recovery.py':'5acb3fb153e4d8a96b7ef0af7433d501a1786b4989abfd639d55deb79ea51ef0',
 'test_runtime_anchor.py':'bfa31421fa7289a608f6663d580e65375c48c04fcb6bb1e8d197d42d44e4ef30'}

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
 require(re.search(r'^165 passed(?:, \d+ warnings?)? in [\d.]+s',cpu_text,re.M) is not None
  and re.search(r'^\d+ (?:failed|skipped|deselected)',cpu_text,re.M) is None,'Actual CPU165 pass summary absent')
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

def merge_source_overlay(base_snapshot,overlay,changed):
 require(overlay.get('status')=='PASS_CODE_ONLY_OFFLOAD_FIX_INDEPENDENT_REVIEW'
  and overlay.get('base_source')=='379b14c65e127cd3a71524f7783deaebafbbdb41'
  and overlay.get('must_fix')==[] and overlay.get('native_acceptance') is False,'Independent offload overlay unresolved')
 expected_overlay={'verl/utils/fsdp_utils.py','search_r1/baselines/fresh200_contract.py','tests/utils/test_fsdp_optimizer_transfer.py'}
 require(set(overlay['files_sha256'])==expected_overlay,'Exact three-file offload overlay required')
 require(changed==expected_overlay,'New source has unreviewed changes beyond offload fix')
 reviewed_source=dict(base_snapshot);reviewed_source.update(overlay['files_sha256'])
 return reviewed_source

def build_approval(expected_goal_sha256):
 require(not sys.flags.optimize,'Optimized Python forbidden')
 require(isinstance(CPU_RAW_SHA,str) and isinstance(CONTEXT_SHA,str),'New actual R4 CPU/context binding not ready')
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
 changed=set(subprocess.check_output(['git','-C',str(ROOT/'repo'),'diff','--name-only','379b14c65e127cd3a71524f7783deaebafbbdb41',SOURCE],text=True).splitlines())
 reviewed_source=merge_source_overlay(source_review['source_snapshot_sha256'],overlay,changed)
 for name,digest in reviewed_source.items():require(sha(ROOT/'repo'/name)==digest,'Independently reviewed source/overlay bytes changed')
 deployment=document(reviews/'INDEPENDENT-R4-DEPLOYMENT-CODE-REVIEW.json')
 require(deployment['status']=='PASS_CODE_ONLY_R4_SOURCE_TOOLS_HEAD_DEPLOYMENT_REVIEW'
  and deployment['target_root']==str(ROOT) and deployment['source_commit']==SOURCE
  and deployment['tools_manifest_sha256']==TOOLS_SHA and deployment['tools_files_sha256']==manifest['files']
  and deployment['must_fix']==[] and deployment['scoped_source_tools_env_deployment_eligible'] is True
  and deployment['native_acceptance'] is False,'Independent R4 deployment review differs')
 require(re.fullmatch('[0-9a-f]{64}',expected_goal_sha256) is not None,'Trusted actual human-goal SHA required')
 goal_path=ROOT/'logs/latest-human-goal-r4.json';goal_raw=raw(goal_path)
 require(hashlib.sha256(goal_raw).hexdigest()==expected_goal_sha256,'Trusted latest human-goal bytes differ')
 goal=parse_json(goal_raw)
 require(goal.get('threadId')=='01a0fcbd-41d8-7910-a410-372ce5c8870c' and goal.get('status')=='active'
  and type(goal.get('updatedAt')) is int and goal['updatedAt']>=1791287881
  and isinstance(goal.get('objective'),str) and goal['objective'].strip(),'Actual latest human goal differs')
 fixed=document(ROOT/'logs/frozen-policy-safety-config.json');contract.verify_fixed_policy_safety_artifact(fixed)
 fixed_path=Path(fixed['fixed_threshold_artifact']);artifact=document(fixed_path)
 require(fixed.get('source_commit')==SOURCE and artifact.get('human_goal_sha256')==expected_goal_sha256,'User-fixed artifact not bound latest trusted goal')
 fixed_binding=binding(fixed_path,fixed['fixed_threshold_artifact_sha256'])
 spec=importlib.util.spec_from_file_location('reviewed_fresh200_watch',ROOT/'tools/resource_watch.py')
 watcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(watcher)
 runtime_binding=binding(ROOT/'logs/runtime-executable.json',RUNTIME_SHA)
 require(runtime_binding['bytes']==532,'Actual original runtime bytes differ')
 validate_runtime(document(ROOT/'logs/runtime-executable.json'),watcher)
 evidence.extend(binding(ROOT/'logs'/name) for name in ('target-cpu-gate.json','context-budget-audit.json','runtime-tests.actual.json','runtime-tests.actual.log','target-cpu-gate.raw.log','runtime-executable.json','frozen-policy-safety-config.json'))
 evidence.extend(binding(ROOT/'tests'/name,digest) for name,digest in RESOURCE_TEST_HASHES.items())
 evidence.append(fixed_binding)
 evidence.append(dict(path=str(goal_path),bytes=len(goal_raw),sha256=expected_goal_sha256))
 review_path=reviews/'INDEPENDENT-R4-DEPLOYMENT-CODE-REVIEW.json'
 value=dict(protocol='BAPO_FRESH200_LAUNCH_V1',root=str(ROOT),source_commit=SOURCE,tools_manifest_sha256=TOOLS_SHA,
  authorized_phases=['formal200'],scope='FORMAL200_FIRST_FIVE_SAME_PROCESS_EVIDENCE_CAPTURE_NOT_NATIVE_ACCEPTANCE',
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
 require(len(sys.argv)==2,'Trusted actual latest-human-goal SHA argument required')
 publish(sys.argv[1])
