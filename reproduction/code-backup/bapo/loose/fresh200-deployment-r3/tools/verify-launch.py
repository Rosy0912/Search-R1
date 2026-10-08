"""Read-only new-source launch gate. No legacy acceptance is consulted."""
from pathlib import Path
import hashlib,json,subprocess,sys,os
if sys.flags.optimize:
 raise RuntimeError('optimized Python forbidden: launch assertions must execute')
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r3-20261006')
PHASES={'diagnostic-discovery128','diagnostic-discovery256','diagnostic-confirmation128','diagnostic-confirmation256','formal200'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(1048576),b''):h.update(block)
 return h.hexdigest()
def verify(phase):
 assert phase in PHASES
 assert sys.platform=='linux' and os.uname().nodename=='h100-5'
 assert ROOT.resolve()==ROOT and sys.prefix==str(ROOT/'env')
 repo=ROOT/'repo';source=(ROOT/'source-expected.txt').read_text().strip()
 assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
 manifest_path=ROOT/'logs/tools-manifest.json';manifest=json.loads(manifest_path.read_text())
 approval=json.loads((ROOT/'logs/launch-approval.json').read_text())
 assert phase in approval['authorized_phases']
 assert approval['protocol']=='BAPO_FRESH200_LAUNCH_V1' and approval['source_commit']==manifest['source_commit']==source
 assert approval['tools_manifest_sha256']==sha(manifest_path)
 # Fail before phase launch if its original runtime binding is absent or stale.
 from resource_watch import load_runtime_anchor
 load_runtime_anchor(str(ROOT/'runs/__launch_identity_gate__'),verify_current=True)
 assert set(manifest['files'])=={p.name for p in (ROOT/'tools').iterdir() if p.is_file()}
 for name,value in manifest['files'].items():
  path=ROOT/'tools'/name
  assert path.resolve()==path and sha(path)==value
 review=Path(approval['independent_review_path'])
 assert review.is_file() and review.resolve()==review and ROOT in review.parents
 assert sha(review)==approval['independent_review_sha256']
 cpu=json.loads((ROOT/'logs/target-cpu-gate.json').read_text())
 assert cpu['source_commit']==source and cpu['exit_code']==0 and cpu['platform']=='linux' and cpu['host']=='h100-5'
 cpu_log=Path(cpu['log_path']);assert ROOT in cpu_log.parents and sha(cpu_log)==cpu['log_sha256']
 assert cpu['tools_manifest_sha256']==sha(manifest_path)
 from search_r1.baselines.fresh200_contract import critical_source_hashes,verify_official_model_assets
 assert critical_source_hashes()==cpu['critical_source_sha256']
 context_path=ROOT/'logs/context-budget-audit.json'
 assert sha(context_path)==cpu['context_audit_sha256']
 context=json.loads(context_path.read_text())
 assert context['status']=='ACTUAL_TARGET_CPU_ALL_SHARED_CONTEXT_BUDGET_OK_NOT_GPU_ACCEPTANCE'
 assert context['source_commit']==source and context['source_sha256']==critical_source_hashes()
 assert {k:v['rows'] for k,v in context['records'].items()}=={'train':169028,'validation':512,'stage':64,'final':51713}
 assert all(v['maximum_initial_tokens']<=2048 and v['worst_before_fourth_request_tokens']<=4096 for v in context['records'].values())
 import search_r1.baselines.fresh200_contract as contract
 assert Path(contract.__file__).resolve()==repo/'search_r1/baselines/fresh200_contract.py'
 common=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
 assert sha(common/'common-data/manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
 actual_model_manifest=verify_official_model_assets(ROOT/'assets/qwen2.5-3b-instruct-aa8e725')
 assert context['data_manifest_sha256']==sha(common/'common-data/manifest.json')
 assert context['official_model_manifest_sha256']==actual_model_manifest
 assert {key:context[key] for key in ('force_final_answer','max_turns','max_start_length','max_rolling_context','max_response_tokens','max_observation_tokens')}=={'force_final_answer':False,'max_turns':4,'max_start_length':2048,'max_rolling_context':4096,'max_response_tokens':500,'max_observation_tokens':500}
 if phase=='formal200':
  from search_r1.baselines.fresh200_contract import verify_policy_safety_artifact
  frozen=json.loads((ROOT/'logs/frozen-policy-safety-config.json').read_text())
  assert frozen['source_commit']==source
  assert frozen['enabled'] is True and frozen['dual_clip']==3
  verify_policy_safety_artifact(frozen)
 print(json.dumps({'status':'NEW_SOURCE_READONLY_LAUNCH_GATE_RETURNED','phase':phase,'source_commit':source,'tools_manifest_sha256':sha(manifest_path),'scope':'CODE_CPU_INPUT_GATE_NOT_GPU_OR_TRAINING_ACCEPTANCE'}))
if __name__=='__main__':
 assert len(sys.argv)==2;verify(sys.argv[1])
