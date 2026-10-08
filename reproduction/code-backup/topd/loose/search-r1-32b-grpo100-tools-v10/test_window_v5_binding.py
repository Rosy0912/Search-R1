"""Real adapter constructor/source refs only; no NVML/GPU execution."""
import importlib.util,json,pathlib,threading,types
import pytest
from window_guardian import Guardian,validate_policy_review,RUNTIME,GUARD,OLD_GUARD,MANIFEST,sha
HERE=pathlib.Path(__file__).resolve().parent
WINDOW=HERE.parent/'qwen25-32b-window-safety-candidate-v5'

def arrange(tmp_path):
 root=tmp_path;logs=root/'logs';logs.mkdir()
 def ref(name,content):
  p=logs/name;p.write_text(content);return {'path':str(p),'sha256':sha(p)}
 old={'status':'OBSERVED_PATHS_PASS_NOT_FOUND_BRANCH_UNOBSERVED','source_sha256':RUNTIME,'adapter_sha256':OLD_GUARD,'monitor_policy_source_initial_allocation_eligible':True,'approval_scope':'MONITOR_POLICY_SOURCE_AND_OBSERVED_PATHS_ONLY_INITIAL_ALLOCATION_NOT_FORMAL100_NOT_TEACHER_NOT_TOPD','raw_refs':[ref('old-raw.json','{"scope":"CPU_FIXTURE_NOT_REAL_EVIDENCE"}') ]}
 oldref=ref('old-review.json',json.dumps(old))
 manifest=ref('source-manifest.json',(WINDOW/'files.json').read_text());assert manifest['sha256']==MANIFEST
 review={'schema':'WINDOW_V5_CPU_REPLAY_POLICY_REVIEW_V1','status':'CPU_REPLAY_POLICY_PASS_NOT_GPU_TELEMETRY_APPROVAL','approval_scope':'SOURCE_AND_NATIVE_CPU_CONTRACTS_ONLY','actual_gpu_telemetry_approval':False,'runtime_source_sha256':RUNTIME,'adapter_source_sha256':GUARD,'policy':{'ordinary_attempt_cap':6,'explicit_low_high_attempt_cap':9,'call_seconds':30,'startup_seconds':120},'source_manifest':manifest,'native_cpu_result':ref('cpu-result.json','{"scope":"CPU_FIXTURE_NOT_ACTUAL_NATIVE"}'),'native_cpu_log':ref('cpu-log.txt','CPU_FIXTURE'),'independent_source_review':ref('source-review.md','CPU_FIXTURE'),'inherited_runtime_observation_ref':oldref}
 return root,review,{'actual_independent_native_review':ref('new-review.json',json.dumps(review))}

def test_production_review_schema_and_old_approval_rejection(tmp_path):
 root,review,policy=arrange(tmp_path)
 assert validate_policy_review(root,policy)['actual_gpu_telemetry_approval'] is False
 p=pathlib.Path(policy['actual_independent_native_review']['path']);review['status']='OBSERVED_PATHS_PASS_NOT_FOUND_BRANCH_UNOBSERVED';p.write_text(json.dumps(review));policy['actual_independent_native_review']['sha256']=sha(p)
 with pytest.raises(ValueError,match='scope'):validate_policy_review(root,policy)

def test_review_raw_byte_drift_rejected(tmp_path):
 root,review,policy=arrange(tmp_path)
 pathlib.Path(review['native_cpu_log']['path']).write_text('DRIFT')
 with pytest.raises(ValueError,match='raw drift'):validate_policy_review(root,policy)

def test_actual_guardian_begin_phase_real_pinned_constructor(tmp_path):
 spec=importlib.util.spec_from_file_location('v5_actual_constructor_cpu',WINDOW/'startup_guard.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
 assert sha(WINDOW/'startup_guard.py')==GUARD and sha(WINDOW/'runtime_safety.py')==RUNTIME
 # The actual constructor checks actual pinned source bytes; its GPU functions are never called.
 safety=types.SimpleNamespace(__file__=str(WINDOW/'runtime_safety.py'),active_gate=lambda *a:None,nvml_snapshot=lambda:None,collect_process_windows=lambda *a:None,pid_identity=lambda pid:None)
 owner=Guardian.__new__(Guardian);owner.lock=threading.RLock();owner.guard=None;owner.adapter=adapter;owner.safety=safety;owner.policy={'directory':str(WINDOW)};owner.evidence=tmp_path
 owner.begin_phase('CPU_ONLY_CONSTRUCTOR')
 assert owner.guard.low_high_attempts==9 and owner.guard.max_attempts==6 and owner.guard.retry_call_seconds==30 and owner.guard.startup_seconds==120
 owner.guard.close_startup()
