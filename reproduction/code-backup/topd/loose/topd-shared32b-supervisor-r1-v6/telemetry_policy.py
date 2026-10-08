"""Exact existing WindowV5 review consumer; not GPU telemetry approval."""
import hashlib,json,pathlib
ORIGINAL_MODULE_SHA='f5fe9d6f9b25dad062a53efb445a3f294df772b93da4ecb8983d08e926c9adc2'
RUNTIME='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'

GUARD='65165ac7c8f5c9042ba254fd36d26e222096e6764fa83c167061e25530d5ad39'

MANIFEST='c8506109114ac45d2f18ac7f4eaf7c1f4d411ce8c0fabfabb0e78d82f689dac0'

OLD_GUARD='d340c77f54c8c744816da416c91016e59bd5a21bd42262b2ae672a3432f6588b'

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def validate_policy_review(root,policy):
 root=pathlib.Path(root).resolve()
 def read(ref):
  p=pathlib.Path(ref['path']).resolve()
  if root not in p.parents or sha(p)!=ref['sha256']:raise ValueError('Actual telemetry evidence drift')
  return json.loads(p.read_text())
 review=read(policy['actual_independent_native_review'])
 if review.get('schema')!='WINDOW_V5_CPU_REPLAY_POLICY_REVIEW_V1' or review.get('status')!='CPU_REPLAY_POLICY_PASS_NOT_GPU_TELEMETRY_APPROVAL' or review.get('approval_scope')!='SOURCE_AND_NATIVE_CPU_CONTRACTS_ONLY' or review.get('actual_gpu_telemetry_approval') is not False or review.get('runtime_source_sha256')!=RUNTIME or review.get('adapter_source_sha256')!=GUARD:raise ValueError('WindowV5 CPU review identities/scope differ')
 if review.get('policy')!={'ordinary_attempt_cap':6,'explicit_low_high_attempt_cap':9,'call_seconds':30,'startup_seconds':120}:raise ValueError('WindowV5 exact measurement budgets differ')
 for name in ('source_manifest','native_cpu_result','native_cpu_log','independent_source_review','inherited_runtime_observation_ref'):
  ref=review[name];p=pathlib.Path(ref['path']).resolve()
  if root not in p.parents or sha(p)!=ref['sha256']:raise ValueError('Actual telemetry raw drift '+name)
 if review['source_manifest']['sha256']!=MANIFEST:raise ValueError('WindowV5 reviewed manifest differs')
 old=read(review['inherited_runtime_observation_ref'])
 if old.get('status')!='OBSERVED_PATHS_PASS_NOT_FOUND_BRANCH_UNOBSERVED' or old.get('source_sha256')!=RUNTIME or old.get('adapter_sha256')!=OLD_GUARD or old.get('monitor_policy_source_initial_allocation_eligible') is not True or old.get('approval_scope')!='MONITOR_POLICY_SOURCE_AND_OBSERVED_PATHS_ONLY_INITIAL_ALLOCATION_NOT_FORMAL100_NOT_TEACHER_NOT_TOPD':raise ValueError('Inherited unchanged-runtime observation differs')
 for ref in old['raw_refs']:
  p=pathlib.Path(ref['path']).resolve()
  if root/'logs' not in p.parents or sha(p)!=ref['sha256']:raise ValueError('Inherited actual telemetry raw drift')
 return review
