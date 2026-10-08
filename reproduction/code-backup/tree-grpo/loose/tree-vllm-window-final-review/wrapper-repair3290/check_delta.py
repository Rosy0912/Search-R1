"""Exact one installed-wrapper pin delta, frozen package and targeted CPU only."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
REVIEW=Path(__file__).resolve().parent
WORK=REVIEW.parents[1]
OLD=WORK/'single-tp1-window-e29b27d1-466df7c2'
NEW=WORK/'single-tp1-window-3583419c-466df7c2'
COMMIT='3583419cecee4691e72131863a61a0f6925cd7b5'
MANIFEST='20d42ab8f48ab390af6088990661d09f408d30cccb2946b407da007b8da8d48e'
PAYLOAD='4057314d38a35fe9e60ee5e5a02d8d90ec1cb6150175983fff7b268ce361e549'
BASE_SHA='49fe0eedd1667889daa63d5aa572d00f5eefbb794ea4368497f5243b02a54173'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
assert sha(NEW/'package-manifest.json')==MANIFEST
assert sha(OLD/'package-manifest.json')=='d55e3836496fd6439087ee17611f0b65eed18bb07fc722b8f5bf9c087e372084'
m0=read(OLD/'package-manifest.json');m=read(NEW/'package-manifest.json')
b0=read(OLD/'bundle-template.json');b=read(NEW/'bundle-template.json')
assert len(m['files'])==755
actual=set()
for path in NEW.rglob('*'):
    assert not path.is_symlink() and '__pycache__' not in path.parts
    if path.is_file() and path.name!='package-manifest.json':
        name=path.relative_to(NEW).as_posix();actual.add(name)
        assert sha(path)==m['files'][name]
        assert path.suffix not in {'.pt','.safetensors','.parquet','.so','.bin','.pyc','.pyo'}
assert actual==set(m['files'])
added=sorted(set(m['files'])-set(m0['files']))
changed=sorted(k for k in m['files'] if k in m0['files'] and m['files'][k]!=m0['files'][k])
assert not(set(m0['files'])-set(m['files']))
assert added==['repo/WRAPPER_REPAIR3290.md','repo/WRAPPER_REPAIR3290_CPU_RECEIPT.json',
              'repo/fixtures/vllm-model-runner-base-source.txt','repo/test_modelrunner_base_binding.py']
assert changed==['bundle-template.json','probe_package.py','repo/single_window_runner.py','repo/window_contracts.py']
old=(OLD/'repo/single_window_runner.py').read_text()
expected=old.replace("[sources['utils.py']]+list(bundle['torch_wrapper_sources'].values())",
                    "[sources['utils.py'],sources['worker/model_runner_base.py']]+list(bundle['torch_wrapper_sources'].values())").replace(
    "[bundle['vllm_installed_sources']['utils.py']]+list(bundle['torch_wrapper_sources'].values())",
    "[bundle['vllm_installed_sources']['utils.py'],bundle['vllm_installed_sources']['worker/model_runner_base.py']]+list(bundle['torch_wrapper_sources'].values())")
assert expected==(NEW/'repo/single_window_runner.py').read_text()
assert (OLD/'repo/window_contracts.py').read_text().replace(
    "'outputs.py', 'worker/model_runner.py', 'model_executor/layers/activation.py',",
    "'outputs.py', 'worker/model_runner.py', 'worker/model_runner_base.py', 'model_executor/layers/activation.py',")==(NEW/'repo/window_contracts.py').read_text()
assert (NEW/'repo/window_runtime.py').read_bytes()==(OLD/'repo/window_runtime.py').read_bytes()
assert 'Final unwrapped native implementation differs from bound source' in (NEW/'repo/window_runtime.py').read_text()
assert 'Actual native method source SHA changed' in (NEW/'repo/window_runtime.py').read_text()
fixture=NEW/'repo/fixtures/vllm-model-runner-base-source.txt'
captured=WORK/'tree-vllm-installed-hook-base-source-20261006/worker/model_runner_base.py'
assert fixture.read_bytes()==captured.read_bytes() and sha(fixture)==BASE_SHA
source_map=read(WORK/'tree-vllm-installed-hook-base-source-20261006/SOURCE_MAP.json')
assert list(source_map)==['worker/model_runner_base.py']
entry={k:source_map['worker/model_runner_base.py'][k] for k in ('path','sha256')}
assert b['vllm_installed_sources']==dict(b0['vllm_installed_sources'],**{'worker/model_runner_base.py':entry})
assert len(b['vllm_installed_sources'])==30
assert len(b['torch_wrapper_sources'])==len(b0['torch_wrapper_sources'])==1
contracts=ast.parse((NEW/'repo/window_contracts.py').read_bytes())
paths=ast.literal_eval(next(n.value for n in contracts.body if isinstance(n,ast.Assign) and
    any(isinstance(t,ast.Name) and t.id=='INSTALLED_PATHS' for t in n.targets)))
assert len(paths)==len(set(paths))==30 and set(paths)==set(b['vllm_installed_sources'])
owner=read(NEW/'repo/WRAPPER_REPAIR3290_CPU_RECEIPT.json')
assert owner['installed_wrapper_sha256']==BASE_SHA and owner['actual_affected_cpu_tests']==3
for name,wanted in owner['sources'].items():assert sha(NEW/'repo'/name)==wanted

context=WORK/'tree-vllm-window-hook-repair3290-review'
context_proof=context/'ALL_10_HOOK_CHAIN_CPU_PROOF.json'
context_receipt=context/'SOURCE_CPU_RECEIPT.json'
assert sha(context_proof)==owner['independent_all10_source_proof_sha256']=='9c722c0760da49dcb0f485b0ce4f7734ff9d41a4a5c8b10af0cc495b5f42b798'
assert sha(context_receipt)=='983841d711f6f999eab1270c688620f197492f13b2b3fefc80829086db236f83'
cp=read(context_proof);cr=read(context_receipt)
assert cp['method_count']==cr['method_count']==10 and cp['observer_attribute_count']==cr['observer_attribute_count']==35
assert cp['actual_assets_read'] is cp['GPU_executed'] is cp['remote_access'] is False
assert cr['actual_native_method_body_executed'] is False and cr['dispatch_release'] is False
for name,wanted in cr['source_maps_sha256'].items():assert sha(WORK.parent/name)==wanted
for name,record in cr['saved_installed_file_inventory'].items():assert sha(WORK.parent/name)==record['saved_bytes_sha256']
# Read and bind existing source-only context proof; no ten-chain suite rerun.
old_intent='SINGLE_TP1_WINDOW_SCHEDULER_REPAIR3286_ONCE.json'
new_intent='SINGLE_TP1_WINDOW_WRAPPER_REPAIR3290_ONCE.json'
assert (OLD/'probe_package.py').read_text().replace(old_intent,new_intent)==(NEW/'probe_package.py').read_text()
for name in ('start.sh','start.py','supervisor.py','dispatch_once.py','freeze_package.py','resource_adapter_window.py'):
    assert (OLD/name).read_bytes()==(NEW/name).read_bytes()
helpers={
    'build_tree_vllm_window_wrapper_repair3290_package.py':'a4e7733d477a4a0aea042ffe3c798eda70cfeb0540867f28df20947e6923c18f',
    'upload_tree_vllm_window_wrapper_repair3290_source.py':'de2546d6dd4ad6e20c9347a811bc27c41dde3408c41724e890d05168b6045c81'}
builder_old=(WORK/'build_tree_vllm_window_scheduler_repair3286_package.py').read_text().replace(
    'tree-vllm-window-scheduler-repair3286','tree-vllm-window-wrapper-repair3290')
builder_expected=builder_old.replace(
    "'tree-vllm-installed-engine-source-20261006','tree-vllm-installed-sampling-source-20261006']:",
    "'tree-vllm-installed-engine-source-20261006','tree-vllm-installed-sampling-source-20261006',\n                   'tree-vllm-installed-hook-base-source-20261006']:")
assert builder_expected==(WORK/'build_tree_vllm_window_wrapper_repair3290_package.py').read_text()
assert (WORK/'upload_tree_vllm_window_scheduler_repair3286_source.py').read_text().replace(
    'tree-vllm-window-scheduler-repair3286','tree-vllm-window-wrapper-repair3290')==(
    WORK/'upload_tree_vllm_window_wrapper_repair3290_source.py').read_text()
for name,wanted in helpers.items():assert sha(WORK/name)==wanted
repo=WORK/'tree-vllm-window-runner'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==COMMIT
assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)==''
for name in m['files']:
    if name.startswith('repo/'):assert sha(repo/name[5:])==m['files'][name]
assert 'Ran 3 tests' in (REVIEW/'DELTA_CPU.stderr').read_text()
assert (REVIEW/'DELTA_CPU.stderr').read_text().rstrip().endswith('OK')
prefix0='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/'+OLD.name
prefix='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/'+NEW.name
for key in ('window_inputs','window_inputs_sha256','window_contract','window_profile','diagnostic_profile',
            'config_text','config_path','identity','inventory','selection','selection_file','hf_identity','hf_index','hf_path',
            'native_logprob_sha256','installed_versions','installed_sources','torch_wrapper_sources',
            'installed_kernel_binaries','original_worker_path','original_worker_sha256','parent_state_manifest'):
    assert b[key]==b0[key],key
assert m['protocol']==m0['protocol'] and b['runner_commit']==m['runner_commit']==COMMIT
for key in ('weights_core','reviewed_resource_adapter','native_method_sources'):
    assert json.loads(json.dumps(b0[key]).replace(prefix0,prefix))==b[key],key
assert sha(NEW/'repo/vllm_window_weights.py')=='46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350'
assert sha(NEW/'resource_adapter_window.py')=='466df7c26d45342069eb0bd9dc9f79e0cead9ea1a0a0b803fd52a67398f7922b'
sys.path.insert(0,str(NEW))
import probe_package as P
assert P.digest(P.bundle_payload(b))==m['bundle_payload_sha256']==PAYLOAD
assert P.digest(P.validate_window_inputs(b['window_inputs']))==b['window_inputs_sha256']
assert P.DISPATCH_INTENT.endswith('/'+new_intent)
failure=WORK/'tree-vllm-window-final-review/ACTUAL3290_FAILURE_RECEIPT.json'
assert sha(failure)=='79663614b6ffb27d95aa20eb7f6afcfc1de1b2ef7528cff7ff4f8c26b0204c1a'
old_release=WORK/'tree-vllm-window-final-review/scheduler-repair3286/RELEASE.json'
assert sha(old_release)=='4a89af2629a07e4f4c5ef4379ac82b35609f3c3e93457a875aa1671a0e8480cd'
release=copy.deepcopy(read(old_release))
for key in ('owner_interim_false_PASS_receipt_accepted','owner_interim_false_PASS_commit'):release.pop(key,None)
release.update(package_manifest_sha256=MANIFEST,bundle_payload_sha256=PAYLOAD,runner_commit=COMMIT,
    dispatch_intent_path=P.DISPATCH_INTENT,reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='INDEPENDENT_EXACT_INSTALLED_DUMP_WRAPPER_PIN_DELTA_SOURCE_AND_PRE_DISPATCH_REVIEW_ONLY',
    predecessor_failed_job_id='3290',predecessor_failure_receipt_sha256=sha(failure),
    successor_reason='Bind exact captured worker/model_runner_base.py decorator file; strict every-wrapper SHA and final native implementation path/SHA unchanged',
    predecessor_package_manifest_sha256=sha(OLD/'package-manifest.json'),predecessor_package_and_intent_unchanged=True,
    original_intent_retry_accepted=False,full_base_136_tests_repeated=False,core44_repeated=False,delta_CPU_tests=3,
    existing_all10_source_only_proof_sha256=sha(context_proof),all10_context_suite_repeated=False,
    failed3290_runtime_method_MRO_or_chain_identity_proven=False,
    conditional_execution_limit='Once only using new fixed intent after exact same h100-9 source/readback, fresh queue, all8GPU leases/guard; outer and child Python -B/PYTHONDONTWRITEBYTECODE=1; no production training')
assert P.validate_release(m,MANIFEST,b,release) is True
for label,candidate in [('old complete release',read(old_release)),
                        ('old fixed intent',dict(release,dispatch_intent_path=P.DISPATCH_INTENT.replace(new_intent,old_intent)))]:
    try:P.validate_release(m,MANIFEST,b,candidate)
    except ValueError:pass
    else:raise AssertionError('Wrong successor release accepted: '+label)
proof=dict(status='EXACT_INSTALLED_DUMP_WRAPPER_PIN_SUCCESSOR_SOURCE_DELTA_VERIFIED',manifest_sha256=MANIFEST,
    bundle_payload_sha256=PAYLOAD,runner_commit=COMMIT,local_frozen_files_verified=755,changed_files=changed,
    added_files=added,deleted_files=[],helpers_sha256=helpers,installed_sources_before=29,installed_sources_after=30,
    added_installed_wrapper_path=entry['path'],added_installed_wrapper_sha256=BASE_SHA,
    targeted_production_changes='One INSTALLED_PATHS entry and same exact entry in two decorator lists only',
    strict_bind_method_bytes_unchanged=True,native_constructor_sampling_math_and_budget_unchanged=True,
    new_fixed_intent=P.DISPATCH_INTENT,old_intent_or_release_substitutions_refused=2,
    independent_delta_CPU_tests_run=3,independent_delta_CPU_tests_passed=3,base136_tests_repeated=False,
    core44_tests_repeated=False,existing_all10_source_only_proof_sha256=sha(context_proof),all10_context_suite_repeated=False,
    existing_all10_source_only_receipt_sha256=sha(context_receipt),failed3290_runtime_method_chain_proven=False,
    real_tensor_assets_read=False,remote_access=False,actual_gpu_execution_verified=False,actual_runtime_optimizer_writes=None,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,actual_loaded_binary_identity_proven=False,
    predecessor_actual_failure_receipt_sha256=sha(failure),
    logs_sha256={name:sha(REVIEW/name) for name in ('DELTA_CPU.stdout','DELTA_CPU.stderr')})
for name,value in [('RELEASE.json',release),('DELTA_SOURCE_CPU_PROOF.json',proof)]:
    target=REVIEW/name;assert not target.exists()
    target.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(release_sha256=sha(REVIEW/'RELEASE.json'),proof_sha256=sha(REVIEW/'DELTA_SOURCE_CPU_PROOF.json')),sort_keys=True))
