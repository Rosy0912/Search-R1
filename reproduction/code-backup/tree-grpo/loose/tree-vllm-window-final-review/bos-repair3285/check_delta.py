"""Exact BOS metadata repair delta only; no remote, GPU, assets, or full-suite rerun."""
import ast
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
REVIEW=Path(__file__).resolve().parent
WORK=REVIEW.parents[1]
OLD=WORK/'single-tp1-window-3d8c6f6d-466df7c2'
NEW=WORK/'single-tp1-window-ca1f2af0-466df7c2'
MANIFEST_SHA='36bfac94a3bc6fa5ed344d21d0110ab08a01cc098adebbccc42c538147a1f66c'
PAYLOAD_SHA='c6eceea1e0d4a1868f9a948a1c6ad4a05f357a3a441fffaa8c595815b7e0d0f5'
COMMIT='ca1f2af03c501862885cdfb55a2b79f6f4fae55f'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
assert sha(NEW/'package-manifest.json')==MANIFEST_SHA
m0=read(OLD/'package-manifest.json');m=read(NEW/'package-manifest.json')
b0=read(OLD/'bundle-template.json');b=read(NEW/'bundle-template.json')
assert sha(OLD/'package-manifest.json')=='f3e43a57f7164cc987d1ad211d617126d7bc5ece5450e4e13d365f405f15d6e1'
assert len(m['files'])==747
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
assert not (set(m0['files'])-set(m['files']))
assert added==['repo/BOS_REPAIR3285_CPU_RECEIPT.json','repo/fixtures/hf110-config-metadata.json']
assert changed==['bundle-template.json','probe_package.py','repo/WINDOW_RUNNER_CONTRACT.md',
                 'repo/prepare_window_inputs.py','repo/test_single_window_contracts.py']

old_source=(OLD/'repo/prepare_window_inputs.py').read_text()
new_source=(NEW/'repo/prepare_window_inputs.py').read_text()
old_tree=ast.parse(old_source);new_tree=ast.parse(new_source)
old_function=next(x for x in old_tree.body if isinstance(x,ast.FunctionDef) and x.name=='original_tokenizer_ids')
new_function=next(x for x in new_tree.body if isinstance(x,ast.FunctionDef) and x.name=='original_tokenizer_ids')
assert len(old_function.body)==4 and len(new_function.body)==3
removed=old_function.body.pop(2)
assert isinstance(removed,ast.Expr) and isinstance(removed.value,ast.Call)
assert ast.dump(old_tree,include_attributes=False)==ast.dump(new_tree,include_attributes=False)
assert 'Bound parent HF token ID fields missing' in ast.unparse(removed)
assert (NEW/'repo/single_window_runner.py').read_bytes()==(OLD/'repo/single_window_runner.py').read_bytes()
assert "{k:getattr(tokenizer,k) for k in ('pad_token_id','eos_token_id','bos_token_id')} == prepared['tokenizer_metadata']['ids']" in (NEW/'repo/single_window_runner.py').read_text()
assert sha(NEW/'repo/fixtures/hf110-config-metadata.json')=='879f8585cd1807a067222aa146212f59537cc498f8f9e9d3adc8c791739a7fd0'
assert (NEW/'repo/fixtures/hf110-config-metadata.json').read_bytes()==(WORK.parent/'outputs/tree-instruct-300/PPO_HF110_ACTUAL_CONFIG_METADATA.json').read_bytes()
fixture=read(NEW/'repo/fixtures/hf110-config-metadata.json')
assert 'bos_token_id' not in fixture

old_intent='SINGLE_TP1_WINDOW_AFTER_3282_ONCE.json'
new_intent='SINGLE_TP1_WINDOW_BOS_REPAIR3285_ONCE.json'
assert (OLD/'probe_package.py').read_text().replace(old_intent,new_intent)==(NEW/'probe_package.py').read_text()
for name in ('start.sh','start.py','supervisor.py','dispatch_once.py','freeze_package.py','resource_adapter_window.py'):
    assert (OLD/name).read_bytes()==(NEW/name).read_bytes()
helpers={
 'build_tree_vllm_window_bos_repair3285_package.py':'fc0e6b090036638c46a49e920aed17e5d3595652c1c4494964f4ac3c18b223cd',
 'upload_tree_vllm_window_bos_repair3285_source.py':'32f8b6ea7093e58953c9a554a75a2115893540c41f1d82b2e7e937cb2f319c91'}
assert (WORK/'build_tree_vllm_window_package.py').read_text().replace(
    'tree-vllm-window-owned/launcher','tree-vllm-window-bos-repair3285/launcher').replace(
    'tree-vllm-window-owned/PACKAGE_BUILD.json','tree-vllm-window-bos-repair3285/PACKAGE_BUILD.json') == (
    WORK/'build_tree_vllm_window_bos_repair3285_package.py').read_text()
assert (WORK/'upload_tree_vllm_window_source.py').read_text().replace(
    'tree-vllm-window-owned/SOURCE_UPLOAD_PREPARED.json','tree-vllm-window-bos-repair3285/SOURCE_UPLOAD_PREPARED.json') == (
    WORK/'upload_tree_vllm_window_bos_repair3285_source.py').read_text()
for name,wanted in helpers.items():assert sha(WORK/name)==wanted

repo=WORK/'tree-vllm-window-runner'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==COMMIT
assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)==''
for name in m['files']:
    if name.startswith('repo/'):assert sha(repo/name[5:])==m['files'][name]
assert 'Ran 2 tests' in (REVIEW/'DELTA_CPU.stderr').read_text() and (REVIEW/'DELTA_CPU.stderr').read_text().rstrip().endswith('OK')

prefix0='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/'+OLD.name
prefix='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/'+NEW.name
# Source paths and runner commit change, but input/protocol/model/config do not.
for key in ('window_inputs','window_inputs_sha256','window_contract','window_profile','diagnostic_profile',
            'config_text','config_path','identity','inventory','selection','selection_file','hf_identity','hf_index','hf_path',
            'native_logprob_sha256','installed_versions','installed_sources','vllm_installed_sources',
            'torch_wrapper_sources','installed_kernel_binaries','original_worker_path','original_worker_sha256','parent_state_manifest'):
    assert b[key]==b0[key],key
assert b['runner_commit']==m['runner_commit']==COMMIT
for key in ('weights_core','reviewed_resource_adapter','native_method_sources'):
    assert json.loads(json.dumps(b0[key]).replace(prefix0,prefix))==b[key],key
assert sha(NEW/'repo/vllm_window_weights.py')=='46ef590201bfd34014ca5efdafd44a20907a242b5dc520c38ebfb8c5e206a350'
sys.path.insert(0,str(NEW))
import probe_package as P
assert P.digest(P.bundle_payload(b))==m['bundle_payload_sha256']==PAYLOAD_SHA
assert P.digest(P.validate_window_inputs(b['window_inputs']))==b['window_inputs_sha256']
assert P.DISPATCH_INTENT.endswith('/'+new_intent)
failure=WORK/'tree-vllm-window-final-review/ACTUAL3285_FAILURE_RECEIPT.json'
assert sha(failure)=='ec486daca1addfbd5ee0387bb682bae5b3750a9175ce802495d94ebf25e575d7'
release=copy.deepcopy(read(WORK/'tree-vllm-window-final-review/RELEASE.json'))
release.update(package_manifest_sha256=MANIFEST_SHA,bundle_payload_sha256=PAYLOAD_SHA,runner_commit=COMMIT,
    dispatch_intent_path=P.DISPATCH_INTENT,reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='INDEPENDENT_OMITTED_DEFAULT_BOS_DELTA_SOURCE_AND_PRE_DISPATCH_REVIEW_ONLY',
    predecessor_failed_job_id='3285',predecessor_failure_receipt_sha256=sha(failure),
    successor_reason='Exact metadata compatibility repair: omitted BOS defaults to None; mandatory PAD/EOS integer and actual tokenizer equality unchanged',
    predecessor_package_manifest_sha256=m0['files'].get('package-manifest.json',sha(OLD/'package-manifest.json')),
    predecessor_package_and_intent_unchanged=True,original_intent_retry_accepted=False,
    full_base_136_tests_repeated=False,core44_repeated=False,delta_CPU_tests=2)
assert P.validate_release(m,MANIFEST_SHA,b,release) is True
for label,candidate in [('old complete release',read(WORK/'tree-vllm-window-final-review/RELEASE.json')),
                         ('old fixed intent',dict(release,dispatch_intent_path=P.DISPATCH_INTENT.replace(new_intent,old_intent)))]:
    try:P.validate_release(m,MANIFEST_SHA,b,candidate)
    except ValueError:pass
    else:raise AssertionError('Wrong successor release accepted: '+label)
for name,value in [('RELEASE.json',release),('DELTA_SOURCE_CPU_PROOF.json',dict(
    status='EXACT_OMITTED_DEFAULT_BOS_SUCCESSOR_SOURCE_DELTA_VERIFIED',manifest_sha256=MANIFEST_SHA,
    bundle_payload_sha256=PAYLOAD_SHA,runner_commit=COMMIT,local_frozen_files_verified=747,
    changed_files=changed,added_files=added,deleted_files=[],helpers_sha256=helpers,
    altered_production_AST_statements=1,actual_HF_config_fixture_sha256=sha(NEW/'repo/fixtures/hf110-config-metadata.json'),
    actual_tokenizer_equality_check_unchanged=True,unchanged_allocation_resource_native_budgets=True,
    new_fixed_intent=P.DISPATCH_INTENT,old_intent_or_release_substitutions_refused=2,
    independent_delta_CPU_tests_run=2,independent_delta_CPU_tests_passed=2,
    base136_tests_repeated=False,core44_tests_repeated=False,real_tensor_assets_read=False,
    remote_access=False,actual_gpu_execution_verified=False,actual_runtime_optimizer_writes=None,
    probability_accepted=False,training_accepted=False,GPU_capacity_proven=False,
    actual_loaded_binary_identity_proven=False,
    predecessor_actual_failure_receipt_sha256=sha(failure),
    logs_sha256={name:sha(REVIEW/name) for name in ('DELTA_CPU.stdout','DELTA_CPU.stderr')}))]:
    target=REVIEW/name;assert not target.exists()
    target.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(release_sha256=sha(REVIEW/'RELEASE.json'),proof_sha256=sha(REVIEW/'DELTA_SOURCE_CPU_PROOF.json')),sort_keys=True))
