import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;WORK=HERE.parent;V9=WORK/'tree-fresh200-mu-runtime-v10';V7=WORK/'tree-fresh200-dispatch-v8';SOURCE=WORK/'tree-behavior-mu-ppo';OLD=WORK/'tree-fresh200-f7-v8-v6-context-free-review'
NOW=datetime.now(timezone.utc).isoformat();COMMIT='f7e762af365a2beb9429e4cb3b6d5f0d808d4d7d'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def put(name,value):(HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
inventory=json.loads((V9/'FILE_HASHES_V10.json').read_text());helper_inventory=json.loads((V7/'FILE_HASHES_V8.json').read_text());old_audit=json.loads((OLD/'BYTE_AUDIT.json').read_text())
assert sha(V9/'FILE_HASHES_V10.json')=='2f0f7df1b37ebbab9111cf3e0da976ff9be69e5fafcb9c5a4802ee09770ce917'
assert sha(V9/'SOURCE_CPU_RECEIPT_V10.json')=='e9af4a8daa172045125ea93973e0d40da022794a4162f40b5717f922e824b087'
assert sha(V7/'FILE_HASHES_V8.json')=='23cecbd3a1fdebad3995ba53363f2a15b9480c497aa8e6ba8a541693894b1b4b'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();clean=not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip();assert head==COMMIT and clean
wrapper={n:dict(ref(V9/n),matches=sha(V9/n)==expected) for n,expected in inventory['wrapper_files'].items()};assert len(wrapper)==15 and all(x['matches'] for x in wrapper.values())
entries={n:dict(ref(V9/n),matches_wrapper=sha(V9/n)==entry['sha256']) for n,entry in inventory['runtime_entries'].items()};assert len(entries)==4 and all(x['matches_wrapper'] for x in entries.values())
inventory_files={n:dict(ref(V9/n),matches=sha(V9/n)==entry['sha256'],matches_byte_count=(V9/n).stat().st_size==entry['bytes']) for n,entry in inventory['files'].items()};assert all(x['matches'] and x['matches_byte_count'] for x in inventory_files.values())
source_groups={}
for group in ('source_files','ancillary_source_files','source_test_files'):
    result={}
    for n,old in old_audit[group].items():
        value=sha(SOURCE/n);git_bytes=subprocess.check_output(['git','show',COMMIT+':'+n],cwd=SOURCE)
        result[n]=dict(ref(SOURCE/n),matches_git_head=value==hashlib.sha256(git_bytes).hexdigest(),matches_previous_independent_audit=value==old['sha256'])
    assert all(x['matches_git_head'] and x['matches_previous_independent_audit'] for x in result.values());source_groups[group]=result
helper_files={n:dict(ref(V7/n),matches=sha(V7/n)==expected) for n,expected in helper_inventory['helper_files'].items()};assert len(helper_files)==3 and all(x['matches'] for x in helper_files.values())
def algorithm(path):
    tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('WRAPPER_INVENTORY_ID','WRAPPER_HASHES') for t in n.targets))];return ast.dump(tree,include_attributes=False)
assert algorithm(V7/'fresh200_dispatch_common.py')==algorithm(WORK/'tree-fresh200-dispatch-v6/fresh200_dispatch_common.py')
assert all((V7/n).read_bytes()==(WORK/'tree-fresh200-dispatch-v6'/n).read_bytes() for n in ('prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py'))
frozen_v8={n:dict(ref(Path(x['path'])),matches=sha(x['path'])==x['sha256']) for n,x in old_audit['wrapper_files'].items()};assert all(x['matches'] for x in frozen_v8.values())
frozen_v6={n:dict(ref(Path(x['path'])),matches=sha(x['path'])==x['sha256']) for n,x in old_audit['helper_v6_frozen_files'].items()};assert all(x['matches'] for x in frozen_v6.values())
spec=importlib.util.spec_from_file_location('independent_helper_v8',V7/'fresh200_dispatch_common.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
external_map={'repo/reproduction/h20/retriever_local.py':WORK/'tree-grpo/reproduction/h20/retriever_local.py','tools/check_real_checkpoint.py':WORK/'check_real_checkpoint.py','tools/probe_tree_shared_retrieval.py':WORK/'probe_tree_shared_retrieval.py','tools/resource_guard_identity.py':WORK/'resource_guard_identity.py','tools/tree_shared_owned_runtime_fixed.py':WORK/'tree_shared_owned_runtime_fixed.py'}
external={n:dict(ref(path),matches=sha(path)==h.EXTERNAL_RELATIVE_FILES[n],actual_remote_path_verified=False) for n,path in external_map.items()};assert all(x['matches'] for x in external.values())
audit=dict(created_utc=NOW,git_head=head,initial_clean=clean,final_clean=clean,source_root=str(SOURCE),source_interface_count=15,wrapper_inventory=ref(V9/'FILE_HASHES_V10.json'),wrapper_owner_receipt=ref(V9/'SOURCE_CPU_RECEIPT_V10.json'),helper_inventory=ref(V7/'FILE_HASHES_V8.json'),wrapper_files=wrapper,runtime_entries=entries,inventory_files=inventory_files,helper_files=helper_files,external_source_closure=external,old_v8_files_unchanged=frozen_v8,old_v6_files_unchanged=frozen_v6,prior_sealed_review=ref(OLD/'RECEIPT.json'),prior_sealed_audit=ref(OLD/'BYTE_AUDIT.json'),helper_v8_v6_algorithm_diff=dict(common_only_two_pin_assignments=True,prepare_identical=True,submit_identical=True),actual_gpu_executed=False,actual_remote_executed=False,actual_materialized_package_review_completed=False,actual_stage_envelope_or_cpu_argv_review_completed=False,**source_groups)
put('BYTE_AUDIT.json',audit)
helper=dict(status='FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED',created_utc=NOW,code_commit=COMMIT,helper_files=helper_inventory['helper_files'],actual_gpu_accepted=False,actual_gpu_executed=False,actual_remote_executed=False,actual_slurm_executed=False,actual_slurm_submitted=False,actual_cpu_preparation_executed=False,actual_official_model_bytes_read=False,actual_official_model_or_dataset_assets_loaded=False,actual_same_node_cpu_argv_validated=False,source_clean=True,helper_source_review_completed=True,cpu_preparation_source_review_accepted=True,submit_source_review_accepted=True,wrapper_source_review_accepted=True,wrapper_inventory_sha256=sha(V9/'FILE_HASHES_V10.json'),wrapper_files=inventory['wrapper_files'],runtime_entries=entries,training_release=False,gpu_dispatch_allowed=False,zero_ready=False,source_cpu_capability_accepted=True,scope='V8 exact helper3 algorithms inherited only after same-byte/AST check against independently reviewed V6; V10 pin map checked. SOURCE acceptance still excludes actual zero dispatch. No Slurm or GPU permission.',inherited_v6_source=ref(OLD/'HELPER_V6_RECEIPT.json'),old_tests_reexecuted=False,old_tests_recounted=False,limits=['Common changes only the two V9 inventory/hash-map assignments.','Helper capability status cannot replace exact actual zero READY.','Actual package/envelope/CPU argv and zero READY are excluded.'],reviewer_context={'fork_turns':'none','inherited_parent_conversation':False})
put('HELPER_V8_RECEIPT.json',helper)
source_interface_owner=WORK/'tree-fresh200-guard-continue-owner/REPAIRED_SOURCE_INTERFACE_CPU_RECEIPT.json'
source_interface=json.loads(source_interface_owner.read_text());assert source_interface['code_commit']==COMMIT
for n,value in source_interface['source_files'].items():assert source_groups['source_files'][n]['sha256']==value
scope=dict(local_source_and_synthetic_cpu_review_only=True,local_code_only_builder_allowed=True,same_h100_9_cpu_only_preparation_allowed=True,code_package_transfer_allowed=False,actual_remote_executed=False,actual_slurm_executed=False,actual_gpu_executed=False,actual_official_model_or_dataset_assets_loaded=False,actual_package_transferred=False,actual_cpu_preparation_executed=False,actual_materialized_package_review_completed=False,actual_same_node_cpu_argv_validated=False,zero_gpu_dispatch_allowed=False,production_training_allowed=False,scope='Local code-only package and source capability; same-node CPU asset/config validation capability requires separately reviewed actual package/input/CPU argv. V10 fixes Z09 only for this frozen wrapper; final actual zero READY remains absent.')
stages={'preparation':dict(accepted=True,review_completed=True,source_cpu_capability_only=True,actual_cpu_preparation_executed=False),'fresh_mu_calibration':dict(source_review_completed=True,source_review_accepted=True,source_interface_review_accepted=True,source_wrapper_review_accepted=True,source_only=True,actual_gpu_ready=False,actual_gpu_accepted=False,actual_zero_ready=False,materialized_package_review_completed=False,blockers=['Exact materialized package/envelope/inputs/CPU argv not reviewed','Actual same-node CPU preparation, fresh-tokenizer raw preflight and zero READY not reviewed']),'fresh_mu_train200':dict(source_review_completed=False,source_review_accepted=False,implemented=False,actual_gpu_accepted=False,actual_completed_outer=0,goal_complete=False),'helper_v8':dict(source_review_completed=True,cpu_preparation_source_review_accepted=True,submit_source_review_accepted=True,actual_slurm_executed=False,actual_gpu_accepted=False)}
receipt=dict(schema_version=1,status='FRESH200_F7_V10_V8_SOURCE_CPU_CONTEXT_FREE_REVIEWED',review_status='completed',created_utc=NOW,reviewer_context={'fork_turns':'none','inherited_parent_conversation':False,'owner_PASS_not_used_as_independent_verdict':True,'previous_review_read_as_evidence':True},source={'audited_clean_head':COMMIT,'initial_clean':clean,'final_clean':clean,'source_path':str(SOURCE),'source_interface_count':15,'runtime_source_keys':len(source_groups['source_files']),'source_review_completed':True,'source_review_accepted':True},source_interface_reviewed=True,source_interface_review_accepted=True,stages=stages,execution_scope=scope,findings=[],resolved_findings=[{'id':'Z01','scope':'V10 exact allocation predicate rejects original mismatch'},{'id':'Z09','scope':'V10 actual child rejects current-cgroup mismatch before any scheduler query; original V9 stays rejected'}],independent_cpu={'distinct_new_case_groups':24,'proof':ref(HERE/'INDEPENDENT_CPU_PROOF.json'),'all_scheduler_calls_mocked':True,'original_guard_entire_script_executed_with_mock_NVML_and_proc':True,'old_owner43_not_recounted':True,'prior_source_tests_not_reexecuted_or_recounted':True},latest_human_instruction_sha256='c9e92c3f45792ffeadb000974e4bb9ee7a5cf3a6a31a194f99d95ea140317862',actual_gpu_accepted=False,actual_gpu_ready=False,actual_3b_training_accepted=False,actual_new_optimizer_calls_per_rank=0,actual_new_outer_updates=0,long_training_ready=False,production_thresholds_accepted=False,original_sampler_actor_gate_accepted=False,training_release=False,goal_complete=False,artifacts={'BYTE_AUDIT.json':sha(HERE/'BYTE_AUDIT.json'),'HELPER_V8_RECEIPT.json':sha(HERE/'HELPER_V8_RECEIPT.json'),'INDEPENDENT_CPU_PROOF.json':sha(HERE/'INDEPENDENT_CPU_PROOF.json'),'CHILD_BEFORE_QUERY_REGRESSION.json':sha(HERE/'CHILD_BEFORE_QUERY_REGRESSION.json')},limitations=['No real Slurm, GPU, official assets, remote commands or model loads.','Mock NVML/proc data exercise complete original frozen guard CPU control flow; no actual GPU identity or kernel pidfd behaviour asserted.','Owned cleanup real algorithm executed against synthetic kernel identity/cgroup interfaces; parent actual main AST exercised with mocked service/package/asset functions.','V10 new independent24 groups passed on their first execution; V9 sealed failure logs and rejected artifacts remain unchanged.','Historical5120 tokenizer and UTF8 diagnostic reports have separate V8 provenance; valid intermediate byte sequences need a newly frozen source repair. No historical/fresh native acceptance is inferred; sampling-supported rows remain0.','Any source capability acceptance is distinct from actual package/CPU argv, final zero READY, thresholds, first5 or train200 authorization.'])
put('RECEIPT.json',receipt)
# Check actual common validator retains rejected source status for CPU preparation
# and rejects it for dispatch, with no model/service/remote calls.
manifest=dict(stage=h.STAGE,training_protocol=h.PROTOCOL,code_commit=h.COMMIT,scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',official_model_path=str(h.MODEL),model_identity={'sha256':h.MODEL_ID},common_data_identity=h.DATA_ID,production_thresholds_accepted=False,actual_3b_training_accepted=False,long_training_ready=False,original_sampler_actor_gate_accepted=False,trial_thresholds=dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),probability_calibration={'environment':h.ENVIRONMENT,'source_interface_reviewed':True},files=dict(inventory['wrapper_files'],**{'source/repo/'+n:x['sha256'] for n,x in source_groups['source_files'].items()}),wrapper_inventory_sha256=h.WRAPPER_INVENTORY_ID,wrapper_files=inventory['wrapper_files'],external_files={str(h.ROOT/n):value for n,value in h.EXTERNAL_RELATIVE_FILES.items()})
h.validate_manifest(manifest)
inputs={'source_interface_receipt':{'path':'source'},'context_free_review':{'path':'context'},'context_byte_audit':{'path':'audit','sha256':sha(HERE/'BYTE_AUDIT.json')}}
bound={'source':source_interface,'context':receipt,'audit':audit}
with patch.object(h,'json_bound',lambda entry:bound[entry['path']]):
    cpu_value=h.validate_reviews(inputs,manifest,for_dispatch=False)
    source_dispatch_value=h.validate_reviews(inputs,manifest,for_dispatch=True)
    assert cpu_value['source_review_accepted'] is True and source_dispatch_value['source_review_accepted'] is True
spec_submit=importlib.util.spec_from_file_location('independent_submit_v8',V7/'submit_tree_fresh200_once.py');submit=importlib.util.module_from_spec(spec_submit);spec_submit.loader.exec_module(submit)
try:submit.validate_dispatch(h,Path('/synthetic-target'),manifest,receipt,{}, {}, {}, {'manifest':'a'*64})
except ValueError as err:message=str(err)
else:raise AssertionError('SOURCE receipt bypassed final READY')
assert 'exact independent zero-write release' in message
load_results={}
original_resolve=Path.resolve
def virtual_resolve(path,*a,**kw):
    resolved=original_resolve(path,*a,**kw)
    if resolved.is_relative_to(HERE):return h.ROOT/'source-review-fixture'/resolved.relative_to(HERE)
    return resolved
for module_name in ('prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py'):
    module_spec=importlib.util.spec_from_file_location('load_check_'+module_name.split('.')[0],V7/module_name);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    with patch.object(Path,'resolve',virtual_resolve):common=module.load_common(str(HERE/'HELPER_V8_RECEIPT.json'),sha(HERE/'HELPER_V8_RECEIPT.json'))
    assert common.WRAPPER_INVENTORY_ID==h.WRAPPER_INVENTORY_ID and common.COMMIT==COMMIT
    load_results[module_name]='EXACT_RECEIPT_AND_HELPER_BYTES_ACCEPTED_WITH_LOCAL_PATH_FIXTURE'
put('HELPER_CONTEXT_COMPATIBILITY_CPU.json',dict(status='PASS_SOURCE_CPU_ONLY_WITH_FINAL_READY_REFUSAL',v10_manifest_pin_accepted=True,cpu_preparation_source_accepted=True,source_dispatch_capability_accepted=True,final_dispatch_ready_required=True,source_review_accepted=True,source_receipt_for_dispatch_rejected=True,error=message,helper_load_common=load_results,local_path_translation_fixture=True,actual_gpu_ready=False,actual_slurm_executed=False))
print(json.dumps(dict(status=receipt['status'],receipt_sha256=sha(HERE/'RECEIPT.json'),audit_sha256=sha(HERE/'BYTE_AUDIT.json'),helper_sha256=sha(HERE/'HELPER_V8_RECEIPT.json'))))
