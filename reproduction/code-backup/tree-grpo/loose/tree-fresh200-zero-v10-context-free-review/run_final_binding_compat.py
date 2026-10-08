"""Exercise final saved JSONs through actual source validators; no node execution."""
import hashlib,importlib.util,json
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;HELPERS=WORK/'tree-fresh200-dispatch-v8'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):
    spec=importlib.util.spec_from_file_location('independent_final_'+name.split('.')[0],HELPERS/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
h=load('fresh200_dispatch_common.py');submit=load('submit_tree_fresh200_once.py');prepare=load('prepare_tree_fresh200_package.py')
audit=json.loads((HERE/'BYTE_AUDIT.json').read_text());context=json.loads((HERE/'RECEIPT.json').read_text());inventory=json.loads((WORK/'tree-fresh200-mu-runtime-v10/FILE_HASHES_V10.json').read_text())
owner=WORK/'tree-fresh200-guard-continue-owner/REPAIRED_SOURCE_INTERFACE_CPU_RECEIPT.json'
manifest=dict(stage=h.STAGE,training_protocol=h.PROTOCOL,code_commit=h.COMMIT,scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',official_model_path=str(h.MODEL),model_identity={'sha256':h.MODEL_ID},common_data_identity=h.DATA_ID,production_thresholds_accepted=False,actual_3b_training_accepted=False,long_training_ready=False,original_sampler_actor_gate_accepted=False,trial_thresholds=dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.),probability_calibration={'environment':h.ENVIRONMENT,'source_interface_reviewed':True},files=dict(inventory['wrapper_files'],**{'source/repo/'+name:x['sha256'] for name,x in audit['source_files'].items()}),wrapper_inventory_sha256=h.WRAPPER_INVENTORY_ID,wrapper_files=inventory['wrapper_files'],external_files={str(h.ROOT/name):value for name,value in h.EXTERNAL_RELATIVE_FILES.items()})
inputs={key:{'path':str(path),'sha256':sha(path)} for key,path in {'source_interface_receipt':owner,'context_free_review':HERE/'RECEIPT.json','context_byte_audit':HERE/'BYTE_AUDIT.json'}.items()}
old_resolve=Path.resolve
def virtual_resolve(path,*a,**kw):
    value=old_resolve(path,*a,**kw)
    if value.is_relative_to(HERE):return h.ROOT/'independent-source-review-fixture'/value.relative_to(HERE)
    if value==owner:return h.ROOT/'independent-source-review-fixture/owner-source-interface.json'
    return value
h.validate_manifest(manifest)
with patch.object(Path,'resolve',virtual_resolve):
    cpu=h.validate_reviews(inputs,manifest,for_dispatch=False)
    try:h.validate_reviews(inputs,manifest,for_dispatch=True)
    except ValueError as err:source_error=str(err)
    else:raise AssertionError('Rejected zero submit capability passed source dispatch')
    load_errors={}
    for name,loader in [('prepare',prepare),('submit',submit)]:
        try:loader.load_common(str(HERE/'HELPER_V8_RECEIPT.json'),sha(HERE/'HELPER_V8_RECEIPT.json'))
        except ValueError as err:load_errors[name]=str(err)
        else:raise AssertionError('Current rejection receipt was accepted by load_common')
assert cpu['source_review_accepted'] is False and context['stages']['fresh_mu_calibration']['source_wrapper_review_accepted'] is True
assert 'source review remains blocked' in source_error
assert all('Missing independent helper source review' in value for value in load_errors.values())
try:submit.validate_dispatch(h,Path('/fixture-target'),manifest,context,{}, {}, {}, {'manifest':'a'*64})
except ValueError as err:failure=str(err)
else:raise AssertionError('Actual submit accepted SOURCE receipt as final READY')
assert 'exact independent zero-write release' in failure
for name,digest in context['artifacts'].items():assert sha(HERE/name)==digest,name
for group in ['source_files','ancillary_source_files','source_test_files','wrapper_files','runtime_entries','helper_files','external_source_closure','old_v8_files_unchanged','old_v6_files_unchanged','old_v9_wrapper_unchanged','old_v7_helper_unchanged']:
    for name,x in audit[group].items():assert sha(x['path'])==x['sha256'],(group,name)
compatibility=dict(status='PASS_FINAL_WRAPPER_ACCEPTANCE_AND_ZERO_HELPER_REJECTION_PRESERVED_CPU_ONLY',actual_bound_json_read=True,local_path_translation_fixture=True,wrapper_source_review_accepted=True,combined_zero_source_review_accepted=False,cpu_preparation_review_can_read_completed_blocked_context=True,source_dispatch_refused=True,source_dispatch_error=source_error,helper_load_common_refused=True,helper_load_common_errors=load_errors,source_receipt_rejected_by_final_submit=True,error=failure,actual_remote_executed=False,actual_slurm_executed=False,actual_gpu_executed=False,actual_gpu_ready=False,actual_threshold_accepted=False,training_release=False)
(HERE/'HELPER_CONTEXT_COMPATIBILITY_CPU.json').write_text(json.dumps(compatibility,indent=2,sort_keys=True)+'\n')
proof=dict(compatibility,receipt_sha256=sha(HERE/'RECEIPT.json'),audit_sha256=sha(HERE/'BYTE_AUDIT.json'),runtime_sha256=sha(HERE/'WRAPPER_V10_RECEIPT.json'),helper_sha256=sha(HERE/'HELPER_V8_RECEIPT.json'),source_interface_receipt_sha256=sha(owner))
(HERE/'FINAL_BINDING_COMPATIBILITY_CPU.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
print(json.dumps(proof))
