"""Seal independent SOURCE/CPU-only acceptance; no actual execution permission."""
import ast,hashlib,json,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
def deny(*a,**kw):raise AssertionError('Receipt sealing forbids subprocess/network')
for name in ('Popen','call','run','check_call','check_output'):setattr(subprocess,name,deny)
socket.create_connection=deny
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
V5=BASE/'tree-fresh200-train-runtime-v5';V4REVIEW=BASE/'tree-fresh200-formal-v4-02b1-context-free-review'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
delta=json.loads((HERE/'INDEPENDENT_V5_DELTA_RESULT.json').read_text())
prior=json.loads((V4REVIEW/'REVIEW.json').read_text())
assert delta['necessary_delta_checks_passed'] and not delta['unexpected_external_operations']
assert prior['status']=='REJECTED_FORMAL_V4_02B1_COMBINATION_SOURCE_CPU'
created=datetime.now(timezone.utc).isoformat()
code='02b1ea2688ff3fb032ee6699003a7e09d651f166'
context={**prior['reviewer_context'],'provenance':'Original reviewer was fork:none with isolated self-contained brief. This V5 continuation received only explicit narrow repair/rebinding instructions and owner artifact pins; no parent work conversation inherited.'}
flags={key:False for key in (
 'READY','gpu_ready','training_release','independent_dispatch_release','actual_dispatch_permission',
 'long_training_ready','materialized_package_accepted','actual_zero_accepted','actual_zero_writes_accepted',
 'bootstrap_thresholds_accepted','production_thresholds_accepted','short_thresholds_accepted',
 'actual_first5_accepted','actual_same_live_first5_accepted','actual_long200_accepted',
 'actual_official5120_accepted','actual_official_tokenizer_executed','actual_gpu_executed',
 'actual_slurm_executed','actual_slurm_submitted','actual_remote_executed',
 'actual_3b_training_accepted','original_sampler_actor_gate_accepted')}
source_bindings=prior['exact_byte_audit']['input_bindings']
bindings={**delta['exact_binding']['input_bindings'],
          'source_owner':source_bindings['source_owner'],
          'source_independent_cpu':source_bindings['source_independent_cpu']}
scope='Independent SOURCE/CPU-only acceptance of clean source02b1 plus immutable FormalV5 exact33-wrapper combination. Four-entry delta newly reviewed with necessary CPU probes; unchanged19-source and29-wrapper bytes rebound to prior actual exact V4 audit, whose V4 rejection remains immutable. No formal runtime, asset, tensor, scheduler, GPU or dispatch acceptance.'
compact=dict(schema_version=1,status='CONTEXT_FREE_FRESH200_SOURCE_ACCEPTED',
             acceptance_scope='SOURCE_CPU_ONLY',review_scope=scope,created_utc=created,
             fork_turns='none',inherited_parent_conversation=False,reviewer_context=context,
             code_commit=code,source_commit=code,source_clean=True,blocking_findings=[],
             source_cpu_review_accepted=True,source_wrapper_combination_accepted=True,
             wrapper_version='FormalV5',wrapper_count=33,source_interface_count=15,source_closure_count=19,
             changed_wrapper_count_from_rejected_v4=4,
             input_bindings=bindings,prior_v4_rejection_preserved=True,
             prior_source_cpu_receipt_sha256='c29fdcb835263b9cf03705cabe5cca80903aba72c20c07e797b8cc70e4e621c5',
             prior_source_cpu_receipt_not_relabelled=True,source_tests_rerun=False,v4_tests_rerun=False,
             necessary_v5_delta_cpu_passed=True,scope_permissions=flags,
             actual_training_outers=0,actual_optimizer_writes=0,
             actual_remote_calls=0,actual_scheduler_calls=0,gpu_nvml_calls=0,
             official_asset_loads=0,pt_deserializations=0,
             nonce=delta['nonce'],**flags)
# Evaluate precisely the immutable real intake's compact-source AST branch.
# No fake calibration-ready/threshold records are created or passed to intake.
tree=ast.parse((V5/'production_contract.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observed_production_inputs')
assign=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='compact' for t in n.targets))
expression=compile(ast.Expression(assign.value),'actual-v5-production-source-compact-branch','eval')
assert eval(expression,{'review':compact,'package':{'code_commit':code}}) is True
negative=dict(compact,status='REJECTED_FORMAL_V4_02B1_COMBINATION_SOURCE_CPU',blocking_findings=[{'id':'unresolved'}])
assert eval(expression,{'review':negative,'package':{'code_commit':code}}) is False
compact['real_intake_source_branch_verified']=True
compact['actual_zero_threshold_dispatch_intake_not_exercised']=True
def write(name,value):
    with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
review=dict(schema_version=1,status='ACCEPTED_FORMAL_V5_02B1_COMBINATION_SOURCE_CPU',
            created_utc=created,review_scope=scope,acceptance_scope='SOURCE_CPU_ONLY',
            code_commit=code,reviewer_context=context,blocking_findings=[],scope_permissions=flags,
            exact_binding=delta['exact_binding'],actual_delta_cpu_evidence=delta,
            source_clean_before=True,source_clean_after=True,
            source_git_head_before=code,source_git_head_after=code,
            prior_v4_receipt_sha256='4e618c6daee703dd84db9cbbb57ec87d8f0281287220aca85a166457d7194619',
            prior_v4_review_sha256='d39fb77d7be107317b08d8a56f0130eafcb18712f53ca5419b954d3a382b5bf7',
            v4_rejection_unchanged=True,
            prior_v4_cpu_evidence_scope='9 distinct changed-interface methods, tiny actual source mu/checker registration, semantic drift rejection, and two entry failure reproductions. Those tests are not rerun or renamed as V4 acceptance.',
            resolved_findings=[
              dict(id='FORMAL_V4_02B1_P01',resolution='All three actual preflight entries call explicit guarded_preflight_main via -c, which verifies actual cgroup, installs guard and calls canonical main. Exact argv/reuse/inherited exec semantics and current-cgroup recheck passed necessary CPU probes.'),
              dict(id='FORMAL_V4_02B1_P02',resolution='Both actual startup required import and install exceptions produce fatal SystemExit before any nonce payload executes. Two strictly allowlisted local tiny Python children proved fatal startup; no GPU/scheduler/network operations.')],
            old_v3_blockers_closed_at_source_cpu_level=True,
            unchanged_contracts=prior['unchanged_contracts'],
            original_sampler_profile=dict(native_sampler_actor_gate_accepted=False,original_tolerance=0.01,target_old_policy_kl=0.2,mu_centered_alternative_requires_explicit_optin=True,production_thresholds_accepted=False),
            source_compact_schema_probe=dict(method='Actual immutable compact branch AST evaluated against the real independent CPU receipt',accepted_source_branch=True,rejected_v4_branch_refused=True,actualzero_threshold_dispatch_intake_not_exercised=True),
            limitations=[
              'Official same-node tokenizer class/path/backend compatibility, official unfiltered5120, actualzero and independent bootstrap thresholds remain unverified here.',
              'All scheduler/GPU controls exercised through CPU identity fixtures or sentinels; no actual allocation, resource admission, training, first5 or200 acceptance.',
              'Source/CPU receipt can satisfy only the source-review intake field. Actualzero/threshold/package/dispatch/first5/final gates remain separate and false.',
              'Initial independent probe helper path-parenthesizing typo occurred before child execution; original error record retained. Source and wrapper bytes were not edited.' ],
            operations=dict(local_tiny_python_children=2,source_wrapper_main_output_edits=0,
                            actual_gpu_queries=0,actual_scheduler_calls=0,remote_calls=0,
                            official_asset_loads=0,pt_deserializations=0,training_outers=0,optimizer_writes=0,
                            source_tests_rerun=False,v4_tests_rerun=False,full_suite_rerun=False,gloo_rerun=False),
            artifact_bindings={name:dict(path=str(HERE/name),sha256=sha(HERE/name)) for name in ('INDEPENDENT_V5_DELTA_PROBES.py','INDEPENDENT_V5_DELTA_RESULT.json','INITIAL_REVIEWER_PROBE_ERROR.json')})
write('REVIEW.json',review)
compact['detailed_review_path']=str(HERE/'REVIEW.json');compact['detailed_review_sha256']=sha(HERE/'REVIEW.json')
write('SOURCE_CONTEXT_FREE_FORMAL_REVIEW.json',compact)
receipt={**compact,'compact_source_review_path':str(HERE/'SOURCE_CONTEXT_FREE_FORMAL_REVIEW.json'),
         'compact_source_review_sha256':sha(HERE/'SOURCE_CONTEXT_FREE_FORMAL_REVIEW.json'),
         'review_path':str(HERE/'REVIEW.json'),'review_sha256':sha(HERE/'REVIEW.json')}
write('RECEIPT.json',receipt)
hashes={name:sha(HERE/name) for name in ('RECEIPT.json','REVIEW.json','SOURCE_CONTEXT_FREE_FORMAL_REVIEW.json')}
write('BYTE_HASHES.json',hashes)
print(json.dumps(dict(status=compact['status'],acceptance_scope='SOURCE_CPU_ONLY',blocking_findings=0,hashes=hashes)))
