"""Seal honest independent V4 rejection from already-completed local evidence."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
V4=BASE/'tree-fresh200-train-runtime-v4'
SOURCE=BASE/'tree-behavior-mu-utf8-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((HERE/'EXACT_BYTE_AUDIT_RESULT.json').read_text())
cpu=json.loads((HERE/'CPU_REVIEW_PROBES_RESULT.json').read_text())
assert cpu['changed_interface_tests']['passed']
assert cpu['independent_current_process_mu_registration']['passed']
assert cpu['module_main_guard_bypass']['reproduced']
assert cpu['startup_installation_exception_failopen']['reproduced']
assert audit['all_pinned_input_sha_verified'] and audit['byte_closure_verified']
created=datetime.now(timezone.utc).isoformat()
context=dict(fork_turns='none',inherited_parent_conversation=False,
             inherited_parent_work_conversation_observed=False,isolated_task_brief_only=True,
             provenance='This reviewer received an isolated self-contained task brief; no parent work conversation was inherited. Supplemental static child was also spawned with fork_turns=none.')
permissions={key:False for key in (
 'READY','gpu_ready','training_release','independent_dispatch_release','actual_dispatch_permission',
 'long_training_ready','materialized_package_accepted','source_wrapper_combination_accepted',
 'new_source_formal_combination_review_accepted','actual_zero_accepted','bootstrap_thresholds_accepted',
 'production_thresholds_accepted','short_thresholds_accepted','actual_first5_accepted',
 'actual_same_live_first5_accepted','actual_long200_accepted','actual_official5120_accepted',
 'actual_official_tokenizer_executed','actual_gpu_executed','actual_slurm_executed',
 'actual_slurm_submitted','actual_remote_executed','original_sampler_actor_gate_accepted')}
def loc(path,line):return dict(path=str(path),line=line)
findings=[
 dict(id='FORMAL_V4_02B1_P01',priority=1,category='module_cli_current_cgroup_query_bypass',
      title='Module CLI redefines the GPU snapshot after canonical-module guard installation',
      consequence='The actual launch and inherited lease CLI can request nvidia-smi without an immediately preceding actual own host/job/cgroup check. Canonical-module guard success does not cover the __main__ function.',
      evidence_method='Actual frozen preflight executed only under runpy with all process/network/exec calls failclosed and GPU check_output sentinel; wrong-host canonical call refused with zero requests, __main__ hit the intercepted nvidia-smi argv.',
      locations=[loc(V4/'launch.sbatch',34),loc(V4/'run_tree.sh',45),loc(V4/'run_tree.sh',47),
                 loc(V4/'sitecustomize.py',3),loc(V4/'slurm_identity.py',75),
                 loc(SOURCE/'search_r1/baselines/preflight.py',142),loc(SOURCE/'search_r1/baselines/preflight.py',169)],
      correction_required='Use an explicit guarded canonical preflight entry that checks actual current cgroup and installs guards before canonical preflight.main(), without runpy __main__ redefinition; keep argv/reuse-inherited/lease behavior.'),
 dict(id='FORMAL_V4_02B1_P02',priority=1,category='required_startup_guard_exception_failopen',
      title='Required sitecustomize guard installation errors are suppressed by Python startup',
      consequence='A missing or failed required guard import/installation can print an error and allow the production interpreter to continue. Supervisor later installation cannot establish query order for preceding launch preflight.',
      evidence_method='Actual stdlib site.execsitecustomize loaded actual V4 sitecustomize while only the required slurm_identity import was injected to fail with ValueError; handler returned and printed the error. No subprocess, scheduler, GPU or assets were touched.',
      locations=[loc(V4/'sitecustomize.py',3),loc(V4/'sitecustomize.py',5),loc(V4/'sitecustomize.py',6),loc(V4/'supervisor.py',68)],
      correction_required='Make every required TREE_INSTRUCT_TOOLS guard bootstrap failure terminate startup through a fatal BaseException such as SystemExit, and verify fatal behavior with CPU-only probes.')]
contracts=[
 dict(contract='Official aa8e725 actor/reference fresh; no inherited weights/Adam/scheduler/cursor; fixed commondata SHA and train169028/dev512/final51713',
      basis='Exact unchanged configuration/calibration/intake and source closure bytes; no official assets loaded; actual fresh initialization remains pending.'),
 dict(contract='512 questions x five real Tree leaves [3,2], DP8/SP1, mini128, actor16global/2local, old/ref32global/4local, epochs1, LR1e-6 and 57 credited-outer warmup, entropy/refKL0, lowvariance clip0.2/grad1',
      basis='Exact unchanged V3 specification, source math/guard/accounting interfaces and configuration checks; actual performance or numeric compatibility not claimed.'),
 dict(contract='All20 opportunities per rank plus at least1 successful Adam return earn completed outer/scheduler credit; allskip/abandon receive none; partial writes remain physically retained and separately reported',
      basis='Exact unchanged attempt_accounting/production_contract/train_monitor/tree_guard_continue/dp_actor/FSDP accounting; no rollback/exactresume claim.'),
 dict(contract='DualClip negative advantage3, sampled-policyKL0.2 with actualzero-derived negative-tail and repeat limits, finite before exp30; original native sampler versus actor0.01 remains unaccepted; explicit mu alternative never rebrands the original gate',
      basis='Unchanged source guards and production threshold intake; actualzero/threshold receipts absent and never fabricated.'),
 dict(contract='Same living production instance must complete first5 and receive external fork:none actual acceptance before6..200; saves5/50/100/150/200, dev every5, fullfinal51713/HF/fullstate/mu/cost/cleanup',
      basis='Exact unchanged external approval publishing and live supervisor gate; no actual five-step or continuation acceptance.'),
 dict(contract='h100-9 same-node retrieval/Ray/training/assets/storage and loopback, 8GPU exclusive32CPU256G UUID double15 foreign occupancy thresholds, protected original8 PIDs, immutable script bytes stdin, no foreign kill/cancel/hot edits',
      basis='Pinned wrapper/source/external-chain inspection; query ordering remains rejected by P01/P02. No allocation or dispatch performed.')]
scope='Independent local exact-byte/source-interface and necessary changed-CPU review of immutable FormalV4 plus clean source02b1. No full-suite/Gloo/official5120/official assets/PT/remote/scheduler/GPU/formal execution.'
review=dict(schema_version=1,status='REJECTED_FORMAL_V4_02B1_COMBINATION_SOURCE_CPU',
            disposition='REJECT_COMBINATION_SOURCE_CPU',created_utc=created,review_scope=scope,
            code_commit='02b1ea2688ff3fb032ee6699003a7e09d651f166',reviewer_context=context,
            blocking_findings=findings,scope_permissions=permissions,exact_byte_audit=audit,
            actual_cpu_evidence=cpu,source_clean_before=True,source_clean_after=True,
            source_git_head_before='02b1ea2688ff3fb032ee6699003a7e09d651f166',
            source_git_head_after='02b1ea2688ff3fb032ee6699003a7e09d651f166',
            prior_V3_blocker_resolution=dict(
                FORMAL_V3_02B1_P01='Repaired at source-interface/CPU-fixture level: fixed package official tokenizer path, four file pins before/after loader, real live registration before mu; official compatibility still unverified.',
                FORMAL_V3_02B1_P02='Early scheduler/supervisor/NVML checks improved but not fully repaired: CLI alias and startup-error bypasses remain blocking.',
                FORMAL_V3_02B1_P03='Repaired through exact adapter aliases and current compact receipt intake; owner original status/tests/limits preserved. This rejected receipt is deliberately not compact acceptance.'),
            nonblocking_metadata_observations=[audit['stale_runtime_entry_effect']],
            unchanged_contracts=contracts,historical_source_cpu_receipt_not_relabelled=True,
            operations=dict(actual_remote_calls=0,actual_scheduler_calls=0,gpu_nvml_calls=0,
                            official_asset_loads=0,pt_deserializations=0,optimizer_writes=0,
                            completed_training_outers=0,source_edits=0,wrapper_edits=0,
                            other_main_output_edits=0,foreign_signals=0,transfers=0,
                            distinct_changed_test_methods_executed=9,
                            changed_test_method_executions=18,
                            independent_tiny_mu_probe_executions=2,
                            cli_guard_bypass_probe_executions=2,startup_failure_probe_executions=1),
            review_execution_note='One initial -I dependency-path setup failure retained; after controls, existing local CPU dependencies were exposed. Changed tests and tiny/CLI checks ran twice because actual startup-failure probe was added to resolve the newly identified exception boundary; no full suite/Gloo rerun.',
            review_artifacts={name:dict(path=str(HERE/name),sha256=sha(HERE/name)) for name in (
                'CPU_REVIEW_PROBES.py','CPU_REVIEW_PROBES.log','CPU_REVIEW_PROBES_RESULT.json',
                'CPU_PROBE_INITIAL_ENVIRONMENT_FAILURE.json','EXACT_BYTE_AUDIT.py',
                'EXACT_BYTE_AUDIT_RESULT.json','BYTE_AUDIT_INITIAL_ASSERTION_RECORD.json')})
with (HERE/'REVIEW.json').open('x') as stream:json.dump(review,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
receipt=dict(schema_version=1,status=review['status'],disposition=review['disposition'],created_utc=created,
             review_scope=scope,fork_turns='none',reviewer_context=context,
             inherited_parent_conversation=False,code_commit=review['code_commit'],source_commit=review['code_commit'],
             source_clean=True,source_interface_count=15,source_closure_count=19,wrapper_count=33,
             full_wrapper_inventory_count=audit['full_inventory_count'],changed_wrapper_count=7,
             all_pinned_input_sha_verified=True,byte_closure_verified=True,
             blocking_findings=[{k:f[k] for k in ('id','priority','category','title')} for f in findings],
             scope_permissions=permissions,source_wrapper_combination_accepted=False,
             new_source_formal_combination_review_accepted=False,
             source_independent_cpu_receipt_sha256=audit['input_bindings']['source_independent_cpu']['actual_sha256'],
             source_independent_cpu_receipt_not_relabelled=True,
             input_bindings=audit['input_bindings'],actual_training_outers=0,actual_optimizer_writes=0,
             distinct_changed_cpu_test_methods=9,tiny_cpu_mu_registration_passed=True,
             actual_gpu_scheduler_remote_asset_calls=0,
             review_path=str(HERE/'REVIEW.json'),review_sha256=sha(HERE/'REVIEW.json'))
with (HERE/'RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
hashes={name:sha(HERE/name) for name in ('RECEIPT.json','REVIEW.json')}
with (HERE/'BYTE_HASHES.json').open('x') as stream:json.dump(hashes,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(status=receipt['status'],blocking_findings=len(findings),hashes=hashes)))
