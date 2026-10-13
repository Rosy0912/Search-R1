"""Read-only truthful training200 + authorized evaluation-handoff gate.

The original3360 training completed; its slow final evaluator was authorized to
stop. This gate never invents original exit0 or an original full-final terminal.
Composite evidence and a separate context-free final acceptance are required.
"""
import hashlib
import os
from pathlib import Path
import re
import stat
from production_contract import (ROOT, read_bound, require, direct_rank_records,
                                 checkpoint_manifest)

SCHEMA = 'tree_train200_eval_handoff_composite_v1'
COMPOSITE_STATUS = 'JOB3360_TRAIN200_AUTHORIZED_EVAL_HANDOFF_FULL_HF_ACCEPTED'
INDEPENDENT_STATUS = 'INDEPENDENT_JOB3360_TRAIN200_EVAL_HANDOFF_ACCEPTED'
COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa':12576}, musique=2417, bamboogle=125)


def hash_bound(value):
    """Hash immutable ordinary files, including tensors, without loading them."""
    require(type(value) is dict and set(value)=={'path','sha256'}, 'Evidence descriptor required')
    path=Path(value['path']); expected=value['sha256']
    require(path.is_absolute() and '..' not in path.parts
            and path.resolve().is_relative_to(ROOT.resolve()), 'Evidence outside own remote filesystem')
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}',expected), 'Exact evidence SHA required')
    require(stat.S_ISREG(path.lstat().st_mode), 'Evidence must be ordinary: '+str(path))
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno()); digest=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        after=os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            and digest.hexdigest()==expected, 'Preserved evidence changed: '+str(path))
    return expected


def same_path(actual, expected):
    require(Path(actual).is_absolute() and Path(actual).resolve()==Path(expected).resolve(),
            'Evidence path differs: '+str(expected))


def verify_predecessor(package):
    spec=package['predecessor_gate']; run=ROOT/'runs/tree-branpo-fresh200-3360'
    require(spec['job_id']=='3360' and spec['run_id']==run.name and spec['run_root']==str(run)
            and spec['mutations_allowed'] is False and spec['required_completed_outer']==200
            and spec['required_final_rows']==51713 and spec['schema_version']==SCHEMA,
            'Exact preserved3360 composite gate required')
    old_path=ROOT/'instruct300/candidates'/('branpo-fresh200-'+spec['source_commit'])/'manifest.json'
    old,package_sha,_=read_bound(old_path,spec['package_manifest_sha256'])
    require(old['code_commit']==spec['source_commit'] and old['common_data_identity']==package['common_data_identity'],
            'Predecessor source/split differs')
    receipt_path=run/'output/eval-handoff-hf-acceptance/identity-bound-handoff-hf.json'
    review_path=receipt_path.parent/'independent-composite-acceptance.json'
    require(spec['composite_receipt']==str(receipt_path) and spec['independent_acceptance_receipt']==str(review_path),
            'Fixed new composite/final acceptance paths required')
    composite,composite_sha,_=read_bound(receipt_path)
    require(composite['schema_version']==SCHEMA and composite['status']==COMPOSITE_STATUS
            and composite['training_job_id']=='3360' and composite['training_run_id']==run.name
            and composite['training_source_commit']==spec['source_commit']
            and composite['evaluation_source_commit']==spec['evaluation_source_commit']
            and composite['config_sha256']==spec['config_sha256']
            and composite['predecessor_package_manifest_sha256']==package_sha
            and composite['split_identity_sha256']==old['common_data_identity'], 'Composite identities differ')
    require(composite['actual_completed_outer']==200 and composite['actual_writes_per_rank']=={str(r):4000 for r in range(8)}
            and composite['original_training_normal_exit_claimed'] is False
            and composite['original_training_body_exit_code'] is None
            and composite['original_final_evaluation_completed'] is False
            and composite['original_evaluation_stop_authorized'] is True
            and composite['original_cancelled_state']=='CANCELLED'
            and composite['original_slurm_exit_code']=='0:15'
            and composite['training_or_optimizer_restored'] is False,
            'Truthful completed training + authorized cancellation required; never synthesize exit0')
    require(composite['replacement_evaluation_complete'] is True and composite['evaluated_count']==51713
            and composite['preserved_count']==30880 and composite['replacement_count']==20833
            and composite['final_per_dataset']==COUNTS and composite['replacement_worker_exit_code']==0
            and composite['replacement_own_cleanup_residual']=={}
            and composite['actual_cpu_model_cold_reload'] is True and composite['actual_tokenizer_cold_reload'] is True
            and composite['independent_final_acceptance'] is False, 'Actual final/HF/tokenizer/cleanup incomplete')
    before=composite['input_hashes_before']; after=composite['input_hashes_after']
    require(type(before) is dict and before and before==after, 'Acceptance inputs changed across cold reload')
    required={'wrapper','resolved_config','package_manifest','frozen_verifier','native_manifest',
              'original_job_terminal','original_scontrol_observation','handoff_journal',
              'eval_asset_manifest','eval_release','eval_review','replacement_terminal',
              'replacement_allocation','preserved_prefix','preserved_prefix_done',
              'final_metrics','final_metrics_done'}
    require(required<=set(before), 'Missing composite source/terminal/split/prefix/ledger bindings')
    eval_tools=ROOT/'instruct300'/('eval-shared-'+spec['evaluation_source_commit'])
    eval_run=ROOT/'runs'/'sst-tree-eval-shared-0e00998'
    scores=run/'output/eval-resume/scores'
    fixed=dict(resolved_config=run/'resolved.yaml',package_manifest=old_path,
               native_manifest=run/'checkpoints/training_state/global_step_200/manifest.json',
               frozen_verifier=old_path.parent/'source/repo/reproduction/baselines/tree-validation-20261002/evidence/runtime/check_real_checkpoint.py',
               handoff_journal=run/'output/eval-resume/dispatch-once.json',
               eval_asset_manifest=run/'output/eval-resume/asset-manifest.json',
               eval_release=eval_tools/'release.json',eval_review=eval_tools/'SHARED_CODE_REVIEW.json',
               replacement_terminal=eval_run/'run-result.json',replacement_allocation=eval_run/'allocation-identity.json',
               preserved_prefix=run/'output/eval-resume/bound-prefix-30880.json',
               preserved_prefix_done=run/'output/eval-resume/bound-prefix-30880.json.done.json',
               final_metrics=scores/'final-metrics.json',final_metrics_done=scores/'final-metrics.done.json')
    for name,path in fixed.items():same_path(before[name]['path'],path)
    require(Path(before['original_job_terminal']['path']).resolve().is_relative_to((run/'output/eval-handoff-acceptance-inputs').resolve())
            and Path(before['original_scontrol_observation']['path']).resolve().is_relative_to(run.resolve()),
            'Original honest cancellation snapshots must remain preserved under3360')
    require(before['package_manifest']['sha256']==package_sha and before['resolved_config']['sha256']==spec['config_sha256']
            and before['preserved_prefix']['sha256']==spec['preserved_prefix_sha256']
            and before['eval_release']['sha256']==spec['evaluation_release_sha256']
            and before['eval_asset_manifest']['sha256']==spec['evaluation_asset_manifest_sha256'],
            'Exact old/new config/source/split/prefix pins differ')
    for item in before.values():hash_bound(item)
    for relative,expected in old['files'].items():
        item=before['frozen_package/'+relative];same_path(item['path'],old_path.parent/relative)
        require(item['sha256']==expected,'Frozen predecessor source inventory differs')
    original,_,_=read_bound(before['original_job_terminal']['path'],before['original_job_terminal']['sha256'])
    require(original['job_id']=='3360' and original['state']=='CANCELLED' and original['exit_code']=='0:15'
            and original['source']=='scontrol-preserved','Actual original scheduler observation differs')
    require(before['original_job_terminal']['sha256']==spec['original_job_observation_sha256']
            and before['original_scontrol_observation']['sha256']==spec['original_scontrol_observation_sha256'],
            'Original honest snapshot SHA pins differ')
    same_path(original['preserved_source_path'],before['original_scontrol_observation']['path'])
    require(original['preserved_source_sha256']==spec['original_scontrol_observation_sha256']
            and original['original_run_result_absent'] is True and original['original_train_terminal_absent'] is True,
            'Original absence/historical source facts differ')
    require(not (run/'run-result.json').exists() and not (run/'output/train200-terminal.json').exists(),
            'Original absent terminal was synthesized or changed')
    actual_terminal,_,_=read_bound(receipt_path.parent/'actual-terminal-observation.json',
                                  composite['actual_terminal_observation_sha256'])
    require(actual_terminal['original_run_result_absent'] is True
            and actual_terminal['original_train_terminal_absent'] is True
            and actual_terminal['current_old_queue_absent'] is True,
            'Acceptance actual old terminal/queue absence evidence incomplete')
    observed_state=dict(line.split('=',1) for line in actual_terminal['replacement_systemd_raw'].splitlines() if '=' in line)
    require(observed_state.get('MainPID')=='0' and observed_state.get('ExecMainStatus')=='0'
            and observed_state.get('ActiveState')=='inactive', 'Replacement unit terminal cleanup absent')
    handoff,_,_=read_bound(fixed['handoff_journal'],before['handoff_journal']['sha256'])
    require(handoff['status']=='HANDOFF_REQUESTED' and handoff['old_job']==3360
            and handoff['training_updates']==200 and handoff['cancel_returncode']==0
            and handoff['prefix_count']==30880 and handoff['prefix_sha256']==spec['preserved_prefix_sha256']
            and handoff['scope']=='User-authorized evaluation-only handoff; no training restart',
            'Actual authorized evaluation handoff journal differs')
    terminal,terminal_sha,_=read_bound(fixed['replacement_terminal'],before['replacement_terminal']['sha256'])
    require(terminal['status']=='COMPLETE' and terminal['unit']=='sst-tree-eval-shared-0e00998.service'
            and terminal['worker_exit_code']==0 and terminal['own_cleanup_residual']=={}
            and terminal['evaluation_only'] is True and terminal['training_updates']==200
            and terminal['predecessor_job']==3360 and terminal_sha==composite['replacement_terminal_sha256'],
            'Actual replacement evaluator exit/cleanup/source differs')
    release,_,_=read_bound(fixed['eval_release'],before['eval_release']['sha256'])
    require(release['source_commit']==spec['evaluation_source_commit'],'Replacement release source differs')
    for relative,expected in release['files'].items():
        item=before['eval_source/'+relative];same_path(item['path'],eval_tools/relative)
        require(item['sha256']==expected,'Replacement source inventory differs')
    assets,_,_=read_bound(fixed['eval_asset_manifest'],before['eval_asset_manifest']['sha256'])
    require(assets['dataset_count']==51713 and assets['expected_source_counts']==COUNTS
            and assets['config_sha256']==spec['config_sha256']
            and assets['dataset_sha256']==spec['evaluation_dataset_sha256']==composite['eval_dataset_identity_sha256'],
            'Exact final split/assets identity differs')
    for entry in assets['dataset_files']:
        item=before['dataset/'+entry['path']];same_path(item['path'],entry['path'])
        require(item['sha256']==entry['sha256'],'Actual final split bytes differ')
    prefix_done,_,_=read_bound(fixed['preserved_prefix_done'],before['preserved_prefix_done']['sha256'])
    require(prefix_done['count']==30880 and prefix_done['sha256']==spec['preserved_prefix_sha256']
            and prefix_done['actual_trained_outer']==200,'Prefix durability/count differs')
    final,final_sha,_=read_bound(fixed['final_metrics'],before['final_metrics']['sha256'])
    final_done,_,_=read_bound(fixed['final_metrics_done'],before['final_metrics_done']['sha256'])
    require(final_done['sha256']==final_sha and final['evaluated_count']==51713
            and final['metrics']==composite['final_metrics']
            and all(final['metrics']['val/evaluated_count/'+name]==count for name,count in COUNTS.items()),
            'Full final metrics descriptor/count differs')
    expected_provenance={f'provenance/rank-{rank}-provenance.json' for rank in range(8)}
    require({name for name in before if name.startswith('provenance/')}==expected_provenance,
            'Eight-rank replacement provenance binding missing')
    for name in expected_provenance:same_path(before[name]['path'],scores/name.split('/',1)[1])
    batch_files={path.name for path in (scores/'batches').glob('*.json') if not path.name.endswith('.done.json')}
    done_files={path.name for path in (scores/'batches').glob('*.done.json')}
    require(batch_files and {name.replace('.done.json','.json') for name in done_files}==batch_files
            and {name.split('/',1)[1] for name in before if name.startswith('batch/')}==batch_files
            and {name.split('/',1)[1] for name in before if name.startswith('batch_done/')}==done_files,
            'Complete immutable replacement ledger bindings differ')
    for name in before:
        if name.startswith(('batch/','batch_done/')):same_path(before[name]['path'],scores/'batches'/name.split('/',1)[1])
    # Independently recompute existing training evidence, with no final-eval assumption.
    ranks=direct_rank_records(run,old,spec['config_sha256'],200)
    require(ranks['actual_writes_per_rank']==composite['actual_writes_per_rank'], 'Actual rank Adam counts differ')
    for path,expected in ranks['record_sha256'].items():
        item=before['training/'+path];same_path(item['path'],path)
        require(item['sha256']==expected,'Actual training ledger binding differs')
    native,native_sha=checkpoint_manifest(run,old,spec['config_sha256'],200)
    require(native_sha==composite['native_manifest_sha256']
            and native['hf_identity']['sha256']==composite['hf_identity_sha256'], 'Native/HF preservation identity differs')
    require(before['native_manifest']['sha256']==native_sha,'Native manifest descriptor differs')
    for record in native['worker_states']+[native['driver']]:
        item=before['native_file/'+record['path']];same_path(item['path'],record['path'])
        require(item['sha256']==record['sha256'],'Actual native shard preservation binding differs')
    hf=run/'checkpoints/actor/global_step_200'
    for relative,expected in native['hf_identity']['files'].items():
        path=Path(relative)
        require(not path.is_absolute() and '..' not in path.parts,'Unsafe predecessor HF member')
        hash_bound(dict(path=str(hf/path),sha256=expected))
    full_path=receipt_path.parent/'full-HF-cold-reload.json'
    tokenizer_path=receipt_path.parent/'tokenizer-cold-reload.json'
    full,_,_=read_bound(full_path,composite['full_hf_report_sha256'])
    tokenizer,_,_=read_bound(tokenizer_path,composite['tokenizer_report_sha256'])
    require(full['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED'
            and full['all_parameters_finite'] is True and full['cpu_forward_logits_finite'] is True
            and full['changed_parameters']>0 and full['total_parameters']>1_000_000_000
            and full['checkpoint_file_sha256']==native['hf_identity']['files']
            and full['max_abs_parameter_delta']>0
            and not any(full['hf_loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
            'Actual full cold-HF parameter/key/shape proof differs')
    same_path(full['checkpoint'],hf);same_path(full['base'],old['official_model_path'])
    require(tokenizer['status']=='ACTUAL_COLD_TOKENIZER_IDENTITY_VERIFIED'
            and all(tokenizer.get(k) is True for k in ('vocab_equal','special_pad_bos_eos_unk_equal',
                'all_special_tokens_ids_equal','chat_template_equal','canonical_backend_equal',
                'full_added_token_attributes_equal','full_special_roles_equal','normalized_semantic_config_equal'))
            and tokenizer['actual_encode_decode_probes']>=3 and tokenizer['actual_chat_template_comparisons']==4,
            'Actual full tokenizer semantics acceptance differs')
    wrappers=composite['acceptance_wrapper_files']
    require(type(wrappers) is dict and wrappers=={'verify_handoff_hf.py':before['wrapper']}
            and composite['wrapper_sha256']==before['wrapper']['sha256'],
            'Actual reviewed acceptance wrapper files required')
    for item in wrappers.values():hash_bound(item)
    review,review_sha,_=read_bound(review_path)
    require(review['schema_version']==SCHEMA and review['status']==INDEPENDENT_STATUS
            and review['fork_turns']=='none' and review['blocking_findings']==[]
            and review['composite_receipt_sha256']==composite_sha
            and review['training_source_commit']==spec['source_commit']
            and review['evaluation_source_commit']==spec['evaluation_source_commit']
            and review['reviewed_wrapper_files']==wrappers
            and all(review.get(k) is True for k in ('training200_accepted','final51713_accepted',
                    'full_hf_accepted','tokenizer_accepted','native_preservation_accepted',
                    'authorized_handoff_accepted','cleanup_accepted','cost_accepted')),
            'Separate context-free actual composite acceptance missing')
    # Actual scope cost is a separate independently accepted receipt. The
    # already running immutable CPU-HF wrapper is never hot-edited to add it.
    cost_descriptor=review['cost_summary']
    same_path(cost_descriptor['path'],receipt_path.parent/'cost-summary-v2.json')
    hash_bound(cost_descriptor)
    cost,cost_sha,_=read_bound(cost_descriptor['path'],cost_descriptor['sha256'])
    require(cost['schema_version']=='tree_train200_eval_handoff_cost_v1'
            and cost['status']=='JOB3360_COMPOSITE_ACTUAL_SCOPE_COST_COMPLETE'
            and cost['training_job_id']=='3360' and cost['training_source_commit']==spec['source_commit']
            and cost['evaluation_source_commit']==spec['evaluation_source_commit']
            and cost['composite_receipt_sha256']==composite_sha,
            'Independent scope cost identity/composite binding differs')
    flags=cost['scope_flags']
    require(all(flags.get(name) is True for name in ('original_train200','original_dev40',
                'original_partial_evaluation','successful_replacement_evaluation',
                'failed_own_evaluation_startups','actual_cpu_hf_acceptance')),
            'Actual requested experiment cost scopes incomplete')
    cost_inputs=cost['input_sha256']
    require(type(cost_inputs) is dict and cost_inputs,'Actual scope cost input SHA map missing')
    for path,expected in cost_inputs.items():hash_bound(dict(path=path,sha256=expected))
    def cost_covers(item):
        matches=[expected for path,expected in cost_inputs.items() if Path(path).resolve()==Path(item['path']).resolve()]
        require(matches==[item['sha256']],'Actual scope cost omits/misbinds input: '+item['path'])
    memory_descriptor=package['dispatch']['completed200_memory_peak_report']
    cost_covers(memory_descriptor)
    memory,_,_=read_bound(memory_descriptor['path'],memory_descriptor['sha256'])
    evidence=memory.get('evidence_sha256',memory.get('input_sha256'))
    require(type(evidence) is dict and evidence,'Full200 underlying cost evidence missing')
    for path,expected in evidence.items():cost_covers(dict(path=path,sha256=expected))
    for name,item in before.items():
        if name in ('replacement_terminal','replacement_allocation','final_metrics','final_metrics_done') or name.startswith(('batch/','batch_done/','original_partial_cost/')):
            cost_covers(item)
    cost_covers(dict(path=str(full_path),sha256=composite['full_hf_report_sha256']))
    cost_covers(dict(path=str(tokenizer_path),sha256=composite['tokenizer_report_sha256']))
    cost_covers(dict(path=str(receipt_path.parent/'actual-terminal-observation.json'),sha256=composite['actual_terminal_observation_sha256']))
    if (eval_run/'cost.json').is_file():
        matches=[path for path in cost_inputs if Path(path).resolve()==(eval_run/'cost.json').resolve()]
        require(len(matches)==1,'Existing successful evaluator cost.json omitted from actual scope cost')
    resource_path=eval_run/'resource-samples.jsonl'
    matches=[path for path in cost_inputs if Path(path).resolve()==resource_path.resolve()]
    require(len(matches)==1,'Successful shared resource cost evidence missing')
    failed_units=('sst-tree-eval-shared-1c15c49','sst-tree-eval-shared-b316a77',
                  'sst-tree-eval-shared-993109d','sst-tree-eval-shared-13a5388','sst-tree-eval-shared-7d28b1b')
    failed_inputs=cost['failed_own_startup_inputs']
    require(type(failed_inputs) is dict and set(failed_inputs)=={unit+'.service' for unit in failed_units},
            'All five actual failed own startup scopes must be bound')
    for unit in failed_units:
        # Admission-only failures have no run-result: actual preserved systemd
        # terminal/cost snapshots + once-dispatch journals are sufficient.
        items=failed_inputs[unit+'.service']
        require(type(items) is list and len(items)>=2,'Failed own startup observation/journal evidence missing: '+unit)
        for item in items:cost_covers(item)
    require(type(cost['cold_hf_service_inputs']) is list and cost['cold_hf_service_inputs'],
            'Actual CPU verification service cost/terminal/elapsed descriptors missing')
    for item in cost['cold_hf_service_inputs']:cost_covers(item)
    # Unavailable systemd CPUUsageNSec may remain null/unknown; it is never
    # converted to zero or combined as if it were measured process CPU time.
    return dict(job_id='3360',completed_outer=200,final_rows=51713,
                source_commit=spec['source_commit'],evaluation_source_commit=spec['evaluation_source_commit'],
                predecessor_package_sha256=package_sha,native_manifest_sha256=native_sha,
                composite_receipt_sha256=composite_sha,independent_acceptance_sha256=review_sha,
                cost_summary_sha256=cost_sha,actual_scope_cost_accepted=True,
                original_training_normal_exit_claimed=False,original_slurm_exit_code='0:15',
                predecessor_mutations=False)
