"""Materialize a human-authorized code-only direct200 packet; never submit/test.

The a18 source was already frozen. This writer records byte identity and the
latest human instruction without manufacturing calibration/review acceptance.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/'package-code-only'
REMOTE_ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
REMOTE_PARENT=REMOTE_ROOT/'instruct300'
REMOTE_PACKAGE=REMOTE_PARENT/'candidates/fresh200-direct-train-a18f420'
AUTHORITY_NAME='DIRECT_TRAIN200_HUMAN_AUTHORITY-a18f420.json'
RELEASE_NAME='DIRECT_TRAIN200_RELEASE-a18f420.json'
DISPATCH_NAME='DIRECT_TRAIN200_DISPATCH-a18f420.sbatch'
SOURCE='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
MODE='HUMAN_AUTHORIZED_DIRECT_TRAIN200'
THRESHOLDS=dict(target_old_policy_kl=.2,negative_tail_limit=.01,initial_actor_repeat_max_abs_delta=0.)
INSTRUCTION='别验证了，快点跑起来吧！很费时间大哥，训练起来才是王道，报错了就解决问题，而不是重复验证。'
ALLOWED_SUFFIXES={'.py','.sh','.sbatch','.yaml','.yml','.md','.json','.txt','.toml','.cfg','.ini','.patch'}
ALLOWED_EXTENSIONLESS={'.gitignore','LICENSE','NOTICE','Makefile','version'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def allowed(path):return path.suffix.lower() in ALLOWED_SUFFIXES or path.name in ALLOWED_EXTENSIONLESS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-proof-sha')
    args=parser.parse_args()
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Only read the frozen source package metadata. No git/model/GPU/test calls.
    frozen=HERE.parent/'tree-fresh200-freeze-a18f420/package-code-only/manifest.json'
    base=json.loads(frozen.read_text())
    overrides=list(base['configuration_overrides'])
    overrides=[value for value in overrides if not value.startswith('trainer.default_local_dir=')]
    overrides.append('trainer.default_local_dir=${oc.env:SEARCH_R1_BASELINE_CHECKPOINTS}')
    authority=dict(status=MODE,code_commit=SOURCE,human_instruction=INSTRUCTION,
        original_instruction_time=None,instruction_observed_at_utc=observed,
        instruction_provenance='Verbatim latest human instruction relayed by root in this active conversation; original timestamp not provided',
        same_instance_continuation=True,completed_outer_goal=200,final_evaluation_rows=51713,
        first_five_credited_outers_are_part_of_200=True,extra_calibration_required=False,
        external_first5_approval_required=False,configured_prospective_thresholds=THRESHOLDS,
        production_thresholds_accepted=False,independent_acceptance=False,training_accepted=False,
        original_sampler_actor_gate_accepted=False,actual_zero_ready=False,actual_calibration_accepted=False,
        bootstrap_thresholds_accepted=False,scope='Human authorizes direct training and same living-process continuation; numerical/independent acceptance remains unobserved')
    authority_path=HERE/AUTHORITY_NAME;write_json(authority_path,authority)
    dispatch=dict(cluster_name='wuhan-h100',nodes=['h100-9'],gpus=8,cpus_per_task=32,
        expected_allocated_cpus=192,memory_gib=256,exclusive=True,no_requeue=True,train_wall_hours=168,
        partition_max_time_observation='UNLIMITED observed by root before this packet',
        once_intent=str(REMOTE_PARENT/'SUBMISSION_ONCE-fresh200-direct-a18f420.json'))
    files={}
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file() or path==PACKAGE/'manifest.json':continue
        if path.is_symlink() or not allowed(path):raise ValueError('Code-only package contains unsupported member: '+str(path))
        files[str(path.relative_to(PACKAGE))]=sha(path)
    source_files={k.removeprefix('source/repo/'):v for k,v in files.items() if k.startswith('source/repo/')}
    manifest=dict(status=MODE,scope=MODE,execution_mode=MODE,stage='fresh_mu_train200',
        training_protocol='tree_fresh200_five_leaf_v1',code_commit=SOURCE,
        official_model_path=base['official_model_path'],model_identity=base['model_identity'],
        common_data_identity=base['common_data_identity'],configuration_overrides=overrides,
        configured_prospective_thresholds=THRESHOLDS,bootstrap_thresholds=THRESHOLDS,
        bootstrap_thresholds_accepted=False,production_thresholds_accepted=False,
        independent_acceptance=False,training_accepted=False,actual_training_accepted=False,
        original_sampler_actor_gate_accepted=False,actual_zero_ready=False,actual_calibration_accepted=False,
        actual_3b_training_accepted=False,long_training_ready=False,actual_gpu_executed=False,
        actual_remote_executed=False,actual_slurm_submitted=False,actual_execution_unknown=True,
        human_training_authority=dict(path=str(REMOTE_PARENT/AUTHORITY_NAME),sha256=sha(authority_path)),
        full_run_continuation_requires_same_instance_first5_threshold_assessment=False,
        external_first5_approval_required=False,first_five_credited_outers_are_part_of_200=True,
        first_five_continuation_receipt='output/acceptance-step5/human-authorized-continuation.json',
        checkpoints_at_credited_outers=[5,50,100,150,200],development_rows=512,development_every_credited_outers=5,
        final_evaluation_rows=51713,dispatch=dispatch,external_files=base['external_files'],
        files=files,wrapper_files={k:v for k,v in files.items() if not k.startswith('source/')},
        source_selection=dict(protocol='frozen_a18_code_metadata_only',selected_files=source_files,selected_count=len(source_files),
            not_a_full_repository_asset_copy=True,contains_model_data_index_checkpoint_assets=False),
        historical_synthetic_receipts=base['historical_synthetic_receipts'],
        historical_receipts_are_reference_only=True,new_cpu_tests_executed=False,new_reviews_executed=False,
        protocol=dict(local_runtime_only=True,local_runtime_host='h100-9',http='127.0.0.1:18082',ray='127.0.0.1:16382',
            retriever='same h100-9 E5 and complete 8-GPU sharded FP16 FAISS',fresh_official_3b_no_inheritance=True,
            dp_ranks=8,questions_per_outer=512,leaves_per_question=5,true_tree_leaf_quotas=[3,2],
            opportunities_per_rank=20,all_opportunities_including_skips_must_be_handled=True,
            one_or_more_actual_adam_writes_required_for_outer_credit=True,all_skipped_or_abandoned_credit=0,
            credited_outer_advances_scheduler_once=True,actual_optimizer_writes_recorded_separately=True,
            no_automatic_rollback=True),materialized_at_utc=observed)
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest)
    manifest_sha=sha(manifest_path)
    release=dict(status=MODE,execution_mode=MODE,code_commit=SOURCE,manifest_sha256=manifest_sha,
        human_training_authority_sha256=sha(authority_path),cluster_name='wuhan-h100',node_list='h100-9',
        dispatch_intent_path=dispatch['once_intent'],dispatch_renderer_sha256=files['render_dispatch.py'],
        dispatch_submitter_sha256=files['submit_once.py'],train_wall_hours=168,
        production_thresholds_accepted=False,independent_acceptance=False,training_accepted=False,
        original_sampler_actor_gate_accepted=False,release_kind='human_instruction_bound_execution_identity',
        scope='Byte-bound concrete execution permission; no independent PASS/calibration/threshold acceptance')
    release_path=HERE/RELEASE_NAME;write_json(release_path,release)
    sys.path.insert(0,str(PACKAGE))
    from render_dispatch import render
    rendered=render(REMOTE_PACKAGE,manifest_sha,REMOTE_PARENT/RELEASE_NAME,sha(release_path),168)
    dispatch_path=HERE/DISPATCH_NAME
    with dispatch_path.open('x') as stream:stream.write(rendered)
    argv=[str(REMOTE_ROOT/'env/bin/python'),str(REMOTE_PACKAGE/'submit_once.py'),
        '--package',str(REMOTE_PACKAGE),'--manifest-sha',manifest_sha,
        '--release-path',str(REMOTE_PARENT/RELEASE_NAME),'--release-sha',sha(release_path),
        '--script',str(REMOTE_PARENT/DISPATCH_NAME),'--script-sha',sha(dispatch_path),
        '--predecessor-proof',str(REMOTE_ROOT/'runs/tree-fresh200-3319/direct-training-predecessor-proof.json')]
    if args.predecessor_proof_sha:argv.extend(['--predecessor-proof-sha',args.predecessor_proof_sha])
    # Importing only the renderer creates cache locally; caches are not assets
    # or selected package bytes and never enter this archive.
    archive=HERE/'fresh200-direct-train-a18f420-code-only.tar.gz'
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted([*files,'manifest.json']):
            bundle.add(PACKAGE/relative,arcname='candidates/fresh200-direct-train-a18f420/'+relative,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_DIRECT200_PACKET_MATERIALIZED_NOT_SUBMITTED',local_package=str(PACKAGE),
        remote_extract_destination=str(REMOTE_PARENT),remote_package=str(REMOTE_PACKAGE),archive=str(archive),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,manifest_sha256=manifest_sha,
        authority_path=str(authority_path),authority_sha256=sha(authority_path),release_path=str(release_path),release_sha256=sha(release_path),
        dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),
        once_submission_argv=argv if args.predecessor_proof_sha else None,
        submission_argv_pending_predecessor_sha=argv if not args.predecessor_proof_sha else None,
        required_remaining_argv_option=None if args.predecessor_proof_sha else '--predecessor-proof-sha',
        once_intent=dispatch['once_intent'],new_tests_executed=False,new_reviews_executed=False,
        actual_slurm_submitted=False,contains_model_data_index_checkpoint_assets=False)
    write_json(HERE/'DIRECT_PACKET_BOUND_ARGV.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
