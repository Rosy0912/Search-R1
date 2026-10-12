"""Prospective guards for a saved full-state origin, never fresh0 admission.

This freezes actual child service/resource inputs before its first trainer claim.
It requires the separate production launcher and all-rank full-state restoration;
it neither starts a trainer nor claims a healthy checkpoint or probability test.
Historical verification reads immutable receipts without asserting liveness.
"""
import argparse
import json
import math
import time
from pathlib import Path

from reproduction.common8.formal_contract import (
    FORMAL_COUNTING, FORMAL_SAFEGUARDS, SAFETY_MODE, assert_formal_run,
)
from reproduction.common8.formal_inline_safety import (
    PLAN, native_birth, owned_path, read, write_once,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import actor_probes, head_round
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import binding_digest, identity, validate_recovery

ROLE = 'formal_actor_probe_restored_origin_candidates'
METHOD = 'preparation/actor-probe-recovery-method.json'
REQUIRED = (
    'logs/actor-probe-recovery-candidate.json',
    'logs/actor-probe-recovery-preparation.json',
    'logs/actor-probe-recovery-heads-staged.json',
    'logs/actor-probe-head-service-manifest.json',
    'logs/initial-probes-accepted.json',
    'logs/native-allocation-admission.json',
    'logs/local-resource-admission.json',
    'logs/retrieval-admission.json',
    'logs/retrieval-http-admission.json',
    'logs/evaluator-release-admission.json', METHOD,
)
MUTABLE_PLACEMENT = {'pipeline_identity', 'probe_visible_device',
                     'retrieval_placement', 'probe_logical_ordinal'}


def _manifest_without_placement(run):
    return {key:value for key,value in run.items() if key not in MUTABLE_PLACEMENT}


def _time(value):
    assert type(value) in (int,float) and math.isfinite(value) and value > 0
    return value


def _unclaimed(base, run):
    """Inherited counters do not grant permission to overwrite child work."""
    assert namespace_origin(run)['completed_outer'] > 0
    for name in ('training-launch-claim.json', 'actual-attempts.jsonl',
                 'actual-updates.jsonl', 'actor-probe-recovery-restored.json',
                 'actor-probe-recovery-launch-admission.json'):
        path=base/'logs'/name
        assert not path.exists() and not path.is_symlink(), 'Child trainer already attempted; never refreeze'
    assert not list((base/'logs').glob('*-rank-*.jsonl'))
    assert not list((base/'logs').glob('formal-inline-optimizer-*.json'))
    log=Path(run['train_log']);assert not log.exists() and not log.is_symlink()
    checkpoint=Path(run['checkpoint_dir']);assert checkpoint == checkpoint.resolve()
    assert not checkpoint.exists() or not any(checkpoint.iterdir())


def method(run, created_unix):
    origin=namespace_origin(run);assert origin['completed_outer'] > 0
    return dict(schema_version=1, role='formal_actor_probe_restore_method',
        identity=identity(run), source_map_sha256=document_sha256(run['source_sha256']),
        recovery_binding_sha256=binding_digest(run['actor_probe_recovery']), origin=origin,
        policy_safety_mode=SAFETY_MODE, target_updates=200, warmup_steps=57,
        fresh_states=False, full_model_optimizer_scheduler_rng_sampler_heads_required=True,
        next_logical_update=origin['completed_outer']+1,
        new_namespace_optimizer_writes=0, inherited_optimizer_writes=origin['actual_optimizer_writes'],
        optimizer_counting=FORMAL_COUNTING, safeguards=dict(FORMAL_SAFEGUARDS),
        negative_tail_semantics='prospective operational candidate; not calibrated health',
        created_unix=_time(created_unix))


def _verify_retrieval_and_release(base,run,allocation,receipts,version):
    from reproduction.common8.actor_probe_head_service import ROLE as HEAD_ROLE
    retrieval=read(owned_path(base,'logs/retrieval-admission.json'))
    http=read(owned_path(base,'logs/retrieval-http-admission.json'))
    released=read(owned_path(base,'logs/evaluator-release-admission.json'))
    for document in (retrieval,http,released):
        assert document['run_id']==run['run_id'] and document['execution_id']==run['execution_id']
        assert document['execution_backend']=='ssh_local' and document['slurm_job_id'] is None
    for document in (retrieval,http):
        assert document['accepted'] is True and document['stage_sha256']==run['input_sha256']['stage.parquet']
    ids=retrieval['question_ids']
    assert len(ids)==len(set(ids))==64 and http['question_ids']==ids
    assert http['retrieval_admission_sha256']==receipts['logs/retrieval-admission.json']
    hashes=retrieval['gpu_document_sha256']
    assert len(hashes)==64 and all(len(row)==3 for row in hashes)
    assert http['document_sha256']==hashes
    assert retrieval['host']==allocation['node']
    assert retrieval['physical_gpu_indices']==run['retriever_physical_indices']==[5,6]
    assert retrieval['actual_cuda_runtime_uuids']==[allocation['assigned_gpu_uuids'][str(i)] for i in (5,6)]
    for key,flag in (('released','suspended'),('resumed','resumed')):
        document=released[key]
        assert document[flag] is True and document['version']==version
        assert document['role']==HEAD_ROLE and document['frozen_encoder_loaded'] is False
    assert type(released['released']['allocated_bytes_after_release']) is int
    assert released['released']['allocated_bytes_after_release']==0


def _verify_services(base, run, receipts):
    """Bind immutable admissions; actual endpoint liveness is checked at freeze."""
    from reproduction.common8.actor_probe_head_admission import ARCHIVE
    from reproduction.common8.actor_probe_head_service import verify_manifest, head_paths
    origin=namespace_origin(run);binding=run['actor_probe_recovery']
    candidate=read(owned_path(base,'logs/actor-probe-recovery-candidate.json'))
    assert _manifest_without_placement(candidate)==_manifest_without_placement(run)
    prep=read(owned_path(base,'logs/actor-probe-recovery-preparation.json'))
    assert prep['role']=='actor_probe_full_recovery_preparation' and prep['prepared'] is True
    assert prep['production_admission'] is False and prep['identity']==identity(run)
    assert prep['candidate_manifest']=='logs/actor-probe-recovery-candidate.json'
    assert prep['candidate_manifest_sha256']==receipts[prep['candidate_manifest']]
    assert prep['origin']==origin and prep['recovery_binding_sha256']==binding_digest(binding)
    assert prep['parent_trainer_session_exit_verified'] is True and prep['canonical_finite_tensor_count']==435
    assert prep['parent_manifest_sha256']==binding['parent_manifest_sha256']
    assert prep['parent_full_state_receipt_sha256']==binding['full_state_receipt_sha256']
    parent_base=Path(binding['parent_base'])
    assert parent_base.is_absolute() and parent_base==parent_base.resolve() and parent_base!=base
    parent_path=owned_path(parent_base,'logs/run-manifest.json');parent=read(parent_path)
    assert digest(parent_path)==binding['parent_manifest_sha256']
    directory=Path(parent['checkpoint_dir'])/'actor'/f"global_step_{origin['completed_outer']}"
    assert directory==directory.resolve() and directory.is_relative_to(parent_base)
    saved_path=owned_path(parent_base,(directory/'full-state-complete.json').relative_to(parent_base).as_posix())
    saved=read(saved_path);assert digest(saved_path)==binding['full_state_receipt_sha256']
    validate_recovery(binding,run,parent,saved)
    staged=read(owned_path(base,'logs/actor-probe-recovery-heads-staged.json'))
    assert staged['role']=='actor_probe_restored_heads_staged' and staged['staged'] is True
    assert staged['production_admission'] is False and staged['identity']==identity(run)
    assert staged['recovery_binding_sha256']==binding_digest(binding)
    assert staged['preparation_sha256']==receipts['logs/actor-probe-recovery-preparation.json']
    assert staged['parent_full_state_receipt_sha256']==binding['full_state_receipt_sha256']
    assert staged['parent_probe_manifest_sha256']==saved['probe_manifest_sha256']
    archive=owned_path(base,ARCHIVE);live=read(archive)
    verify_manifest(run,live,logical_update=origin['completed_outer']+1)
    assert live['round']==staged['round']==head_round(origin['completed_outer']+1)
    assert live['version']==staged['version']==run['initial_probe_version']==saved['probe_version']
    assert live['actor_step']==staged['actor_step']==saved['probe_actor_step']
    assert live['head_sha256']==staged['head_sha256']==saved['probe_head_hashes']
    assert live['calibration_data_sha256']==staged['calibration_data_sha256']
    assert digest(archive)==staged['archive_manifest_sha256']==staged['live_manifest_sha256']
    preserved=Path(staged['archive_manifest'])
    assert preserved==preserved.resolve() and preserved.is_relative_to(base/'artifacts')
    assert digest(owned_path(base,preserved.relative_to(base).as_posix()))==digest(archive)
    paths=head_paths(base,live)
    for path in [preserved,*paths.values()]:
        relative=path.relative_to(base).as_posix()
        assert receipts[relative]==digest(owned_path(base,relative))
    accepted=read(owned_path(base,'logs/initial-probes-accepted.json'))
    assert accepted['role']=='actual_actor_binary_head_endpoint_admission' and accepted['accepted'] is True
    assert accepted['identity']==identity(run) and accepted['origin']==origin
    assert accepted['version']==live['version']
    assert accepted['archived_manifest']==ARCHIVE
    assert accepted['live_manifest_sha256']==accepted['archived_manifest_sha256']==digest(archive)
    assert accepted['actor_features_exercised'] is False and accepted['task_accuracy_claimed'] is False
    from reproduction.common8.actor_probe_head_admission import verify_endpoint_documents
    verify_endpoint_documents(run,live,accepted['actual_http_health'],accepted['actual_http_readouts'],digest(archive))
    allocation=read(owned_path(base,'logs/native-allocation-admission.json'))
    assert allocation['accepted'] is True and allocation['run_id']==run['run_id']
    assert allocation['execution_id']==run['execution_id'] and allocation['execution_backend']=='ssh_local'
    birth=native_birth(allocation)
    assert allocation['pipeline_identity']==run['pipeline_identity']
    assert run['probe_visible_device']=='6' and type(run['probe_logical_ordinal']) is int and run['probe_logical_ordinal']==0
    for key in ('training_physical_indices','auxiliary_physical_indices',
                'retriever_physical_indices','probe_physical_index'):
        assert allocation[key]==run[key]
    resource=read(owned_path(base,'logs/local-resource-admission.json'))
    assert resource['accepted'] is True and resource['run_id']==run['run_id']
    assert resource['execution_id']==run['execution_id'] and resource['execution_backend']=='ssh_local'
    assert resource['slurm_job_id'] is None and resource['host']==allocation['node'] and resource['training_world_size']==4
    assert resource['historical_gpu_smoke_executed'] is False
    assert resource['occupancy_policy']=='monitor_only' and resource['sampled_peak_fraction_stop_limit'] is None
    assert resource['fitted_probe_manifest_sha256']==digest(archive)
    for field,relative in [('allocation','logs/native-allocation-admission.json'),
                           ('retrieval_admission','logs/retrieval-admission.json'),
                           ('retrieval_http_admission','logs/retrieval-http-admission.json'),
                           ('evaluator_release_admission','logs/evaluator-release-admission.json')]:
        assert resource[field+'_sha256']==receipts[relative]
    _verify_retrieval_and_release(base,run,allocation,receipts,live['version'])
    return origin,birth,live


def build(base, run, document, created_unix):
    """Freeze only previously written admissions and the actual resolved recipe."""
    from reproduction.common8.actor_probe_head_service import head_paths
    from reproduction.common8.formal_contract import verify_formal_config
    from reproduction.common8.precision_contract import sampling_journal_only
    from reproduction.common8.topology import monitor_scope
    assert_formal_run(run);assert actor_probes(run) and sampling_journal_only(run)
    assert run['execution_backend']=='ssh_local' and run['training_world_size']==4
    receipts={name:digest(owned_path(base,name)) for name in REQUIRED}
    live=read(owned_path(base,'logs/actor-probe-head-service-manifest.json'))
    staged=read(owned_path(base,'logs/actor-probe-recovery-heads-staged.json'))
    paths=[Path(staged['archive_manifest']),*head_paths(base,live).values()]
    for path in paths:
        assert path.is_relative_to(base)
        relative=path.relative_to(base).as_posix();receipts[relative]=digest(owned_path(base,relative))
    origin,birth,live=_verify_services(base,run,receipts)
    proof=verify_formal_config(document,run,base)
    return dict(schema_version=1,role=ROLE,identity=identity(run),
        source_sha256=run['source_sha256'],initial_model_sha256=run['initial_model_sha256'],
        input_sha256=run['input_sha256'],budget=run['budget'],origin=origin,
        recovery_binding_sha256=binding_digest(run['actor_probe_recovery']),
        accepted_for_guarded_training=True,production_launch_admitted=False,
        short_train_accepted=False,healthy_checkpoint_accepted=False,
        new_namespace_optimizer_writes_at_freeze=0,inherited_optimizer_writes_at_freeze=origin['actual_optimizer_writes'],
        full_state_restore_required_before_rollout=True,initial_probe_version=live['version'],
        initial_probe_archive='logs/actor-probe-head-service-manifest.json',
        native_birth=birth,slurm_job_id=None,execution_backend='ssh_local',execution_id=run['execution_id'],
        training_world_size=4,monitoring_scope=monitor_scope(run),
        receipts_sha256=receipts,method_sha256=receipts[METHOD],resolved_config=document,
        resolved_config_sha256=proof['resolved_config_sha256'],
        actual_behavior_probability_accepted=False,actual_full_rollout_gate_required=True,
        trajectory_count=2560,behavior_probability_policy=run['behavior_probability_policy'],
        hf_recomputation_required=False,tolerance=None,sampling=dict(temperature=1.,top_p=1.),
        optimizer_counting=FORMAL_COUNTING,created_unix=_time(created_unix),**FORMAL_SAFEGUARDS)


def verify(base, artifact, run):
    """Pure historical receipt binding; live process/restore checks are separate."""
    from reproduction.common8.formal_contract import verify_formal_config
    base=Path(base).resolve();assert_formal_run(run);assert actor_probes(run)
    assert type(artifact['schema_version']) is int and artifact['schema_version']==1 and artifact['role']==ROLE
    receipts=artifact['receipts_sha256'];assert set(REQUIRED)<=set(receipts)
    for relative,sha in receipts.items():assert digest(owned_path(base,relative))==sha
    expected_method=method(run,read(owned_path(base,METHOD))['created_unix'])
    assert read(base/METHOD)==expected_method
    assert _time(run['started_unix'])<=expected_method['created_unix']<=_time(artifact['created_unix'])
    # Reconstruct from the actual immutable inputs, not self-asserted flags.
    expected=build(base,run,artifact['resolved_config'],artifact['created_unix'])
    assert canonical_bytes(expected)==canonical_bytes(artifact)
    assert artifact['method_sha256']==digest(base/METHOD)
    verify_formal_config(artifact['resolved_config'],run,base)
    return artifact['negative_tail_limit'],document_sha256(artifact)


def prepare(base):
    from omegaconf import OmegaConf
    from reproduction.common8.configuration import resolved
    from reproduction.common8.actor_probe_head_admission import verify_training_gate
    from reproduction.common8.topology import validate_base
    base=Path(base).resolve();run=read(owned_path(base,'logs/run-manifest.json'))
    validate_base(base,run)
    assert Path(__file__).resolve().parents[2]==base/'repo', 'Freeze must execute the pinned child source'
    entry=Path(__file__).resolve().relative_to(base/'repo').as_posix()
    assert digest(Path(__file__))==run['source_sha256'][entry]
    _unclaimed(base,run)
    for relative in (METHOD,PLAN):
        path=base/relative;assert not path.exists() and not path.is_symlink(), 'Inspect incomplete freeze; never overwrite'
    assert verify_training_gate(base)==run  # Includes actual endpoint PID birth, inputs and source.
    actual,config,_=resolved(base);assert actual==run
    write_once(base/METHOD,method(run,time.time()))
    artifact=build(base,run,OmegaConf.to_container(config,resolve=True),time.time())
    verify(base,artifact,run);_unclaimed(base,run)
    write_once(base/PLAN,artifact)
    return artifact


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();artifact=prepare(args.base)
    print(json.dumps(dict(role=ROLE,origin=artifact['origin'],production_launch_admitted=False,
                         full_state_restore_required_before_rollout=True)))


if __name__=='__main__':main()
