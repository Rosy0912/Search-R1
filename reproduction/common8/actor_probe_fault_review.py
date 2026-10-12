"""Bind an independent recovery review to actual original failure evidence.

Collection and verification are read-only. This module never creates an accepted
review; the controlling agent records it only after reviewing the actual failure
and the independently reviewed candidate source/deployment. A scalar fixture is
not authorization or native recovery acceptance.
"""
from pathlib import Path

from reproduction.common8.actor_probe_rank_tail import _sealed_bytes, read_unreturned_tail
from reproduction.common8.actor_probe_retained_prefix import load_prefix, console_evidence, rank_prefix
from reproduction.common8.actor_probe_segment_acceptance import verify_trainer_exit, read_manifest
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from search_g1.actor_probe_recovery import identity, binding_digest

ROLE = 'actor_probe_parent_fault_evidence_review'
PATH = 'logs/actor-probe-parent-fault-review.json'


def collect(base, run, retained_through):
    base = Path(base); assert read_manifest(base)[0] == run
    prefix = load_prefix(base,run,retained_through)
    assert prefix['segment']['observed_completed_through'] < 200, \
        'A completed trainer needs final evaluation/acceptance repair, not replay from an earlier checkpoint'
    exited = verify_trainer_exit(base,run)
    from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup
    cleanup = verify_cleanup(base,run)
    tail = read_unreturned_tail(base,run,retained_through=retained_through)
    console = console_evidence(base,run,prefix)
    ranks = {str(rank):{kind:rank_prefix(base,run,prefix,rank,kind)[1]
        for kind in ('optimizer','scheduler')} for rank in range(4)}
    raw, file = _sealed_bytes(base,'logs/pipeline-status.json',limit=32*1024**2)
    pipeline = _strict_json(raw); assert pipeline['run_id'] == run['run_id']
    return dict(role='actor_probe_actual_parent_failure_evidence',identity=identity(run),
        retained_through=retained_through,trainer_exit=exited,pipeline_status=pipeline,pipeline_status_file=file,
        retained_prefix=prefix,discarded_rank_tail=tail,console_evidence=console,rank_prefix_metadata=ranks,
        parent_owned_cleanup_files=cleanup['files'],parent_owned_cleanup_reverified=True,
        no_rollback_claimed=True,discarded_work_counted_as_new_complete_outers=0)


def verify_documents(review, child, parent, evidence):
    assert review['role'] == ROLE and review['accepted'] is True
    assert review['identity'] == identity(child) and review['parent_identity'] == identity(parent)
    assert review['recovery_binding_sha256'] == binding_digest(child['actor_probe_recovery'])
    assert review['retained_through'] == child['actor_probe_recovery']['completed_outer'] == evidence['retained_through']
    assert review['parent_evidence_sha256'] == document_sha256(evidence)
    assert review['reviewed_source_commit'] == child['source_commit']
    assert review['reviewed_source_map_sha256'] == document_sha256(child['source_sha256'])
    assert review['independent_review_initialized_without_parent_context'] is True
    assert review['actual_failure_reviewed'] is True and review['actual_candidate_deployment_reviewed'] is True
    assert review['original_failure_source_and_logs_preserved'] is True
    assert review['no_retained_failure_ignored'] is True
    assert review['resolution_kind'] in ('reviewed_source_repair','full_state_restore_after_external_exit')
    assert isinstance(review['reviewer_task'],str) and review['reviewer_task'].startswith('/root/')
    assert isinstance(review['resolution_summary'],str) and review['resolution_summary'].strip()
    return review


def verify(base, child):
    base = Path(base); binding = child['actor_probe_recovery']
    parent_base = Path(binding['parent_base']); parent, file = read_manifest(parent_base)
    assert file['sha256'] == binding['parent_manifest_sha256']
    evidence = collect(parent_base,parent,binding['completed_outer'])
    raw, review_file = _sealed_bytes(base,PATH,limit=32*1024**2)
    review = verify_documents(_strict_json(raw),child,parent,evidence)
    return dict(role='actor_probe_verified_parent_fault_review',accepted=True,
        review_file=review_file,parent_evidence_sha256=review['parent_evidence_sha256'],
        parent_identity=identity(parent),resolution_kind=review['resolution_kind'],
        discarded_rank_tail=evidence['discarded_rank_tail'],
        actual_failure_and_candidate_review_verified=True)
