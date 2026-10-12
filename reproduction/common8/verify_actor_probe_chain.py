"""Actual retained1..200 acceptance, followed by a separate owned cleanup audit.

The training/final verifier runs while pipeline services are still owned/live.
It never declares cleanup complete. --after-cleanup is read-only with respect
to processes and audits their actual births/cache/source once the pipeline exits.
"""
import argparse
import json
import time
from pathlib import Path

from reproduction.common8.actor_probe_segment_acceptance import discover_chain, verify_segment, read_manifest
from reproduction.common8.actor_probe_segment_acceptance import verify_canonical_files
from reproduction.common8.actor_probe_chain import assemble_counter_chain
from reproduction.common8.actor_probe_calibration_acceptance import verify_calibration_chain, _document
from reproduction.common8.actor_probe_first_five_acceptance import verify_root_five
from reproduction.common8.actor_probe_fault_review import verify as verify_fault_review
from reproduction.common8.actor_probe_owned_cleanup import verify_cleanup,birth_inventory
from reproduction.common8.final_evaluation_acceptance import read_final_evaluation
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_recovery import identity

ENTRY = 'reproduction/common8/verify_actor_probe_chain.py'
ACCEPTANCE = 'logs/full-run-acceptance.json'


def _executed_source(base,run):
    assert Path(__file__).resolve() == base/'repo'/ENTRY
    assert digest(Path(__file__)) == run['source_sha256'][ENTRY]


def _predecessor(base, run):
    from reproduction.a800_common.predecessor_admission import admit
    actual = admit(base,run)
    saved, file = _document(base,base/'logs/predecessor-completion-admission.json')
    assert saved['accepted'] is True and saved['predecessor'] == actual['predecessor'] == run['predecessor_completion']
    for key in ('actual_complete_outers','actual_optimizer_writes','surviving_session_processes','scope'):
        assert saved[key] == actual[key]
    return dict(actual_predecessor_completion_reverified=True,predecessor=actual['predecessor'],
        admission_file=file)


def compose(entries,segments,calibration,first_five,faults,predecessors,final):
    """Compose verified results; all actual readers run before this function."""
    assert len(entries) == len(segments) == len(predecessors)
    counter = assemble_counter_chain([s['counter_segment'] for s in segments],final_outer=200)
    assert calibration['original_calibration_provenance_verified'] is True
    assert calibration['retained_version_intervals_verified'] is True
    assert first_five['accepted'] is True and first_five['identity'] == identity(entries[0]['run'])
    assert set(faults) == {e['run']['run_id'] for e in entries[1:]}
    for entry,segment,predecessor in zip(entries,segments,predecessors):
        assert segment['actual_interval_accepted'] is True
        assert segment['identity'] == identity(entry['run']) and segment['base'] == entry['base']
        assert segment['first'] == entry['origin']['completed_outer']+1 and segment['last'] == entry['last']
        assert predecessor['actual_predecessor_completion_reverified'] is True
        if entry['run']['run_id'] in faults:
            review = faults[entry['run']['run_id']]
            assert review['accepted'] is True and review['actual_failure_and_candidate_review_verified'] is True
    terminal = entries[-1]['run']
    assert final['run_id'] == terminal['run_id'] and final['step'] == 200 and final['question_count'] == 51713
    assert len(final['counts']) == 7 and sum(final['counts'].values()) == 51713 and final['counts']['nq'] == 3610
    checkpoints = {}; full_states = {}; updates = []
    for segment in segments:
        for step,item in segment['checkpoint_proofs'].items():
            assert step not in checkpoints and item['finite_tensor_count'] == 435
            checkpoints[step] = item['receipt']; full_states[step] = segment['full_states'][step]
        updates += segment['counter_segment']['retained_updates']
    assert set(checkpoints) == set(full_states) == {'5','50','100','150','200'}
    assert [row['step'] for row in updates] == list(range(1,201))
    processed = sum(20*len(segment['counter_segment']['retained_attempts']) for segment in segments)
    assert processed >= 4000 and 200 <= counter['successful_adam'] <= 4000
    assert counter['reward_phases']['g1']['count'] == counter['reward_phases']['binary_em']['count'] == 100
    return dict(role='actor_probe_actual_complete_retained_chain',schema_version=1,
        run_id=terminal['run_id'],identity=identity(terminal),experiment_root_run_id=entries[0]['run']['run_id'],
        structural_accepted=True,actual_outer_updates=200,retained_counter_chain=counter,
        per_rank_optimizer_opportunities=processed,
        per_rank_actual_optimizer_writes={str(rank):counter['successful_adam'] for rank in range(4)},
        retained_skipped_opportunities=processed-counter['successful_adam'],
        actual_updates=updates,checkpoint_proofs=checkpoints,full_state_proofs=full_states,
        training_segments=segments,original_calibration=calibration,actual_root_first_five=first_five,
        recovery_fault_reviews=faults,predecessor_admissions=predecessors,
        final_saved_tensor_count=435,final_counts=final['counts'],final_correct=final['correct'],
        final_weighted_em=final['weighted_em'],final_weighted_f1=final['weighted_f1'],
        final_source_em=final['source_em'],final_source_f1=final['source_f1'],final_evaluation=final,
        full_training_and_final_evaluation_verified=True,owned_cleanup_verified=False,goal_complete=False,
        checked_unix=time.time(),semantics='Actual logical1..200 once across retained namespaces; '
            'discarded returned and unreturned lower-bound writes remain separate original evidence; no rollback. '
            'The actual pipeline/session/cache/source audit after owned cleanup remains mandatory.')


def verify(base):
    from reproduction.common8.calibration_runtime import write_once
    base = Path(base).resolve(); entries = discover_chain(base); terminal = entries[-1]['run']
    _executed_source(base,terminal)
    assert not (base/ACCEPTANCE).exists() and not (base/ACCEPTANCE).is_symlink()
    segments = [verify_segment(Path(e['base']),e['run'],e['last']) for e in entries]
    calibration = verify_calibration_chain(entries,segments)
    first_five = verify_root_five(entries[0],segments[0])
    faults = {e['run']['run_id']:verify_fault_review(Path(e['base']),e['run']) for e in entries[1:]}
    predecessors = [_predecessor(Path(e['base']),e['run']) for e in entries]
    final = read_final_evaluation(base,terminal)
    proof = compose(entries,segments,calibration,first_five,faults,predecessors,final)
    from reproduction.common8.actor_probe_evidence_seals import fixed_logs
    proof['fixed_evidence_byte_seals']={e['run']['run_id']:fixed_logs(Path(e['base']),e['run']) for e in entries}
    proof['historical_owned_births']={e['run']['run_id']:birth_inventory(Path(e['base']),e['run']) for e in entries}
    write_once(base/ACCEPTANCE,proof)
    return proof


def after_cleanup(base):
    """No stop/delete actions; do not promote a marker to an actual exit proof."""
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.actor_probe_head_admission import verify_inputs_and_source
    base = Path(base).resolve(); run,_ = read_manifest(base); _executed_source(base,run)
    proof,file = _document(base,base/ACCEPTANCE)
    assert proof['role'] == 'actor_probe_actual_complete_retained_chain' and proof['identity'] == identity(run)
    assert proof['structural_accepted'] is True and proof['full_training_and_final_evaluation_verified'] is True
    assert proof['actual_outer_updates'] == 200 and proof['owned_cleanup_verified'] is False
    entries = discover_chain(base)
    assert [s['identity'] for s in proof['training_segments']] == [identity(e['run']) for e in entries]
    audits = []
    from reproduction.common8.actor_probe_evidence_seals import verify_fixed_logs,sealed_digest
    assert set(proof['fixed_evidence_byte_seals']) == {e['run']['run_id'] for e in entries}
    assert set(proof['historical_owned_births']) == {e['run']['run_id'] for e in entries}
    for entry, segment in zip(entries,proof['training_segments']):
        root = Path(entry['base']); current = entry['run']
        verify_inputs_and_source(root,current)
        verify_fixed_logs(root,current,proof['fixed_evidence_byte_seals'][current['run_id']])
        assert document_sha256(current['source_sha256']) == segment['source_map_sha256']
        # Rehash immutable files selected by the actual full acceptance. This
        # avoids rerunning training or adding another generated evaluation.
        for step,item in segment['checkpoint_proofs'].items():
            directory = Path(item['receipt']['checkpoint'])
            actual, _ = _document(root,directory/'complete.json')
            assert actual == item['receipt']
            verify_canonical_files(root,directory,actual)
            state, _ = _document(root,directory/'full-state-complete.json')
            assert state == segment['full_states'][step]
            assert digest(directory/'full-state-complete.json') == item['full_state_receipt_sha256']
            for row in state['workers']+[state['driver']]:
                path = Path(row['path']); assert path == path.resolve() and path.parent == directory
                from reproduction.common8.formal_inline_safety import owned_path
                assert owned_path(root,path.relative_to(root).as_posix()) == path
                assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
            probe = directory/'probe-state'
            assert digest(probe/'version-manifest.json') == state['probe_manifest_sha256']
            assert all(digest(probe/name)==sha for name,sha in state['probe_head_hashes'].items())
        audits.append(verify_cleanup(root,current,preserved_births=proof['historical_owned_births'][current['run_id']]))
    by_run={e['run']['run_id']:e for e in entries}
    for original in proof['original_calibration']['original_versions'].values():
        entry=by_run[original['original_identity']['run_id']];root=Path(entry['base'])
        assert original['original_identity'] == identity(entry['run'])
        archive=Path(original['original_archive']['path'])
        assert sealed_digest(root,archive) == {key:original['original_archive'][key] for key in ('bytes','sha256')}
        for name,sha in original['artifact_files_sha256'].items():
            assert sealed_digest(root,root/name)['sha256'] == sha
        encoder=Path(original['encoder'])
        assert encoder == Path(entry['run']['initial_model']) or encoder.is_relative_to(root)
        assert encoder == encoder.resolve()
        for name,sha in original['encoder_files_sha256'].items():
            assert Path(name).name == name and name not in ('.','..')
            assert sealed_digest(encoder,encoder/name)['sha256'] == sha
    root=Path(entries[0]['base']);gpu=proof['actual_root_first_five']['historical_gpu_occupancy']
    prefix=sealed_digest(root,root/('logs/gpu-samples-'+entries[0]['run']['run_id']+'.jsonl'),
        prefix_bytes=gpu['observed_complete_prefix_bytes'])
    assert prefix['sha256']==gpu['observed_complete_prefix_sha256'], 'Original first-five GPU sample prefix changed'
    final = read_final_evaluation(base,run)
    assert final == proof['final_evaluation']
    stage, stage_file = _document(base,base/'logs/pipeline-status.json')
    assert stage['run_id'] == run['run_id']
    expected = ('FINISHED_FULL200_RECOVERY_CHAIN_FINAL51713_AND_OWNED_CLEANUP'
                if run.get('actor_probe_recovery') is not None else 'FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP')
    assert stage['stage'] == expected
    result = dict(role='actor_probe_actual_complete_after_owned_cleanup',identity=identity(run),accepted=True,
        full_run_acceptance=file,pipeline_stage_file=stage_file,owned_cleanup_audits=audits,
        actual_outer_updates=200,final_count=51713,final_em=final['weighted_em'],final_f1=final['weighted_f1'],
        source_and_checkpoint_bytes_reverified=True,original_calibration_and_fixed_evidence_bytes_reverified=True,
        actual_all_owned_cleanup_verified=True,checked_unix=time.time(),
        goal_completion_requires_user_report=True)
    write_once(base/'logs/actor-probe-final-cleanup-acceptance.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--after-cleanup',action='store_true'); args = parser.parse_args()
    proof = after_cleanup(args.base) if args.after_cleanup else verify(args.base)
    # Do not forward full interval/label evidence through a local control path.
    print(json.dumps({key:proof[key] for key in ('role','identity','actual_outer_updates','checked_unix')},allow_nan=False))


if __name__ == '__main__': main()
