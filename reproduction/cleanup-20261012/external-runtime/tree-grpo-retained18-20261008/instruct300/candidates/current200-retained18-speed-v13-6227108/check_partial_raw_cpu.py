"""Future h100-10 CPU verification of retained actual raw partial states.

Local development tests use plain dict fixtures and non-PT tiny files only.
This checker never restores a worker, performs a forward, or writes an optimizer.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

from partial_raw_evidence import (held_ordinary_file, partial_directory, raw_file_records,
    read_json_bound, require, stream_sha256, verify_partial_raw_files,
    verify_worker_logical, verify_driver_logical, verify_actor_inputs_logical)


def held_state(path, expected_sha, loader, expected_bytes=None):
    """Hash and deserialize the very same held FD, with post-load stat binding."""
    with held_ordinary_file(path) as (stream, before):
        require(expected_bytes is None or before.st_size == expected_bytes, 'Raw bound byte count changed')
        require(stream_sha256(stream) == expected_sha, 'Raw bound SHA changed before deserialization')
        stream.seek(0)
        state = loader(stream, map_location='cpu', weights_only=False)
    return state


def _history_before(event, entries, completed, partial, refused):
    return dict(protocol='tree_guard_continue_fresh_rollout_v1', training_attempts=event['attempt'],
        refused_attempts=refused, guard_refused_attempts=sum(entry['attempt_id'] < event['attempt'] for entry in entries),
        completed_outer_updates=event.get('outer', event.get('candidate_outer')) - 1,
        partial_optimizer_calls_per_rank=dict(partial), completed_optimizer_calls_per_rank=dict(completed),
        total_optimizer_calls_per_rank={str(rank): completed.get(str(rank), 0) + partial.get(str(rank), 0) for rank in range(8)},
        partial_evidence=[entry for entry in entries if entry['attempt_id'] < event['attempt']])


def verify(run, package, config_hash, outer, output):
    from attempt_accounting import attempt_results, checkpoint_accounting
    from production_contract import checkpoint_manifest, require_identity
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Partial checker must be explicitly CPU-only')
    run = Path(run)
    require(run.is_absolute(), 'Actual run must be absolute')
    start, cpu = time.perf_counter(), time.process_time()
    checkpoint_path = run / 'checkpoints/training_state' / f'global_step_{outer}' / 'manifest.json'
    pinned_checkpoint, checkpoint_sha, _ = read_json_bound(checkpoint_path)
    manifest, checked_sha = checkpoint_manifest(run, package, config_hash, outer)
    require(manifest == pinned_checkpoint and checked_sha == checkpoint_sha, 'Complete checkpoint changed during accounting validation')
    observed = attempt_results(run, package, config_hash, outer)
    accounting = checkpoint_accounting(manifest['identity']['guard_continue_accounting'], observed, outer, run)
    # Accounting bindings may include raw PT paths as well as event JSON.
    # Verify all of them physically without parsing/deserializing their bytes.
    for path, expected in observed['record_sha256'].items():
        with held_ordinary_file(path) as (stream, _):
            require(stream_sha256(stream) == expected, 'Actual event/accounting binding changed')
    expected_identity = dict(job_id=run.name.rsplit('-', 1)[-1], run_id=run.name,
        code_commit=package['code_commit'], config_sha256=config_hash,
        data_identity=package['common_data_identity'], model_identity=package['model_identity']['sha256'])
    entries = accounting['partial_evidence']
    events = sorted(observed['credited'] + observed['all_skipped'], key=lambda event: event['attempt'])
    require(len(events) == observed['training_attempts'] and
            [event['attempt'] for event in events] == list(range(1, observed['training_attempts'] + 1)),
            'Every actual attempt through the completed boundary must be checked')
    by_attempt = {entry['attempt_id']: entry for entry in entries}
    require(len(by_attempt) == len(entries) and set(by_attempt) <= {event['attempt'] for event in events},
            'Partial evidence has a duplicate or lacks an actual event')
    partials, raw_mapping, manifest_mapping = {}, {}, {}
    completed, partial_totals, refused = {}, {}, 0
    for event in events:
        attempt = event['attempt']
        writes = event.get('actual_writes_per_rank', {str(rank): 0 for rank in range(8)})
        if attempt in by_attempt:
            entry = by_attempt[attempt]
            candidate = event.get('outer', event.get('candidate_outer'))
            expected_path = run / 'checkpoints/training_state' / f'partial_attempt_{attempt:06d}_outer_{candidate:06d}' / 'manifest.json'
            require(Path(entry['path']) == expected_path, 'Saved partial manifest must match the exact attempt/outer path')
            value, msha, _ = read_json_bound(expected_path, entry['sha256'])
            require_identity(value['identity'], expected_identity)
            require(partial_directory(run, value['identity']) / 'manifest.json' == expected_path and
                    value['continuation_decision'] == 'CONTINUE_FRESH_ROLLOUT' and
                    value['credited_outer_candidate'] == ('outer' in event) and
                    value['actual_optimizer_calls_per_rank'] == entry['optimizer_calls_per_rank'] == writes,
                    'Actual partial identity/credit/writes differ from event history')
            bindings = verify_partial_raw_files(run, value)
            require(not set(raw_mapping).intersection(bindings), 'Raw partial path appears in more than one attempt')
            raw_mapping.update(bindings)
            manifest_mapping[str(expected_path)] = msha
            prior = _history_before(event, entries, completed, partial_totals, refused)
            cumulative = {str(rank): prior['total_optimizer_calls_per_rank'][str(rank)] + writes[str(rank)] for rank in range(8)}
            partials[attempt] = (value, prior, cumulative)
        if 'outer' in event:
            completed = {str(rank): completed.get(str(rank), 0) + writes[str(rank)] for rank in range(8)}
        else:
            partial_totals = {str(rank): partial_totals.get(str(rank), 0) + writes[str(rank)] for rank in range(8)}
            refused += 1
    logical_records = []
    deserializations = 0
    torch = None
    if partials:
        require(sys.platform.startswith('linux') and socket.gethostname().split('.')[0] == 'h100-10',
                'Actual PT deserialization is restricted to same h100-10 CPU; never Mac tensor assets')
        import torch
        from search_r1.baselines.training_state import digest, finite
        torch.set_num_threads(8)
        require(torch.cuda.is_initialized() is False, 'CPU checker started with initialized CUDA')
        for attempt, (value, prior, cumulative) in partials.items():
            workers = []
            records = raw_file_records(run, value)
            for path, record, _ in records[:8]:
                state = held_state(path, record['sha256'], torch.load, record['bytes'])
                workers.append(verify_worker_logical(state, value, record, digest, finite,
                    cumulative_calls=cumulative[str(record['rank'])]))
                deserializations += 1
                del state
            path, record, _ = records[8]
            state = held_state(path, record['sha256'], torch.load, record.get('bytes'))
            driver = verify_driver_logical(state, value, digest, finite, prior_accounting=prior)
            deserializations += 1
            del state
            path, record, _ = records[9]
            state = held_state(path, record['sha256'], torch.load, record['bytes'])
            actor_inputs = verify_actor_inputs_logical(state, run, value)
            deserializations += 1
            del state
            logical_records.append(dict(attempt_id=attempt, candidate_outer=value['identity']['attempted_outer'],
                identity=value['identity'], partial_manifest_sha256=manifest_mapping[str(partial_directory(run, value['identity']) / 'manifest.json')],
                workers=workers, driver=driver, actor_inputs=actor_inputs))
        require(torch.cuda.is_initialized() is False, 'Partial CPU checker initialized CUDA')
    report = dict(status='FRESH200_PARTIAL_RAW_CPU_REVERIFIED', identity=expected_identity,
        outer=outer, completed_outer=outer, checkpoint_manifest_sha256=checkpoint_sha,
        raw_files_sha256=raw_mapping, partial_manifest_sha256=manifest_mapping,
        logical_records=logical_records, raw_files_checked=len(raw_mapping),
        actual_attempts_checked=len(events), partial_attempts_checked=len(partials),
        actual_event_sha256=observed['record_sha256'], actual_pt_deserialization=bool(deserializations),
        pt_deserialization_count=deserializations, same_fd_io=True,
        same_fd_io_details=dict(no_follow_all_ancestors=True, ordinary_files_only=True,
            physical_sha_streamed=True, hash_and_deserialize_same_fd=True, stable_fd_and_path_stat=True),
        optimizer_writes=0, model_forwards=0, cuda_initialized=False,
        resume_permitted=False, automatic_resume_supported=False, automatic_rollback=False,
        live_reload_verified_by_this_checker=False, behavior_mu_binding_owned_by='check_behavior_mu_cpu.py',
        empty_partial_evidence=(len(partials) == 0), wall_seconds=time.perf_counter() - start,
        process_cpu_seconds=time.process_time() - cpu)
    require(deserializations == len(raw_mapping) == 10 * len(partials), 'Raw physical/logical coverage differs')
    with Path(output).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package, _, _ = read_json_bound(Path(args.package) / 'manifest.json')
    print(json.dumps(verify(args.run, package, args.config_sha, args.outer, args.output), sort_keys=True))
