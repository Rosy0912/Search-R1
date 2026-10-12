"""Read-only admission after the exact predecessor's acceptance and cleanup."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

from reproduction.common8.verify_data import digest, MANIFEST_SHA


# The human-authorized successor belongs to this currently running experiment,
# not to any interchangeable finished run with the same dataset and model.
EXPECTED_PREDECESSOR = dict(
    base='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab',
    run_id='1db662001a0f4d998928bcf6f5f33c9d',
    execution_id='4099fdc586b5467bb138e0ec7b35ca8a',
    source_commit='34e63b7b12f4056503803905c1b3dd1be95f0adc')


def validate_binding(binding):
    assert isinstance(binding, dict) and set(binding) == {
        'base', 'run_id', 'execution_id', 'source_commit',
        'full_acceptance_sha256', 'cleanup_sha256'}
    assert re.fullmatch(r'/data/shaoshitong/Search-G1-a800-fresh200-[0-9a-f]{32}', binding['base'])
    for key in ('run_id', 'execution_id'): assert re.fullmatch('[0-9a-f]{32}', binding[key])
    assert re.fullmatch('[0-9a-f]{40}', binding['source_commit'])
    for key in ('full_acceptance_sha256', 'cleanup_sha256'):
        assert re.fullmatch('[0-9a-f]{64}', binding[key])
    assert all(binding[key] == value for key, value in EXPECTED_PREDECESSOR.items()), \
        'Another old finished run cannot replace the human-authorized predecessor'
    return dict(binding)


def read_owned_json(path, *, limit=32*1024**2):
    path = Path(path)
    assert path.resolve() == path and path.stat().st_uid == os.getuid()
    assert path.is_file() and path.stat().st_size <= limit
    return json.loads(path.read_text())


def validate_documents(binding, child, parent, proof, cleanup, updates):
    assert parent['execution_backend'] == 'ssh_local'
    for key in ('run_id', 'execution_id', 'source_commit'):
        assert parent[key] == binding[key]
    assert parent['run_id'] != child['run_id'] and parent['execution_id'] != child['execution_id']
    assert parent['initial_model_sha256'] == child['initial_model_sha256']
    assert parent['model_revision'] == child['model_revision']
    assert parent['input_sha256'] == child['input_sha256']
    assert parent['input_sha256']['manifest.json'] == MANIFEST_SHA
    assert proof['run_id'] == parent['run_id'] and proof['structural_accepted'] is True
    assert proof['actual_outer_updates'] == 200 and proof['per_rank_optimizer_opportunities'] == 4000
    assert proof['final_saved_tensor_count'] == 435
    assert len(proof['final_counts']) == 7 and sum(proof['final_counts'].values()) == 51713
    assert proof['final_counts']['nq'] == 3610
    assert cleanup['run_id'] == parent['run_id'] and cleanup['execution_id'] == parent['execution_id']
    assert cleanup['cleaned'] is True and cleanup['execution_backend'] == 'ssh_local'
    assert cleanup['scope'] == 'only recorded project PID births and private local corpus cache'
    assert [row['step'] for row in updates] == list(range(1, 201))
    assert all(row['run_id'] == parent['run_id'] and row['opportunities_processed'] == 20
               and 1 <= row['actual_optimizer_writes'] <= 20 for row in updates)
    final_writes = updates[-1]['optimizer_writes_after']
    assert len(proof['per_rank_actual_optimizer_writes']) == parent['training_world_size']
    assert all(value == final_writes for value in proof['per_rank_actual_optimizer_writes'].values())
    return final_writes


def admit(base, child):
    """Never stop the parent, amend its ledger, or infer exit from a lock."""
    from search_g1.actor_probe_contract import actor_probes
    from reproduction.a800.owned_processes import identity
    if not actor_probes(child): return None
    binding = validate_binding(child['predecessor_completion'])
    parent_base = Path(binding['base']); base = Path(base).resolve()
    assert parent_base.resolve() == parent_base and parent_base != base
    parent = read_owned_json(parent_base/'logs/run-manifest.json')
    acceptance_path = parent_base/'logs/full-run-acceptance.json'
    cleanup_path = parent_base/'logs/cleanup-status.json'
    assert digest(acceptance_path) == binding['full_acceptance_sha256']
    assert digest(cleanup_path) == binding['cleanup_sha256']
    proof = read_owned_json(acceptance_path); cleanup = read_owned_json(cleanup_path)
    updates_path = parent_base/'logs/actual-updates.jsonl'
    assert updates_path.resolve() == updates_path and updates_path.stat().st_size < 32 * 1024**2
    updates = [json.loads(line) for line in updates_path.read_text().splitlines()]
    writes = validate_documents(binding, child, parent, proof, cleanup, updates)
    # The actual long-running predecessor ledger already exceeds32MiB. Keep
    # this one historical process inventory bounded, without relaxing other JSON.
    ledger = read_owned_json(parent_base/'logs/process-ledger.json', limit=256*1024**2)
    claim = read_owned_json(parent_base/'logs/pipeline-launch-claim.json')
    assert ledger['run_id'] == claim['run_id'] == parent['run_id']
    assert claim['execution_id'] == parent['execution_id']
    records = [claim['pipeline_identity']]
    session_leaders = []
    if records[0]['sid'] == records[0]['pid']:session_leaders.append(records[0])
    for job in ledger['jobs'].values():
        records += list(job['ledger'].values())
        session_leaders.append(job['leader'])
    for record in records:
        try: current = identity(record['pid'])
        except (FileNotFoundError, ProcessLookupError): continue
        assert current['state'] in ('Z', 'X') or any(current[k] != record[k]
            for k in ('start_ticks', 'sid', 'uid')), 'Predecessor process birth is still alive'
    # Detect a surviving unrecorded child in a predecessor-owned session too.
    # This inventory is read-only and never signals any process.
    from reproduction.a800.owned_processes import owned_session_survivors
    live,reused_sessions = owned_session_survivors(session_leaders)
    assert not live, 'Predecessor session still contains live processes'
    assert subprocess.check_output(['git', '-C', str(parent_base/'repo'), 'rev-parse', 'HEAD'],
                                  text=True).strip() == parent['source_commit']
    for name, sha in parent['source_sha256'].items():
        assert digest(parent_base/'repo'/name) == sha, 'Preserved predecessor source differs'
    for step in (5, 50, 100, 150, 200):
        checkpoint = Path(parent['checkpoint_dir'])/'actor'/f'global_step_{step}'
        assert read_owned_json(checkpoint/'complete.json') == proof['checkpoint_proofs'][str(step)]
    return dict(accepted=True, predecessor=binding, actual_complete_outers=200,
        actual_optimizer_writes=writes, checked_unix=time.time(),
        recorded_births_checked=len(records), surviving_session_processes=0,reused_session_ids=reused_sessions,
        scope='exact accepted predecessor and only-owned cleanup; no inherited model or optimizer state')
