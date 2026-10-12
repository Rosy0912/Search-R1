"""Read-only actual process/cache/source audit after a local pipeline exits."""
import os
import socket
import time
from pathlib import Path

from reproduction.common8.actor_probe_rank_tail import _sealed_bytes
from reproduction.common8.policy_calibration_plan import _strict_json
from search_g1.actor_probe_recovery import identity


def _document(base, relative, limit=32*1024**2):
    raw, file = _sealed_bytes(base,relative,limit=limit)
    return _strict_json(raw), file


def birth_inventory(base,run):
    """Seal historical ownership without freezing mutable states/commands."""
    ledger,_=_document(Path(base),'logs/process-ledger.json',256*1024**2)
    assert ledger['run_id'] == run['run_id'] and ledger['jobs']
    keys=('pid','start_ticks','sid','uid');jobs={}
    for name,job in ledger['jobs'].items():
        leader=job['leader']
        rows=[leader]+list(job['ledger'].values())
        assert leader['pid']==leader['sid'] and leader['uid']==os.getuid()
        assert all(type(row[key]) is int for row in rows for key in keys)
        assert all(row['pid']>0 and row['start_ticks']>0 and
            (row['sid'],row['uid'])==(leader['sid'],leader['uid']) for row in rows)
        births=sorted({tuple(row[key] for key in keys) for row in rows})
        jobs[name]=dict(leader={key:leader[key] for key in keys},
            births=[dict(zip(keys,row)) for row in births])
    return dict(identity=identity(run),jobs=jobs)


def verify_preserved_births(saved,current):
    assert saved['identity']==current['identity'] and saved['jobs']
    keys=('pid','start_ticks','sid','uid')
    for name,job in saved['jobs'].items():
        actual=current['jobs'][name]
        assert job['leader']==actual['leader'], 'Owned job leader birth disappeared from the ledger'
        assert {tuple(row[key] for key in keys) for row in job['births']} <= \
            {tuple(row[key] for key in keys) for row in actual['births']}, \
            'Historical owned process births disappeared from the ledger'


def verify_cleanup(base, run, *, preserved_births=None):
    """A STOPPED label or cleanup marker alone cannot prove resources released."""
    from reproduction.a800.owned_processes import identity as process_identity
    from reproduction.common8.run_pipeline import gpu_rows
    base = Path(base)
    assert run['execution_host'] == socket.gethostname()
    cleanup, cleanup_file = _document(base,'logs/cleanup-status.json')
    assert cleanup['run_id'] == run['run_id'] and cleanup['execution_id'] == run['execution_id']
    assert cleanup['execution_backend'] == 'ssh_local' and cleanup['slurm_job_id'] is None
    assert cleanup['cleaned'] is True
    assert cleanup['scope'] == 'only recorded project PID births and private local corpus cache'
    ledger, ledger_file = _document(base,'logs/process-ledger.json',256*1024**2)
    if preserved_births is not None:
        verify_preserved_births(preserved_births,birth_inventory(base,run))
    claim, claim_file = _document(base,'logs/pipeline-launch-claim.json')
    assert ledger['run_id'] == claim['run_id'] == run['run_id']
    assert claim['execution_id'] == run['execution_id']
    if run.get('actor_probe_recovery') is not None:
        assert claim['source_commit'] == run['source_commit']
    producer = claim['pipeline_identity']
    assert producer == run['pipeline_identity']
    records = [producer]; session_leaders = []
    for job in ledger['jobs'].values():
        assert job['state'] == 'STOPPED'
        leader = job['leader']
        assert leader['sid'] == leader['pid'] and leader['uid'] == os.getuid()
        members = [leader]+list(job['ledger'].values())
        assert all((row['sid'],row['uid']) == (leader['sid'],leader['uid']) for row in members)
        records += members; session_leaders.append(leader)
    # Job leaders own private sessions. A pipeline launched from a shared
    # shell session owns only its exact process birth, not the parent shell
    # or unrelated tasks in that session.
    if producer['sid'] == producer['pid']:
        session_leaders.append(producer)
    assert all(type(row[key]) is int for row in records for key in ('pid','start_ticks','sid','uid'))
    assert all(row['pid']>0 and row['start_ticks']>0 and row['uid']==os.getuid() for row in records)
    for record in records:
        try: actual = process_identity(record['pid'])
        except (FileNotFoundError,ProcessLookupError): continue
        assert actual['state'] in ('Z','X') or any(actual[key] != record[key]
            for key in ('start_ticks','sid','uid')), 'An owned process birth remains live'
    from reproduction.a800.owned_processes import remaining_cuda_contexts, owned_session_survivors
    survivors,reused_sessions=owned_session_survivors(session_leaders)
    assert not survivors, 'An unrecorded owned session survivor remains live'
    remaining,reused=remaining_cuda_contexts(records,gpu_rows('gpu_uuid,pid','compute-apps'))
    assert not remaining, 'An owned CUDA context remains allocated after process exit'
    cache_claim, cache_file = _document(base,'logs/corpus-cache-claim.json')
    cache, cache_cleanup_file = _document(base,'logs/corpus-cache-cleanup.json')
    root = base/'cache/retrieval'/run['execution_id']
    assert cache_claim['run_id'] == cache['run_id'] == run['run_id']
    assert cache_claim['execution_id'] == cache['execution_id'] == run['execution_id']
    assert cache_claim['uid'] == os.getuid() and cache_claim['phase'] == 'CLEANED'
    assert cache_claim['root'] == cache['root'] == str(root)
    assert not root.exists() and not root.is_symlink()
    assert cache['cleaned'] is True and cache['original_assets_removed'] is False
    assert all(Path(path).exists() for path in run['retrieval_asset_paths'].values())
    assert Path(run['initial_model']).is_dir() and Path(run['reference_model']).is_dir()
    return dict(role='actor_probe_actual_owned_cleanup_audit',identity=identity(run),accepted=True,
        actual_pipeline_and_owned_sessions_exited=True,owned_cuda_contexts_remaining=0,
        private_cache_removed=True,original_assets_present=True,
        reused_pids_belonging_to_other_births=sorted(set(reused)),reused_session_ids=reused_sessions,
        recorded_births=len(records),
        files=[cleanup_file,ledger_file,claim_file,cache_file,cache_cleanup_file],checked_unix=time.time())
