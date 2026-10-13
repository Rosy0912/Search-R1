"""Audit an actual retained training interval in its original namespace.

The complete chain caller still has to verify original calibration provenance,
root first-five acceptance, all logical owners, final51713 and owned cleanup.
This module starts no process and never writes or rewrites training evidence.
"""
import math
import os
import re
import socket
from pathlib import Path

from reproduction.common8.actor_probe_chain import account_segment
from reproduction.common8.actor_probe_rank_tail import _sealed_bytes, _sealed_log, _context
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from search_g1.actor_probe_contract import actor_probes
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import identity


def read_manifest(base):
    base = Path(base)
    assert base.is_absolute() and base == base.resolve()
    raw, file = _sealed_bytes(base, 'logs/run-manifest.json', limit=32*1024**2)
    run = _strict_json(raw)
    for key, length in (('run_id',32), ('execution_id',32), ('source_commit',40)):
        assert isinstance(run[key],str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',run[key])
    assert actor_probes(run) and run['execution_backend'] == 'ssh_local'
    assert type(run['training_world_size']) is int and run['training_world_size'] == 4
    return run, file


def discover_chain(base):
    """Read exact parent manifests; do not accept caller-supplied ancestor rows."""
    entries = []; seen_bases = set(); seen_runs = set(); last = 200
    while True:
        base = Path(base)
        assert str(base) not in seen_bases, 'Cyclic parent namespace'
        run, file = read_manifest(base)
        assert run['run_id'] not in seen_runs, 'Reused run identity across namespaces'
        seen_bases.add(str(base)); seen_runs.add(run['run_id'])
        origin = namespace_origin(run)
        assert origin['completed_outer'] < last, 'Recovery namespace has no retained interval'
        entries.append(dict(base=str(base), run=run, manifest_file=file, last=last, origin=origin))
        binding = run.get('actor_probe_recovery')
        if binding is None:
            assert origin == dict(completed_outer=0,actual_optimizer_writes=0,attempt_index=0,aborted_attempts=0)
            break
        parent_base = Path(binding['parent_base'])
        parent, parent_file = read_manifest(parent_base)
        assert parent_file['sha256'] == binding['parent_manifest_sha256']
        assert all(parent[key] == binding['parent_'+key] for key in ('run_id','execution_id','source_commit'))
        last = origin['completed_outer']; base = parent_base
        assert len(entries) <= 4, 'More retained recovery boundaries than the four saved milestones'
    entries.reverse(); root = entries[0]['run']['run_id']
    for entry in entries:
        run = entry['run']
        assert run.get('actor_probe_experiment_root_run_id',run['run_id']) == root
        if run.get('actor_probe_recovery') is not None:
            assert run['actor_probe_recovery']['experiment_root_run_id'] == root
    return entries


def verify_trainer_exit(base, run, *, terminal=False):
    """Prove actual owned births and session members exited, without signals."""
    from reproduction.a800.owned_processes import identity as process_identity
    assert run['execution_host'] == socket.gethostname(), 'Local /proc cannot prove another host exited'
    raw, file = _sealed_bytes(Path(base), 'logs/process-ledger.json', limit=256*1024**2)
    ledger = _strict_json(raw); assert ledger['run_id'] == run['run_id']
    job = ledger['jobs']['trainer']; leader = job['leader']
    assert all(type(leader[key]) is int for key in ('pid','start_ticks','sid','uid'))
    assert leader['pid'] > 0 and leader['start_ticks'] > 0 and leader['sid'] == leader['pid']
    assert leader['uid'] == os.getuid()
    claim_raw, claim_file = _sealed_bytes(Path(base), 'logs/training-launch-claim.json', limit=32*1024**2)
    claim = _strict_json(claim_raw)
    assert claim['run_id'] == run['run_id'] and claim['execution_id'] == run['execution_id']
    assert (claim['shell_pid'],claim['shell_start_ticks']) == (leader['pid'],leader['start_ticks'])
    if terminal:
        assert type(job.get('exit_code')) is int and job['exit_code'] == 0, 'Terminal trainer did not return success'
    checked = set()
    for record in list(job['ledger'].values())+[leader]:
        assert record['uid'] == leader['uid'] and record['sid'] == leader['sid']
        assert all(type(record[key]) is int for key in ('pid','start_ticks','sid','uid'))
        checked.add((record['pid'],record['start_ticks']))
        try: actual = process_identity(record['pid'])
        except (FileNotFoundError,ProcessLookupError): continue
        assert actual['state'] in ('Z','X') or any(actual[key] != record[key]
            for key in ('start_ticks','sid','uid')), 'Owned trainer birth remains live'
    from reproduction.a800.owned_processes import owned_session_survivors
    survivors,_=owned_session_survivors([leader])
    assert not survivors, 'Unrecorded trainer session survivor remains live'
    return dict(actual_trainer_exit_verified=True, recorded_births=len(checked),
        terminal_exit_code=job.get('exit_code'), files=[file,claim_file])


def verify_refresh_isolation(base, run, records, states):
    """Bind each due refresh to the version actually used next or durably saved."""
    by_step = {row['step']:row for row in records}; origin = namespace_origin(run)['completed_outer']
    result = {}
    for step in range(10,201,10):
        if not origin < step <= records[-1]['step']: continue
        raw, file = _sealed_bytes(Path(base), f'logs/actor-probe-refresh-isolation-{step}.json',limit=32*1024**2)
        proof = _strict_json(raw)
        assert proof['accepted'] is True and type(proof['step']) is int and proof['step'] == step
        before = proof['before']; assert isinstance(before,dict) and set(before) == {'actor','reference','driver_rng_sha256'}
        assert isinstance(before['driver_rng_sha256'],str) and re.fullmatch('[0-9a-f]{64}',before['driver_rng_sha256'])
        assert before == proof['after']
        for role in ('actor','reference'):
            assert isinstance(before[role],list) and sorted(row['rank'] for row in before[role]) == list(range(4))
            for row in before[role]:
                assert type(row['rank']) is int and row['role'] == role
                keys = ['parameters_sha256','gradients_sha256','module_modes_sha256','rng_sha256']
                if role == 'actor':keys += ['optimizer_sha256','scheduler_sha256','rollout_rng_sha256']
                for key in keys:
                    assert isinstance(row[key],str) and re.fullmatch('[0-9a-f]{64}',row[key])
        version = (by_step[step+1]['probe_version'] if step+1 in by_step else states[str(step)]['probe_version'])
        assert proof['version'] == version, 'Refresh isolation belongs to another adopted Head version'
        result[str(step)] = dict(version=version, file=file)
    return result


def verify_canonical_files(base,directory,proof):
    from reproduction.common8.formal_inline_safety import owned_path
    from reproduction.common8.verify_data import digest
    assert directory == directory.resolve() and directory.is_relative_to(base)
    assert Path(proof['checkpoint']) == directory
    assert isinstance(proof['files_sha256'],dict) and proof['files_sha256']
    for name,sha in proof['files_sha256'].items():
        assert Path(name).name == name and name not in ('.','..')
        assert isinstance(sha,str) and re.fullmatch('[0-9a-f]{64}',sha)
        path = owned_path(base,(directory/name).relative_to(base).as_posix())
        assert path == path.resolve() and path.stat().st_uid == os.getuid()
        assert digest(path) == sha, 'Actual canonical bytes differ from the saved file manifest'


def verify_segment(base, run, last):
    """Complete actual interval checks; calibration/final chain remain separate."""
    from reproduction.common8.actor_probe_head_admission import verify_inputs_and_source
    from reproduction.common8.actor_probe_retained_prefix import load_prefix
    from reproduction.common8.actor_probe_restore_evidence import read_origin
    from reproduction.common8.formal_contract import assert_formal_run, verify_formal_config, SAVE_STEPS
    from reproduction.common8.formal_inline_safety import owned_path, read, verify_plan
    from reproduction.common8.topology import validate_base
    from reproduction.common8.verify_data import verify, digest
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    from reproduction.common8.runtime import budget_rows, verify_scheduler_receipts
    from reproduction.common8.worker_evidence import read_console_form
    from reproduction.common8.formal_attempt_audit import verify_attempt_history
    from reproduction.common8.actor_probe_evidence import verify_actor_readouts
    from reproduction.common8.actor_probe_checkpoint import verify_full_state
    from reproduction.common8.actor_probe_rank_tail import read_unreturned_tail
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint

    base = Path(base); validate_base(base,run); assert_formal_run(run)
    assert read_manifest(base)[0] == run
    assert type(last) is int and last in (*SAVE_STEPS,)
    verify_inputs_and_source(base,run)
    for name,sha in run['initial_model_sha256'].items():
        assert Path(name).name == name and digest(Path(run['initial_model'])/name) == sha
    data = verify(base/'data'); exit_proof = verify_trainer_exit(base,run,terminal=last==200)
    origin = namespace_origin(run); first = origin['completed_outer']+1
    prefix = load_prefix(base,run,last) if last<200 else None
    if prefix is not None:
        segment = prefix['segment']; records = prefix['selected_updates']
    else:
        records, updates_file = _sealed_log(base,'logs/actual-updates.jsonl')
        attempts, attempts_file = _sealed_log(base,'logs/actual-attempts.jsonl')
        assert updates_file['trailing_fragment_bytes'] == attempts_file['trailing_fragment_bytes'] == 0
        segment = account_segment(run['run_id'],records,attempts,retained_through=200,
            start_outer=origin['completed_outer'],start_writes=origin['actual_optimizer_writes'],
            start_attempt_index=origin['attempt_index'])
        assert records == segment['retained_updates'] and not segment['discarded_attempts']
    assert [row['step'] for row in records] == list(range(first,last+1))
    assert all(type(value) in (int,float) and math.isfinite(value) for row in records for value in row['metrics'].values())
    resolved = read(owned_path(base,'logs/resolved-training-config.json'))
    assert resolved['run_id'] == run['run_id']
    config = verify_formal_config(resolved['resolved_config'],run,base)
    assert resolved['resolved_config_sha256'] == config['resolved_config_sha256']
    plan = read(owned_path(base,'logs/policy-safety-calibration.json')); _, plan_sha = verify_plan(base,plan,run)
    origin_evidence = read_origin(base,run)
    launch = None
    if run.get('actor_probe_recovery') is not None:
        from reproduction.common8.actor_probe_launch_admission import verify as verify_launch, verify_claim_binding
        launch = verify_launch(base,run)
        verify_claim_binding(base,run,read(owned_path(base,'logs/training-launch-claim.json')),launch)
    gates = {str(row['step']):read_gate_receipt(base,run,row['step'],update_record=row) for row in records}
    verify_write_receipt_links(base,gates,first,last,retained_prefix=prefix)
    budgets = budget_rows(read_console_form(base,run['run_id'],retained_prefix=prefix),first,last,
        complete_records=records,run=run)
    schedulers = verify_scheduler_receipts(base,run,first,last,gates,retained_prefix=prefix)
    attempt_audit = verify_attempt_history(base,run,records,schedulers,last=last,
        plan_sha256=plan_sha,retained_prefix=prefix)
    readouts = verify_actor_readouts(base,run,records,retained_prefix=prefix)
    states = {}; checkpoints = {}
    for step in SAVE_STEPS:
        if not origin['completed_outer'] < step <= last: continue
        selection = load_prefix(base,run,step) if prefix is not None else None
        states[str(step)] = verify_full_state(base,run,step,load_optimizer=True,retained_prefix=selection)
        directory = Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
        assert directory == directory.resolve() and directory.is_relative_to(base)
        proof = read(owned_path(base,(directory/'complete.json').relative_to(base).as_posix()))
        assert proof['run_id'] == run['run_id'] and proof['step'] == step and proof['finite_all_saved_tensors'] is True
        verify_canonical_files(base,directory,proof)
        count = validate_saved_checkpoint(directory,Path(run['initial_model']),run['started_unix'])
        assert count == 435
        checkpoints[str(step)] = dict(receipt=proof,finite_tensor_count=count,
            full_state_receipt_sha256=digest(directory/'full-state-complete.json'))
        if step+1 in {row['step'] for row in records}:
            assert states[str(step)]['probe_version'] == next(row for row in records if row['step']==step+1)['probe_version']
    isolation = verify_refresh_isolation(base,run,records,states)
    tail = read_unreturned_tail(base,run,retained_through=last)
    if last == 200:
        assert tail['unreturned_attempt'] is None
        assert all(not row['scheduler_tail_kinds'] and not row['recorded_write_positions'] for row in tail['per_rank'].values())
        assert not tail['requires_fault_review'] and all(file.get('trailing_fragment_bytes',0)==0 for file in tail['files'])
    return dict(role='actor_probe_actual_training_segment',identity=identity(run),base=str(base),
        first=first,last=last,source_map_sha256=document_sha256(run['source_sha256']),shared_data=data,
        counter_segment=segment,retained_prefix=prefix,origin_evidence=origin_evidence,
        trainer_exit=exit_proof,launch_admission=launch,formal_config=config,formal_inline_plan_sha256=plan_sha,
        behavior_probability_gates=gates,budgets=budgets,scheduler_receipts=schedulers,formal_inline_all_attempts=attempt_audit,
        actor_readouts=readouts,full_states=states,checkpoint_proofs=checkpoints,refresh_isolation=isolation,
        discarded_rank_tail=tail,actual_interval_accepted=True,full_training_or_final_acceptance=False,
        remaining='original probe calibration provenance, root first5, logical1..200 chain/final51713 and owned cleanup')
