"""CPU launcher after all four arms finish; never changes training namespaces."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
try:
    from .common import (require, read_json, read_rows, check_file, file_ref, exclusive_json,
        remote_output, bind_training, checked_endpoint, cpu_only, VARIANTS, ENDPOINTS)
except ImportError:
    from common import (require, read_json, read_rows, check_file, file_ref, exclusive_json,
        remote_output, bind_training, checked_endpoint, cpu_only, VARIANTS, ENDPOINTS)


def verify_analysis_publication(root, manifest_ref, publication_ref):
    manifest, mp = read_json(manifest_ref); publication, pp = read_json(publication_ref)
    require(publication.get('accepted') is True and publication['branch'] == 'search-e1'
        and len(publication['source_commit']) == 40 and publication['source_commit'] == publication['remote_readback_sha']
        and publication['source_manifest_sha256'] == mp['sha256'], 'Independent analysis source lacks publication/readback')
    root = Path(root).resolve(); expected = set(manifest['files'])
    require({str(p.relative_to(root)) for p in root.rglob('*.py')} == {n for n in expected if n.endswith('.py')},
        'Unpublished analysis Python inventory')
    for name, entry in manifest['files'].items():
        rel = Path(name); require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe analysis source path')
        check_file(dict(path=str(root / rel), sha256=entry['sha256'] if isinstance(entry, dict) else entry))
    return dict(accepted=True, manifest=mp, publication=pp)


def resource_boundary(config, modules, owner, run_id, released=False):
    """Fresh device/PID/directfree checks, no signals and no retired observer."""
    gates = modules['gates']
    require(gates.same_process(gates.process(owner['pid']), owner) and owner['run_id'] == run_id
        and owner['uid'] == config['owner']['uid'], 'Independent replay CPU owner changed')
    oracle = gates.retriever_admission(config, gates.bound(config['gates']['retriever_admission']))
    devices, apps = gates.inventory()
    selected = [d for d in devices if d['index'] in config['owner']['physical_gpus']]
    require([d['index'] for d in selected] == config['owner']['physical_gpus']
        and [d['uuid'] for d in selected] == config['owner']['gpu_uuids'], 'Four-card mapping/UUID changed')
    owned = []
    for app in apps:
        if app['gpu_uuid'] not in config['owner']['gpu_uuids']: continue
        require(app['pid'] != owner['pid'], 'Replay CPU parent initialized CUDA')
        if app['pid'] == oracle['owner']['pid']:
            require(app['gpu_uuid'] == config['owner']['gpu_uuids'][-1] and app['used_mib'] <= 1024,
                'Original approved GPU7 oracle sharing changed')
        else:
            owned.append(gates.owned_gpu_process(app['pid'], run_id, owner, []))
    require(not released or not owned, 'Previous replay GPU worker remains')
    floor = config['resources']['startup_minimum_free_mib'] if released else config['resources']['training_minimum_free_mib']
    require(all(d['actual_free_mib'] >= floor for d in selected), 'Replay actual memory.free below original floor')
    return dict(accepted=True, checked_unix=time.time(), devices=selected, owned=owned, oracle=oracle,
        released=released, no_signals_sent=True)


def owned_release(config, modules, owner, run_id, children, helper_root, output):
    scanner = importlib.import_module(modules['gates'].__package__ + '.owned_process_audit')
    def environment(pid):
        result = scanner.proc_environment(pid)
        if pid == owner['pid']:
            require(modules['gates'].same_process(scanner.proc_snapshot(pid), owner), 'Replay release owner reused')
            result = dict(result); result.pop('E1_RUN_ID', None)
        return result
    result = scanner.audit_same_uid(run_id, owner['uid'], owner['starttick'], children,
        [str(Path(helper_root).resolve()), str(Path(output).resolve())],
        exclusions=scanner.load_exclusions(config.get('gates', {}).get('process_exclusions', [])),
        ignored_exact=[owner], environment_reader=environment, ancestry_anchors=[owner])
    require(result['accepted_no_owned_or_opaque'], 'Replay CPU/GPU descendant remains; retain lock')
    return dict(accepted=True, audit=result, no_signals_sent=True)


def verify_four_group_terminal(value, modules):
    require(value.get('accepted') is True and value['completed_groups'] == 4 and value['completed_outers'] == 100
        and value['full_training_em_rows'] == 100 and len(value['groups']) == 4,
        'All four train+15/25 evaluation groups must finish before replay')
    require([g['variant'] for g in value['groups']] == list(VARIANTS), 'Study group order/coverage changed')
    for group in value['groups']:
        require(group['actual_wait_returncode'] == 0 and group['release']['accepted'] is True,
            'Study group actual wait/release incomplete')
        actual, _ = read_json(group['actual_terminal'])
        require(actual['actual_wait_returncode'] == 0, 'Study controller actual terminal nonzero/unknown')
        summary, _ = read_json(group['summary'])
        require(summary.get('accepted') is True and summary['variant'] == group['variant'],
            'Study TRAIN summary mismatch')
    owner = value['actual_cpu_supervisor']
    try: current = modules['gates'].process(owner['pid'])
    except (FileNotFoundError, ProcessLookupError): current = None
    require(current is None or not modules['gates'].same_process(current, owner),
        'Study supervisor is still actually live; replay must wait')
    return dict(accepted=True, train_groups=4, completed_outers=100, old_supervisor_released=True)


def run(plan_ref, output):
    cpu_only(); plan, pp = read_json(plan_ref)
    require(plan['schema'] == 'opsd-signal-heldout-replay-plan-v1', 'Wrong independent replay plan')
    root = remote_output(output); require(not root.exists(), 'Replay attempt must be a new directory')
    require(os.environ.get('E1_RUN_ID') == plan['analysis_run_id'], 'Independent replay E1_RUN_ID differs')
    publication = verify_analysis_publication(plan['helper_root'], plan['source_manifest'], plan['publication'])
    require(Path(__file__).resolve() == Path(plan['helper_root']).resolve() / 'replay.py', 'Replay entry source path differs')
    configs = {}
    for ref in plan['configs']:
        cfg, cp = read_json(ref); cfg['_config_sha256'] = cp['sha256']
        require(cfg['variant'] not in configs, 'Duplicate replay config')
        configs[cfg['variant']] = (cfg, cp)
    require(set(configs) == set(VARIANTS), 'Replay needs all4 published immutable configs')
    cfg, cp = configs['G_POS']; modules = bind_training(cfg); owner = modules['gates'].process(os.getpid())
    bank, bp = read_json(plan['bank']); terminal, tp = read_json(plan['four_group_terminal'])
    require(bank['accepted'] is True and bank['frozen_before_all_group_15_25_outputs'] is True
        and bank['study_id'] == cfg['study_id'] and bank['configs'] == {v: x[1] for v, x in configs.items()},
        'Frozen bank/group identities differ')
    locks = []
    paths = [str(Path(cfg['paths']['study_base']) / 'study-sequence.lock')]
    paths += cfg['resources']['shared_gpu_lease_paths'] + [cfg['resources']['study_lease_path']]
    paths += [str(Path(configs[v][0]['paths']['base']) / 'online-experiment.lock') for v in VARIANTS]
    paths += [str(Path(cfg['paths']['study_base']) / 'analysis' / 'heldout-replay.lock')]
    require(len(paths) == len(set(paths)), 'Replay resource leases alias')
    jobs = []; opened_logs = []; attempt_created = False
    try:
        for path in paths:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            handle = Path(path).open('a+'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
        verify_four_group_terminal(terminal, modules)
        root.mkdir(parents=True); attempt_created = True
        modules['gates'].storage_admission(cfg, modules['gates'].bound(cfg['gates']['storage']))
        initial = resource_boundary(cfg, modules, owner, plan['analysis_run_id'], released=True)
        exclusive_json(root / 'started.json', dict(accepted=True, plan=pp, bank=bp, four_group_terminal=tp,
            owner=owner, resource_admission=initial, publication=publication, new_samples_or_training=0))
        endpoints = []; seen = set()
        for ref in plan['endpoints']:
            raw, _ = read_json(ref); version = raw['version']; t = raw['completed_outer']
            endpoint_cfg, endpoint_cp = configs['G_POS' if t == 0 else version]
            require((version, t) not in seen and (version, t) in ENDPOINTS, 'Duplicate/foreign replay endpoint')
            seen.add((version, t))
            value, ep, records, data = checked_endpoint(endpoint_cfg, ref, modules)
            checkpoint = None if t == 0 else value['raw_checkpoint']['manifest']
            identity, _ = modules['evaluation'].input_identity(endpoint_cfg, t, checkpoint)
            require(identity['expected_adapter_tensor_sha256'] == value['input_identity']['expected_adapter_tensor_sha256']
                and identity['validation']['sha256'] == bank['validation_identity']['sha256'], 'Actual replay raw adapter/data changed')
            endpoints.append((endpoint_cfg, endpoint_cp, value, ep, identity))
        require(seen == ENDPOINTS, 'All9 accepted formal endpoints must be present before GPU replay')
        outputs = []
        for endpoint_cfg, endpoint_cp, endpoint, ep, identity in sorted(endpoints, key=lambda x: (x[2]['version'], x[2]['completed_outer'])):
            resource_boundary(cfg, modules, owner, plan['analysis_run_id'], released=True)
            directory = root / (endpoint['version'] + '-' + str(endpoint['completed_outer'])); directory.mkdir()
            request = dict(schema='opsd-signal-heldout-replay-request-v1', analysis_run_id=plan['analysis_run_id'],
                config=endpoint_cp, endpoint=ep, bank=bp, input_identity=identity, controller=owner,
                version=endpoint['version'], completed_outer=endpoint['completed_outer'], output=str(directory),
                analysis_helper_root=plan['helper_root'], analysis_source_manifest=plan['source_manifest'],
                analysis_publication=plan['publication'], four_group_terminal=tp, new_samples_or_training=0)
            request_ref = exclusive_json(directory / 'request.json', request)
            jobs = []; exits = []; all_rows = []; worker_proofs = []
            for slot, physical in enumerate(cfg['owner']['physical_gpus']):
                log = (directory / ('worker-%d.log' % slot)).open('x'); opened_logs.append(log)
                command = [cfg['paths']['python'], '-B', str(Path(plan['helper_root']) / 'replay_worker.py'),
                    '--request', request_ref['path'], '--request-sha256', request_ref['sha256'], '--slot', str(slot)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical), E1_RUN_ID=plan['analysis_run_id'], PYTHONDONTWRITEBYTECODE='1')
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((process, None, slot))  # Hold leases even if /proc binding fails immediately after spawn.
                actual = modules['gates'].process(process.pid)
                require(actual['uid'] == 1001 and actual['run_id'] == plan['analysis_run_id'], 'Replay child identity differs')
                jobs[-1] = (process, actual, slot)
            exclusive_json(directory / 'children-start.json', dict(jobs=[dict(identity=i, slot=s) for _, i, s in jobs]))
            with (directory / 'memory.samples.jsonl').open('x') as memory:
                while any(p.poll() is None for p, _, _ in jobs):
                    snapshot = resource_boundary(cfg, modules, owner, plan['analysis_run_id'], released=False)
                    memory.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False) + '\n'); memory.flush()
                    time.sleep(1)
            for process, actual, slot in jobs:
                exits.append(dict(slot=slot, identity=actual, actual_wait_returncode=process.wait()))
            terminal_ref = exclusive_json(directory / 'actual-workers-terminal.json', dict(child_exit_evidence=exits))
            require(all(x['actual_wait_returncode'] == 0 for x in exits), 'Replay actual worker exit failed; preserve this attempt')
            release = owned_release(cfg, modules, owner, plan['analysis_run_id'], [x[1] for x in jobs], plan['helper_root'], root)
            released = resource_boundary(cfg, modules, owner, plan['analysis_run_id'], released=True)
            for _, actual, slot in jobs:
                complete, wp = read_json(file_ref(directory / ('worker-' + str(slot)) / 'complete.json'))
                require(complete.get('accepted') is True and complete['rows'] == 128 and complete['process'] == actual
                    and complete['adapter_unchanged'] is True, 'Replay worker final identity/coverage differs')
                rr, rrp = read_rows(complete['results']); all_rows.extend(rr); worker_proofs.extend([wp, rrp])
            require(len(all_rows) == 512 and {r['validation_index'] for r in all_rows} == set(range(512)), 'Replay512 incomplete')
            output_rows = directory / 'results.jsonl'
            with output_rows.open('x') as handle:
                for row in sorted(all_rows, key=lambda r: r['validation_index']): handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush(); os.fsync(handle.fileno())
            outputs.append(exclusive_json(directory / 'complete.json', dict(schema='opsd-signal-heldout-replay-complete-v1',
                accepted=True, bank=bp, endpoint=ep, request=request_ref, results=file_ref(output_rows), primary_rows=512,
                actual_workers_exit_and_release=True, child_exit_evidence=exits, actual_terminal=terminal_ref,
                worker_proofs=worker_proofs, full_owned_release=release, gpu_release=released, new_samples_or_training=0)))
            jobs = []
        final_release = owned_release(cfg, modules, owner, plan['analysis_run_id'], [], plan['helper_root'], root)
        return exclusive_json(root / 'complete.json', dict(accepted=True, bank=bp, plan=pp, replay_receipts=outputs,
            posthoc_responses=4608, new_samples_or_training=0, actual_workers_exit_and_release=True, final_release=final_release))
    except BaseException as error:
        if attempt_created:
            exclusive_json(root / ('failure-%d.json' % time.time_ns()), dict(accepted=False, exception_type=type(error).__name__,
                message=str(error), known_children=[i if i is not None else dict(pid=p.pid, identity='UNKNOWN') for p, i, _ in jobs],
                no_signals_sent=True, original_inputs_preserved=True))
        # A logging/admission failure never drops GPU leases around a live child.
        while any(p.poll() is None for p, _, _ in jobs): time.sleep(1)
        if attempt_created and jobs:
            exclusive_json(root / ('failed-children-terminal-%d.json' % time.time_ns()),
                dict(child_exit_evidence=[dict(identity=i, slot=s, actual_wait_returncode=p.wait()) for p, i, s in jobs]))
            while True:
                try:
                    owned_release(cfg, modules, owner, plan['analysis_run_id'], [i for _, i, _ in jobs if i is not None], plan['helper_root'], root)
                    resource_boundary(cfg, modules, owner, plan['analysis_run_id'], released=True); break
                except Exception: time.sleep(5)
        raise
    finally:
        for log in opened_logs: log.close()
        for lock in reversed(locks): fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def main():
    p = argparse.ArgumentParser()
    for name in ('plan', 'plan-sha256', 'output'): p.add_argument('--' + name, required=True)
    a = p.parse_args(); proof = run(dict(path=a.plan, sha256=a.plan_sha256), a.output)
    print(json.dumps(dict(accepted=True, complete=proof)))


if __name__ == '__main__': main()
