"""Private stdout-only live online200 snapshot. Never launches, stops or resumes jobs.

This is a CPU/file audit, not a saved Adam tensor read or an evaluation. Only
/health is requested. Transient reads are uncertain, never a training verdict.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import types

sys.dont_write_bytecode = True


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def stat(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), 'absolute non-symlink file required')
    before = stat(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        require(stat_fd(stream) == before, 'opened file changed')
        raw = stream.read(); h.update(raw)
        require(stat_fd(stream) == before, 'file changed during read')
    require(stat(path) == before, 'file replaced during read')
    digest = h.hexdigest()
    require(expected is None or expected == digest, 'file SHA mismatch')
    return raw, dict(path=str(path), sha256=digest, **before)


def stat_fd(stream):
    s = os.fstat(stream.fileno())
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def value(path, expected=None):
    raw, proof = read(path, expected)
    return json.loads(raw), proof


def ref_value(ref):
    return value(ref['path'], ref['sha256'])


def hash_file(path, expected):
    """Stream large source assets, without retaining weight bytes."""
    path = Path(path); require(path.is_absolute() and not path.is_symlink(), 'asset path invalid')
    before = stat(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        require(stat_fd(stream) == before, 'asset inode changed')
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
        require(stat_fd(stream) == before, 'asset mutated during hash')
    require(stat(path) == before and digest.hexdigest() == expected, 'asset SHA/stat changed')
    return dict(path=str(path), sha256=digest.hexdigest(), **before)


def source_tree(root, manifest_ref, count=None):
    root = Path(root).resolve(); manifest, mp = ref_value(manifest_ref)
    require(Path(mp['path']).parent.resolve() == root, 'manifest root mismatch')
    entries = manifest['files']; require(count is None or len(entries) == count, 'source inventory count differs')
    sources = {}; proofs = []
    for name, entry in entries.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe source relative path')
        digest = entry['sha256'] if isinstance(entry, dict) else entry
        raw, proof = read(root / relative, digest)
        if isinstance(entry, dict) and 'bytes' in entry:
            require(entry['bytes'] == proof['bytes'], 'source byte count differs')
        proofs.append(dict(relative_path=name, **proof)); sources[name] = raw
    require({str(p.relative_to(root)) for p in root.rglob('*.py')} == {n for n in entries if n.endswith('.py')}, 'extra/missing Python source')
    return sources, dict(manifest=mp, files=proofs)


def exact_module(name, path, raw, package=False):
    require(name not in sys.modules, 'module name already loaded')
    module = types.ModuleType(name); module.__file__ = str(path)
    module.__package__ = name if package else name.rpartition('.')[0]
    if package:
        module.__path__ = [str(Path(path).parent)]
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def native_modules(config, raw):
    # Only these stdlib-only helpers are executed. Avoid any timestamp-valid pyc.
    root = Path(config['paths']['helper_root']).resolve(); prefix = '_online_snapshot_native'
    exact_module(prefix, root / '__init__.py', b'', True)
    modules = {}
    for name in ('checkpoint', 'acceptance', 'scoring', 'prepare'):
        modules[name] = exact_module(prefix + '.' + name, root / (name + '.py'), raw[name + '.py'])
    # verify_runtime invokes native scoring; pre-bind its two exact frozen files
    # so importlib finds verified source modules rather than cached bytecode.
    mf, _ = ref_value(config['runtime_manifest']); entries = mf.get('files', mf)
    runtime = Path(config['paths']['runtime']).resolve()
    for name, rel, package in [('search_r1', 'upstream/Search-R1/search_r1/__init__.py', True),
                               ('search_r1.qa_protocol', 'upstream/Search-R1/search_r1/qa_protocol.py', False)]:
        entry = entries[rel]; digest = entry['sha256'] if isinstance(entry, dict) else entry
        data, _ = read(runtime / rel, digest); exact_module(name, runtime / rel, data, package)
    return modules


def command_option(argv, name):
    require(argv.count(name) == 1, 'command option absent/duplicated: ' + name)
    index = argv.index(name); require(index + 1 < len(argv), 'option missing value')
    return argv[index + 1]


def controller_identity(config, binding, native):
    require(binding.get('accepted') is True and binding['run_id'] == config['run_id']
            and binding['config_sha256'] == config['_config_sha256'], 'controller binding config/run differs')
    expected = binding['owner']; actual = native.process(expected['pid'])
    argv = expected['argv_tokens']
    require(native.same_process(actual, expected) and actual['uid'] == config['owner']['uid']
            and actual.get('run_id') == config['run_id'] and actual['argv'] == argv,
            'controller UID/tick/run/argv changed')
    require(command_option(argv, '--config-sha256') == config['_config_sha256']
            and command_option(argv, '--run-id') == config['run_id'], 'controller CLI scientific binding differs')
    return actual


def external_identity(argv, config):
    result = []
    for manifest, publication in [(dict(path=command_option(argv, '--source-manifest'), sha256=command_option(argv, '--source-manifest-sha256')),
                                   dict(path=command_option(argv, '--publication'), sha256=command_option(argv, '--publication-sha256'))),
                                  (dict(path=str(Path(command_option(argv, '--wait-root')) / 'source-manifest.json'),
                                        sha256='baa072189d0213309d72dd9d04c67bdaf03361e7e7b452b137040d4bfee9a67f'),
                                   dict(path=command_option(argv, '--wait-publication'), sha256=command_option(argv, '--wait-publication-sha256')))]:
        _, tree = source_tree(Path(manifest['path']).parent, manifest)
        pub, proof = ref_value(publication)
        require(pub.get('accepted') is True and pub.get('branch') == 'search-e1'
                and re.fullmatch('[0-9a-f]{40}', pub.get('source_commit', ''))
                and pub.get('source_commit') == pub.get('remote_readback_sha')
                and pub.get('source_manifest_sha256') == manifest['sha256']
                and pub.get('run_id') == config['run_id'] and pub.get('config_sha256') == config['_config_sha256'], 'external publication identity differs')
        rows = pub.get('files', [])
        require(len({r['path'] for r in rows}) == len(rows), 'duplicate publication file paths')
        directory = Path(manifest['path']).parent.name
        for item in tree['files'] + [dict(relative_path='source-manifest.json', **tree['manifest'])]:
            relative = 'experiments/search_e1/control/' + directory + '/' + item['relative_path']
            matches = [r for r in rows if r['path'] == relative]
            require(len(matches) == 1 and matches[0].get('equal') is True
                    and matches[0]['sha256'] == item['sha256'] and matches[0]['bytes'] == item['bytes'], 'external publication byte readback differs')
        result.append(dict(source=tree, publication=proof, source_commit=pub['source_commit']))
    require(any(str(Path(argv[2]).resolve()) == p['path'] for p in result[0]['source']['files']), 'executed recovery script outside manifest')
    return result


def gpu_check(config, owner, children, oracle, native):
    devices, applications = native.inventory()
    require([d['index'] for d in devices] == list(range(8)) and [d['uuid'] for d in devices] == config['owner']['gpu_uuids'], 'GPU index/UUID changed')
    selected = []; owned_uuids = set(); alerts = []
    for app in applications:
        require(app['gpu_uuid'] in config['owner']['gpu_uuids'], 'foreign GPU UUID')
        if app['pid'] == oracle['pid']:
            actual = native.process(app['pid'])
            require(native.same_process(actual, oracle) and app['gpu_uuid'] == config['owner']['gpu_uuids'][7]
                    and 0 <= app['used_mib'] <= 1024, 'oracle identity/device/memory differs')
            role = 'approved_oracle18004'
        elif app['pid'] == owner['pid']:
            actual = native.process(app['pid'])
            require(native.same_process(actual, owner), 'CPU parent GPU PID was reused')
            role = 'unexpected_CPU_only_controller_GPU_context'
            alerts.append('CPU_only_controller_has_GPU_context')
        else:
            actual = native.owned_gpu_process(app['pid'], config['run_id'], owner, children)
            role = 'same_run_live_descendant'
            owned_uuids.add(app['gpu_uuid'])
        selected.append(dict(**app, role=role, uid=actual['uid'], starttick=actual['starttick'], run_id=actual.get('run_id')))
    limits = [dict(index=d['index'], uuid=d['uuid'], actual_free_mib=d['actual_free_mib'],
                   minimum_free_mib=config['resources']['training_minimum_free_mib'] if d['uuid'] in owned_uuids else config['resources']['startup_minimum_free_mib']) for d in devices]
    return dict(devices=devices, applications=selected,
                occupied_by_owned_run_uuids=sorted(owned_uuids), per_card_actual_free_limits=limits, alerts=alerts,
                resource_scope='Observed live owned worker cards use running floor; all other cards use startup floor. This is not next-stage or owned-release admission.',
                actual_free_limit_met=all(d['actual_free_mib'] >= d['minimum_free_mib'] for d in limits))


def mutable_value(path):
    return value(Path(path).absolute())


def observer(config, native):
    binding, bp = mutable_value(Path(config['paths']['receipts']) / 'memory-binding.json')
    identity = binding['observer_identity']; actual = native.process(identity['pid'])
    require(binding['experiment_run_id'] == config['run_id'] and binding['config_sha256'] == config['_config_sha256']
            and native.same_process(actual, identity) and actual.get('run_id') == identity['run_id']
            and actual['uid'] == config['owner']['uid'], 'observer identity/config changed')
    manifest, _ = ref_value(config['helper_source_manifest'])
    entry = manifest['files']['memory_observer.py']; digest = entry['sha256'] if isinstance(entry, dict) else entry
    require(identity['source_sha256'] == digest, 'observer source changed')
    live, lp = mutable_value(binding['live_report_path'])
    age = time.time() - live['updated_unix']
    require(0 <= age <= 30 and live['observer_identity'] == identity and live['run_id'] == config['run_id']
            and live['config_sha256'] == config['_config_sha256'], 'observer stale/foreign')
    require([d['uuid'] for d in live['current']['devices']] == config['owner']['gpu_uuids'], 'observer UUID changed')
    return dict(binding=bp, report=lp, age_seconds=age, samples=live['samples'],
                minimum_actual_free_mib=live['minimum_actual_free_mib'], samples_below_limit=live['samples_below_limit'], current=live['current'])


def completed(config, checkpoint):
    root = Path(config['paths']['output_root']); rows = []; updates = 0; Ktotal = 0; latest = None
    paths = sorted(root.glob('outers/outer-*/training/outer-complete.json'))
    for index, path in enumerate(paths):
        require(path.parent.parent.name == 'outer-%05d' % index, 'completed outer gap/duplicate')
        data, proof = value(path)
        outer_path = path.parent.parent / 'outer-manifest.json'; outer, op = value(outer_path)
        require(outer['run_id'] == config['run_id'] and outer['config_sha256'] == config['_config_sha256']
                and outer['outer_index'] == index and outer['cumulative_optimizer_updates_before'] == updates, 'outer science identity/continuity changed')
        require(data['schema'] == 'opsd-online-outer-complete-v1' and data['accepted'] is True
                and data['run_id'] == config['run_id'] and data['outer_index'] == index
                and type(data['K']) is int and 0 <= data['K'] <= data['P'] == outer['P'] <= 512
                and data['optimizer_updates_delta'] == math.ceil(data['K']/32)
                and data['optimizer_updates'] == updates + math.ceil(data['K']/32)
                and data['scheduler_advanced_updates'] == 0, 'completion identity/count differs')
        cp, cpp = ref_value(data['checkpoint'])
        require(cp['schema'] == 'opsd-online-checkpoint-v1' and cp['accepted'] is True
                and cp['run_id'] == config['run_id'] and cp['config_sha256'] == config['_config_sha256']
                and cp['runtime_identity'] == config['runtime_manifest']['sha256']
                and cp['source_model_identity'] == config['source_model_manifest']['sha256']
                and cp['optimizer_updates'] == data['optimizer_updates']
                and cp['cursor']['outer_index'] == index and cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'] == data['K']
                and 'outer_end' in cp['protection_reason'], 'completed CP identity/cursor differs')
        require(outer['previous_complete_checkpoint'] == (latest[0]['checkpoint'] if latest else None), 'completed outer checkpoint chain differs')
        updates = data['optimizer_updates']; Ktotal += data['K']; latest = (data, cpp)
        require(cp['cumulative_pair_exposure'] == Ktotal, 'cumulative K/CP exposure differs')
        rows.append(dict(outer_index=index, completion=proof, outer_manifest=op, P=data['P'], K=data['K'],
                         optimizer_updates=updates, optimizer_updates_delta=data['optimizer_updates_delta']))
    if latest:
        cp, evidence = checkpoint.check_checkpoint(latest[0]['checkpoint'], config, config['run_id'], updates, len(rows)-1)
        latest_evidence = evidence
    else:
        latest_evidence = None
    return dict(completed_outer_count=len(rows), total_usable_pair_exposure=Ktotal, optimizer_updates=updates,
                outers=rows, latest_completed_checkpoint=latest_evidence,
                checkpoint_scope='History: completion/outer/checkpoint manifest SHA and identity/count/chain only. Latest completed protected CP: native full file SHA; no CPU Adam tensor/moment readback')


def line_count(path):
    """Count newline-complete records, without parsing or publishing private QA."""
    before = stat(path); count = 0; last = b''
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            count += block.count(b'\n'); last = block[-1:]
    after = stat(path)
    return dict(path=str(path), before=before, after=after, complete_line_count=count,
                stable=before == after, final_line_complete=last in (b'', b'\n'))


def current(config, stage):
    root = Path(config['paths']['output_root']); index = stage.get('outer_index')
    result = dict(phase=stage.get('phase'), outer_index=index, optimizer_updates=stage.get('optimizer_updates'), sampling_parts=[])
    if type(index) is not int:
        return result
    outer = root / 'outers' / ('outer-%05d' % index)
    attempts = sorted(outer.glob('sampling-attempt-*'))
    if attempts:
        for path in sorted(attempts[-1].glob('rollouts-*.jsonl')):
            require(re.fullmatch('rollouts-[0-7]\\.jsonl', path.name), 'unexpected sampling shard')
            row = line_count(path); require(row['complete_line_count'] <= 320, 'sampling shard exceeds 64x5')
            result['sampling_parts'].append(row)
    live_path = outer / 'training' / 'live-update.json'
    if live_path.exists():
        live, proof = value(live_path)
        require(live['run_id'] == config['run_id'] and live['outer_index'] == index, 'live training pointer foreign')
        cp, _ = ref_value(live['checkpoint'])
        require(cp['run_id'] == config['run_id'] and cp['config_sha256'] == config['_config_sha256'], 'current CP identity foreign')
        metrics = []
        if cp['outer_update_receipts']:
            update, up = ref_value(cp['outer_update_receipts'][-1])
            require(update['run_id'] == config['run_id'] and update['outer_index'] == index, 'tail update foreign')
            pairs = [p for r in update['all_rank_actual_token_metrics'] for p in r['pairs']]
            require(len(pairs) == update['global_actual_pairs'] and 1 <= len(pairs) <= 32, 'tail actual pair count differs')
            require(all(type(p['valid_tokens']) is int and p['valid_tokens'] > 0 and all(math.isfinite(p[k]) for k in ('raw_loss', 'post_loss')) for p in pairs)
                    and all(math.isfinite(r['grad_norm']) for r in update['all_rank_actual_token_metrics']), 'nonfinite tail metrics')
            metrics = dict(receipt=up, actual_pairs=len(pairs), valid_tokens=sum(p['valid_tokens'] for p in pairs),
                           mean_raw_kl=sum(p['raw_loss'] for p in pairs)/len(pairs), mean_postclip_loss=sum(p['post_loss'] for p in pairs)/len(pairs))
        result['training'] = dict(live=proof, optimizer_updates=cp['optimizer_updates'], cursor=cp['cursor'], tail_native_summary=metrics,
                                  scope='Latest committed scalar summary only; no token/raw QA/logits reread')
    return result


def snapshot(args):
    started = time.time()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit empty CUDA mask required')
    config, cp = value(args.config, args.config_sha256); config['_config_sha256'] = cp['sha256']
    binding, bp = value(args.controller_binding, args.controller_binding_sha256)
    require(os.getuid() == os.geteuid() == config['owner']['uid'], 'snapshot reader UID differs')
    raw, helpers = source_tree(config['paths']['helper_root'], config['helper_source_manifest'], 23)
    modules = native_modules(config, raw); acceptance = modules['acceptance']; checkpoint = modules['checkpoint']
    modules['prepare'].validate_config(config, launch=True)  # Pure config validation, not launch/resource admission.
    checks = {}; problems = []
    def check(name, action):
        try:
            checks[name] = dict(status='verified', evidence=action())
        except Exception as error:
            # Don't expose third party exceptions which may contain QA/credentials.
            problems.append(dict(check=name, status='uncertain', error_type=type(error).__name__))
            checks[name] = dict(status='uncertain', error_type=type(error).__name__)
    # Immutable science failures reject outright; live races are uncertain.
    runtime = modules['prepare'].verify_runtime(config)
    publication = acceptance.publication_admission(config, ref_value(config['gates']['publication'])[0])
    model, _ = ref_value(config['source_model_manifest']); refs = model.get('files', model.get('source_files'))
    require(len(refs) == 12, 'source100 expected12')
    source = [hash_file(r['path'], r['sha256']) for r in refs]
    model_root = Path(config['paths']['model']).resolve()
    require(len({p['path'] for p in source}) == len({(p['device'], p['inode']) for p in source}) == 12
            and all(Path(p['path']).resolve().is_relative_to(model_root) for p in source)
            and {str(p.resolve()) for p in model_root.rglob('*') if p.is_file()} == {str(Path(p['path']).resolve()) for p in source}, 'source100 actual inventory differs')
    datasets = {name: hash_file(config['paths'][name], config[name + '_identity']['sha256']) for name in ('train', 'validation')}
    owner = controller_identity(config, binding, acceptance)
    external = external_identity(owner['argv'], config)
    output = Path(config['paths']['output_root']); stage, sp = value(output / 'status.json')
    require(stage['run_id'] == config['run_id'] and stage['config_sha256'] == cp['sha256']
            and acceptance.same_process(stage['owner'], owner), 'stage foreign controller')
    history, hp = read(output / 'child-history.jsonl')
    children = [json.loads(line)['identity'] for line in history.splitlines()]
    admitted, _ = ref_value(config['gates']['retriever_admission'])
    def retriever():
        result = acceptance.retriever_admission(config, admitted)
        result['owner'] = {k: result['owner'][k] for k in ('pid', 'uid', 'starttick', 'run_id')}
        return result
    check('completed', lambda: completed(config, checkpoint))
    check('current', lambda: current(config, stage))
    for part in checks.get('current', {}).get('evidence', {}).get('sampling_parts', []):
        if not part['stable'] or not part['final_line_complete']:
            problems.append(dict(check='sampling_part_read_race_or_partial_line', status='uncertain', path=part['path']))
    if stat(output / 'status.json') != {k: sp[k] for k in stat(output / 'status.json')} or stat(output / 'child-history.jsonl') != {k: hp[k] for k in stat(output / 'child-history.jsonl')}:
        problems.append(dict(check='stage_history_race', status='uncertain'))
    # Live resource/observer measurements follow potentially lengthy CPU hashes.
    check('retriever', retriever)
    check('gpu', lambda: gpu_check(config, owner, children, admitted, acceptance))
    check('observer', lambda: observer(config, acceptance))
    def controller_end():
        actual = controller_identity(config, binding, acceptance)
        return {k: actual[k] for k in ('pid','uid','starttick','run_id')}
    check('controller_end', controller_end)
    for proof in helpers['files']:
        hash_file(proof['path'], proof['sha256'])
    attention = []
    gpu = checks.get('gpu', {}).get('evidence', {})
    if gpu and not gpu['actual_free_limit_met']:
        attention.append('actual_free_below_owned1024_or_unused20480MiB')
    attention.extend(gpu.get('alerts', []))
    memory = checks.get('observer', {}).get('evidence', {})
    if memory and memory['samples_below_limit']:
        attention.append('historical_observer_samples_below_limit')
    return dict(schema='opsd-online-readonly-health-snapshot-v1', started_unix=started, checked_unix=time.time(), status='uncertain' if problems else ('attention' if attention else 'verified'),
                config=cp, controller_binding=bp, run_id=config['run_id'], controller={k: owner[k] for k in ('pid','uid','starttick','run_id')},
                helpers=helpers, runtime=runtime, publication=publication, external_sources=external,
                source100_fresh_full_SHA=source, datasets_fresh_SHA=datasets, stage=dict(phase=stage['phase'], outer_index=stage.get('outer_index'), proof=sp),
                checks=checks, uncertainties=problems, attention=attention, no_training_verdict=True,
                http_scope='Native retriever /health GET only; no /retrieve or inference request',
                no_gpu_torch_model_rng_retrieval_signals_git_or_node_writes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'config-sha256', 'controller-binding', 'controller-binding-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        result = snapshot(args)
    except Exception as error:
        result = dict(schema='opsd-online-readonly-health-snapshot-v1', status='uncertain', error_type=type(error).__name__,
                      no_training_verdict=True, no_signals_or_node_writes=True)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
