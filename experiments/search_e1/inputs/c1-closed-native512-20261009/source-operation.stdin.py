# Closed C1 native512; operational reuse of published fixed-handle SHA/USTAR.
import hashlib, json, os, pathlib, runpy, sys, tarfile, time
assert os.getuid() == os.geteuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert 'torch' not in sys.modules
N = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
H = N / 'source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py'
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
S = N / 'eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/C1/runs'
T = N / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs/01-C1/evaluation'
CFG = T.parent / 'evaluation-config.json'
CFG_SHA = '804e7b13849cecda959fc820e872a8809960279d5ffb77578e2e86583c1592ee'
RUN = '8d74046ea72b409d9fdf6682f5637bd8'
VERSION = 'raw10_bf16'
SCHEMA = 'c1-closed-native512-fixed-inventory-v1'
PINS = {
    'summary': (S / 'summary.json', '25ddf5ec110b21db929e94a5c4ef1d169f34c02cd0f9567e2ccc5f18215e688a'),
    'controller': (S / 'controller-complete.json', '23888ac519fdcbbe6a838d87e32656eee09504e8ff6d6a453a0a99bfee4f37b4'),
    'phase': (S / VERSION / 'phase-complete.json', 'f1906a1eaeabd12ce7e978c6c439caa0c27de6cc7bf1430513ba8c3946cb0fa2'),
    'terminal': (S / VERSION / 'actual-terminal.json', '0cab3aec6a7bca8932303c471ffb0cbff6cb54acc7f3e3eaaf17e1bf9a130b7d'),
    'accepted': (T / 'result-accepted.json', '1454ad37b284c99e96118c9766d653b2145f5b6d5f1fb689f565988069dc6b05'),
    'parent_exit': (T / 'actual-child-exit.json', '3e1cef2cded1ae7c7c49cfe98909e3014c6ae4b48226ae6a500f14207fa20afb'),
    'parent_release': (T / 'actual-child-release.json', '0d03622ad8bcd5decf24a7e36da26e6608de555bbf8b6c744203bb33e6f9d9ab'),
}
config, config_ref = b['checked_json'](CFG, CFG_SHA)
assert config['run_id'] == RUN and config['variant'] == 'C1' and config['paths']['output_root'] == str(S)
EXCLUDED = ['Active study driver/global observer/current roles', 'Training raw/checkpoints/models/inference caches; backed up separately',
            'Other group/zero evaluation scopes']

def same_ref(ref, proof):
    assert ref['path'] == proof['path'] and ref['sha256'] == proof['sha256']
    if 'bytes' in ref: assert ref['bytes'] == proof['bytes']

def closure():
    values, proofs = {}, {}
    for name, (path, sha) in PINS.items(): values[name], proofs[name] = b['checked_json'](path, sha)
    phase, terminal, controller, summary, accepted = [values[k] for k in ('phase', 'terminal', 'controller', 'summary', 'accepted')]
    assert phase['accepted'] is True and phase['release']['accepted'] is True
    assert phase['run_id'] == terminal['run_id'] == controller['run_id'] == summary['run_id'] == RUN
    assert phase['version'] == terminal['version'] == VERSION
    assert phase['actual_worker_count'] == phase['expected_worker_count'] == len(phase['child_exit_evidence']) == 8
    assert phase['actual_exit_codes_never_inferred'] is True and terminal['all_actual_wait_zero'] is True
    assert terminal['child_exit_evidence'] == phase['child_exit_evidence']
    assert all(x['actual_wait_returncode'] == 0 and x['identity']['uid'] == 1005
               and x['identity']['run_id'] == RUN and isinstance(x['identity']['starttick'], int)
               for x in phase['child_exit_evidence'])
    assert controller['accepted'] is True and controller['release']['accepted'] is True and controller['primary_slots'] == 512
    same_ref(controller['config'], config_ref); same_ref(controller['summary'], proofs['summary'])
    assert accepted['accepted'] is True
    same_ref(accepted['config'], config_ref); same_ref(accepted['controller_completion'], proofs['controller'])
    same_ref(accepted['summary'], proofs['summary'])
    assert summary['schema'] == 'opsd-ablation-native512-summary-v1' and summary['accepted'] is True
    assert summary['primary_slots'] == 512 and summary['variant'] == 'C1' and set(summary['models']) == {VERSION}
    model = summary['models'][VERSION]
    assert model['overall']['questions'] == 512 and model['overall']['strict_hits'] == 236
    assert model['datasets']['nq']['questions'] == model['datasets']['hotpotqa']['questions'] == 256
    assert model['datasets']['nq']['strict_hits'] == 109 and model['datasets']['hotpotqa']['strict_hits'] == 127
    same_ref(summary['inputs'][0], proofs['phase'])
    parent = values['parent_exit']
    assert parent['run_id'] == RUN and parent['actual_wait_returncode'] == 0 and parent['signal_termination_not_exit0'] is False
    assert parent['identity']['pid'] == 2495227 and parent['identity']['uid'] == 1005 and parent['identity']['starttick'] == 929754695
    assert values['parent_release']['accepted'] is True
    return dict(config=config_ref, source_receipts=proofs, actual_worker_wait_codes=[0]*8,
        actual_native_parent_wait_returncode=0, primary_slots=512, strict_hits=236, nq_hits=109, hotpot_hits=127,
        scope='Existing accepted C1 native512 identity/closure; no new scoring/model inference or active-study termination claim')

def enumerate_files():
    paths = [CFG]
    for tree in (S, T):
        b['checked_path'](str(tree), directory=True)
        for root, dirs, names in os.walk(tree, topdown=True, followlinks=False):
            root = pathlib.Path(root); b['checked_path'](str(root), directory=True)
            for name in dirs: b['checked_path'](str(root / name), directory=True)
            for name in names: paths.append(b['checked_path'](str(root / name)))
    paths = sorted(paths, key=lambda p: p.relative_to(N).as_posix())
    assert paths and len(paths) == len(set(paths))
    for path in paths:
        name = b['safe_name'](path.relative_to(N).as_posix()); tarfile.TarInfo(name).tobuf(format=tarfile.USTAR_FORMAT)
    return paths

mode = sys.argv[1]
closures, paths = closure(), enumerate_files()
if mode == 'inventory':
    files = [dict(relative=p.relative_to(N).as_posix(), **b['fileproof'](p)) for p in paths]
    for f in files: b['stable'](f)
    assert closures == closure() and paths == enumerate_files()
    print(json.dumps(dict(schema=SCHEMA, source=str(N), run_id=RUN, variant='C1', versions=[VERSION], config=config_ref,
        closures=closures, files=files, excluded_scope=EXCLUDED, bytes=sum(f['bytes'] for f in files),
        expected_file_count=len(files), primary_slots=512, strict_hits=236, node_originals_retained=True,
        backup_is_training_dependency=False, observed_unix=time.time()), sort_keys=True))
else:
    raw = sys.stdin.buffer.read(); x = json.loads(raw)
    assert x['schema'] == SCHEMA and x['source'] == str(N) and x['run_id'] == RUN and x['versions'] == [VERSION]
    assert x['config'] == config_ref and x['closures'] == closures
    assert x['expected_file_count'] == len(x['files']) and x['bytes'] == sum(f['bytes'] for f in x['files'])
    assert [f['path'] for f in x['files']] == [str(p) for p in paths]
    assert all(f['relative'] == pathlib.Path(f['path']).relative_to(N).as_posix() for f in x['files'])
    for f in x['files']: b['stable'](f)
    if mode == 'stream': b['stream_archive'](x, sys.stdout.buffer)
    else:
        assert mode == 'after' and paths == enumerate_files() and closures == closure()
        print(json.dumps(dict(schema='c1-closed-native512-source-metadata-after-v1', accepted=True, source=str(N),
            run_id=RUN, versions=[VERSION], config=config_ref, closures=closures, files=len(paths),
            inventory_sha256=hashlib.sha256(raw).hexdigest(), all_source_metadata_stable=True,
            all_noncache_file_set_unchanged=True, observed_unix=time.time()), sort_keys=True))
