# Fixed closed C1 global evidence; published fixed-handle SHA/USTAR primitives.
import hashlib, json, os, pathlib, runpy, sys, tarfile, time
assert os.getuid() == os.geteuid() == 1005
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
N = pathlib.Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
H = N / 'source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py'
assert hashlib.sha256(H.read_bytes()).hexdigest() == 'd2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57'
b = runpy.run_path(str(H))
G = N / 'group-C1'
R = G / 'runs'
T = N / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs/01-C1/training'
CFG = G / 'receipts/authorized-launch-20261009/config.json'
CFG_SHA = '13fdbd51cc290bc6d23b0577c67bf8160a17292fab6f315fbfb07ea3f6ed3310'
RUN = 'a1a56efd32e5456d8c6813454975e38f'
REPAIR_SHA = 'b26ef0a9c11d4acc2b08453c8248b2c532eeb8bde0956245354f42a78e4e381d'
REPAIR = N / 'receipts/dtype-recovery-538ac99-20261009/repair-manifest.json'
STATUS_SHA = '258d746de6994423db4318414017212b62a1b8c82ffb89e71eabcf9524394df3'
GROUP_TERMINAL_SHA = '26d3e734986c1a5db622a46878252eea837c331d0472573369364f73201c1d61'
FINAL_CP_SHA = 'c633589a4585db624c4791a59523a1fd0bf06284f016a6a3beab7ba24a841bb8'
SCHEMA = 'c1-closed-global-terminal-fixed-inventory-v2'
ROOTS = (R / 'phase-exits', T, G / 'receipts/dtype-recovery' / REPAIR_SHA)
SINGLE = (R / 'child-history.jsonl', R / 'status.json', R / 'run-identity.json', REPAIR)
EXCLUDED = ['All outer raw/model/cache/checkpoint payloads; C1 4087 closed raw files already separately accepted',
            'Active study driver/global status/logs, scoped CPU roles and memory observer',
            'C1 heldout evaluation outputs; not closed by this training-terminal scope']
config, config_ref = b['checked_json'](CFG, CFG_SHA)
assert config['run_id'] == RUN and config['variant'] == 'C1'

def read_ref(ref):
    value, proof = b['checked_json'](pathlib.Path(ref['path']), ref['sha256'])
    if 'bytes' in ref: assert proof['bytes'] == ref['bytes']
    return value, proof

def read_json(path):
    raw, proof = b['fileproof'](path, retain=True)
    return json.loads(raw), proof

def exit_binding(parent_ref, exit_proof, phase, count):
    assert exit_proof['accepted'] is True
    target = pathlib.Path(exit_proof['actual_terminal']['path'])
    assert target.parent == R / 'phase-exits'
    actual, target_ref = read_ref(exit_proof['actual_terminal'])
    assert actual['schema'] == 'opsd-ablation-actual-phase-terminal-v1'
    assert actual['run_id'] == RUN and actual['variant'] == 'C1' and actual['config_sha256'] == CFG_SHA
    assert actual['phase'] == phase and actual['child_exit_evidence'] == exit_proof['child_exit_evidence']
    evidence = actual['child_exit_evidence']
    assert len(evidence) == count
    assert all(x['actual_wait_returncode'] == 0 and x['identity']['uid'] == 1005
               and x['identity']['run_id'] == RUN and isinstance(x['identity']['starttick'], int)
               and not x['identity'].get('identity_uncertain') for x in evidence)
    assert exit_proof['gpu_release']['all_owned_release']['accepted'] is True
    return dict(parent=parent_ref, actual_terminal=target_ref, phase=phase, children=count, actual_returncodes=[0]*count)

def closure():
    status, status_ref = b['checked_json'](R / 'status.json', STATUS_SHA)
    assert status['run_id'] == RUN and status['variant'] == 'C1' and status['phase'] == 'TRAIN_COMPLETE'
    assert status['outer_iterations'] == 10 and status['actual_optimizer_updates'] == 101
    assert status['latest_checkpoint']['sha256'] == FINAL_CP_SHA
    terminal, terminal_ref = b['checked_json'](T / 'group-terminal.json', GROUP_TERMINAL_SHA)
    assert terminal['schema'] == 'opsd-ablation-group-foreground-terminal-v1' and terminal['accepted'] is True
    assert terminal['run_id'] == RUN and terminal['config_sha256'] == CFG_SHA
    assert terminal['actual_exit_code'] == 0 and terminal['signal_termination_not_exit0'] is False
    assert terminal['controller']['pid'] == 2276380 and terminal['controller']['uid'] == 1005
    assert terminal['controller']['starttick'] == 929210641 and terminal['status']['sha256'] == STATUS_SHA
    assert terminal['status']['path'] == str(R / 'status.json')
    assert terminal['actual_child_exit']['path'] == str(T / 'actual-child-exit.json')
    assert terminal['actual_release']['path'] == str(T / 'actual-child-release.json')
    _, wait_ref = read_ref(terminal['actual_child_exit'])
    _, release_ref = read_ref(terminal['actual_release'])
    assert terminal['dtype_repair']['path'] == str(REPAIR) and terminal['dtype_repair']['sha256'] == REPAIR_SHA
    _, repair_ref = read_ref(terminal['dtype_repair'])
    identity, identity_ref = read_json(R / 'run-identity.json')
    assert identity['run_id'] == RUN and identity['variant'] == 'C1' and identity['config_sha256'] == CFG_SHA
    parents, outers = [], []
    for index in range(10):
        O = R / 'outers' / ('outer-%05d' % index)
        complete, complete_ref = read_json(O / 'training/outer-complete.json')
        assert complete['accepted'] is True and complete['run_id'] == RUN and complete['outer_index'] == index
        cp, cp_ref = read_ref(complete['checkpoint'])
        assert pathlib.Path(cp_ref['path']).parent.parent == R / 'checkpoints'
        assert cp['accepted'] is True and cp['run_id'] == RUN and cp['config_sha256'] == CFG_SHA
        assert cp['cursor']['outer_index'] == index and cp['optimizer_updates'] == complete['optimizer_updates']
        outers.append(dict(outer_index=index, optimizer_updates=complete['optimizer_updates'], complete=complete_ref, checkpoint=cp_ref))
        sampling, ref = read_json(O / 'sampling-complete.json')
        assert sampling['accepted'] is True and sampling['run_id'] == RUN
        parents.append(dict(outer_index=index, role='sampling', **exit_binding(ref, sampling['exit_proof'], 'sampling-running', 8)))
        teacher, ref = read_json(O / 'teacher-diagnostics/complete.json')
        assert teacher['accepted'] is True and teacher['run_id'] == RUN and teacher['config_sha256'] == CFG_SHA
        assert teacher['outer_index'] == index
        if teacher['not_applicable']:
            assert teacher['count'] == 0 and teacher['child'] is None and teacher['exit_proof'] is None
            parents.append(dict(outer_index=index, role='teacher', parent=ref, not_applicable=True,
                                reason=teacher['not_applicable_reason'], actual_terminal=None))
        else:
            parents.append(dict(outer_index=index, role='teacher', **exit_binding(ref, teacher['exit_proof'], 'sampling-running', 1)))
        exits = sorted((O / 'trainer-logs').glob('attempt-*-exit.json'))
        assert exits
        for path in exits:
            value, ref = read_json(path)
            parents.append(dict(outer_index=index, role='training', **exit_binding(ref, value, 'training-running', 1)))
        merge = O / 'cpu-merge-parent.json'
        if merge.exists():
            value, ref = read_json(merge)
            assert value['accepted'] is True and value['run_id'] == RUN and value['config_sha256'] == CFG_SHA
            assert value['actual_terminal'] == value['exit_proof']['actual_terminal']
            parents.append(dict(outer_index=index, role='cpu-merge', **exit_binding(ref, value['exit_proof'], 'cpu-merge', 1)))
        else:
            assert index == 0 and config['precision']['mode'] == 'bf16'
            parents.append(dict(outer_index=index, role='cpu-merge', not_applicable=True,
                                reason='Fresh BF16 source100 sampling has no CPU merge child', actual_terminal=None))
    assert outers[-1]['checkpoint']['sha256'] == FINAL_CP_SHA and outers[-1]['optimizer_updates'] == 101
    return dict(status=status_ref, run_identity=identity_ref, group_terminal=terminal_ref,
                actual_child_exit=wait_ref, actual_release=release_ref, dtype_repair=repair_ref,
                closed_outers=outers, parent_terminal_bindings=parents,
                scope='C1 training only; preserved study-driver identity in group terminal is historical, not a current driver acceptance')

def enumerate_files():
    paths = list(SINGLE)
    for tree in ROOTS:
        b['checked_path'](str(tree), directory=True)
        for root, dirs, names in os.walk(tree, topdown=True, followlinks=False):
            root = pathlib.Path(root); b['checked_path'](str(root), directory=True)
            for name in dirs: b['checked_path'](str(root / name), directory=True)
            for name in names: paths.append(b['checked_path'](str(root / name)))
    paths = sorted(paths, key=lambda p: p.relative_to(N).as_posix())
    assert paths and len(paths) == len(set(paths))
    for path in paths:
        b['checked_path'](str(path))
        name = b['safe_name'](path.relative_to(N).as_posix())
        tarfile.TarInfo(name).tobuf(format=tarfile.USTAR_FORMAT)
    return paths

mode = sys.argv[1]
closures, paths = closure(), enumerate_files()
if mode == 'inventory':
    files = [dict(relative=p.relative_to(N).as_posix(), **b['fileproof'](p)) for p in paths]
    for f in files: b['stable'](f)
    assert closures == closure() and paths == enumerate_files()
    targets = {x['actual_terminal']['path'] for x in closures['parent_terminal_bindings'] if x.get('actual_terminal')}
    assert targets <= {str(p) for p in paths}
    print(json.dumps(dict(schema=SCHEMA, source=str(N), run_id=RUN, variant='C1', closed_outers=list(range(10)),
        config=config_ref, closures=closures, files=files, excluded_scope=EXCLUDED,
        bytes=sum(f['bytes'] for f in files), expected_file_count=len(files),
        referenced_phase_exit_paths=sorted(targets), additional_phase_exits_are_historical_not_assumed_success=True,
        node_originals_retained=True, backup_is_training_dependency=False, observed_unix=time.time()), sort_keys=True))
else:
    raw = sys.stdin.buffer.read(); x = json.loads(raw)
    assert x['schema'] == SCHEMA and x['source'] == str(N) and x['run_id'] == RUN
    assert x['closed_outers'] == list(range(10)) and x['config'] == config_ref and x['closures'] == closures
    assert x['expected_file_count'] == len(x['files']) and x['bytes'] == sum(f['bytes'] for f in x['files'])
    assert [f['path'] for f in x['files']] == [str(p) for p in paths]
    assert all(f['relative'] == pathlib.Path(f['path']).relative_to(N).as_posix() for f in x['files'])
    for f in x['files']: b['stable'](f)
    if mode == 'stream': b['stream_archive'](x, sys.stdout.buffer)
    else:
        assert mode == 'after' and paths == enumerate_files() and closures == closure()
        print(json.dumps(dict(schema='c1-closed-global-terminal-source-metadata-after-v2', accepted=True,
            source=str(N), run_id=RUN, closed_outers=list(range(10)), config=config_ref, closures=closures,
            files=len(paths), inventory_sha256=hashlib.sha256(raw).hexdigest(), all_source_metadata_stable=True,
            all_noncache_file_set_unchanged=True, observed_unix=time.time()), sort_keys=True))
