"""CPU source/asset integrity layer. Never replays the original late formal gate."""
import argparse
import datetime
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
import types

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
TOOLS90 = ROOT / 'tools/final-acceptance-90c2695a7f46'
FORMAL = ROOT / 'tools/formal-bb5f4872142f'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
TOOLS90_SHA = '90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
COMMON_SHA = '9360f3e80b8525a7291c649fe60f38a4e283b087db4baa482276447915055b5e'
OWNED_SHA = 'f69418c892f3d0a644e06003773bbfdfeafe9355a48dcc013f94c54af65ab1fd'
PREFIXES = ('search_r1', 'verl', 'reproduction')
DRIVER_PID = 1533312
DRIVER_START = '1829085316'


def need(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def physical_fd(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'Absolute physical input required')
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def physical_namespace(path):
    """Bind the two source-tree namespace directories without executing code."""
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'Physical namespace required')
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        before = os.fstat(directory)
        need(before.st_uid == os.getuid(), 'Namespace directory UID differs')
        try:
            os.stat('__init__.py', dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError('Unpinned namespace initializer refused')
        need(identity(before) == identity(os.fstat(directory)), 'Namespace changed during observation')
        return {'physical_path': str(path), 'device': before.st_dev, 'inode': before.st_ino,
                'uid': before.st_uid, 'no_initializer_observed': True}
    finally:
        os.close(directory)


def identity(value):
    return tuple(getattr(value, k) for k in ('st_dev', 'st_ino', 'st_uid', 'st_mode',
                                            'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def verified_bytes(path, pin, limit=32 * 1024 * 1024):
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'Exact SHA required')
    fd = physical_fd(path)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
             and 0 <= before.st_size <= limit, 'Own bounded regular input required')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read()
        after = os.fstat(fd)
        check = physical_fd(path)
        try:
            path_after = os.fstat(check)
        finally:
            os.close(check)
        need(identity(before) == identity(after) == identity(path_after)
             and len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == pin,
             'Executable/metadata bytes or identity changed before use')
        return raw
    finally:
        os.close(fd)


def module_bytes(name, path, pin, package=None):
    raw = verified_bytes(path, pin)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def shared_context():
    helper = module_bytes('post_terminal_exact_shared90', TOOLS90 / 'author200_observation.py', COMMON_SHA)
    manifest, record = helper.json_file(TOOLS90 / 'manifest.json', TOOLS90, TOOLS90_SHA)
    need(manifest['tools']['author200_observation.py'] == COMMON_SHA and manifest['source_commit'] == SOURCE,
         'Original new90 tool/source pin differs')
    return helper, record


def context_args(shared):
    return argparse.Namespace(run_root=str(RUN), job_id='3261', source_root=str(REPO), source_commit=SOURCE,
        accepted5_manifest=str(RUN / 'full_state/global_step_5/manifest.json'),
        accepted5_manifest_sha256=shared.MANIFEST5_SHA,
        accepted5_receipt=str(RUN / 'acceptance-5-receipt.json'), accepted5_receipt_sha256=shared.RECEIPT5_SHA,
        public_key=str(RUN / 'bindings/public.json'), public_key_sha256=shared.PUBLIC_SHA,
        native_import_closure=str(RUN / 'bindings/import_closure.json'),
        native_import_closure_sha256=shared.CLOSURE_SHA)


def origin_driver(proc=Path('/proc')):
    path = Path(proc) / str(DRIVER_PID) / 'stat'
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {'pid': DRIVER_PID, 'expected_start_ticks': DRIVER_START, 'status': 'ORIGINAL_ABSENT_AT_OBSERVATION'}
    fields = raw.rsplit(')', 1)[1].split()
    need(len(fields) >= 20, 'Original PID stat is incomplete')
    result = {'pid': DRIVER_PID, 'expected_start_ticks': DRIVER_START,
              'actual_start_ticks': fields[19], 'state': fields[0]}
    if fields[19] != DRIVER_START:
        result['status'] = 'PID_REUSED_ORIGINAL_GONE'
    elif fields[0] in ('Z', 'X'):
        result['status'] = 'ORIGINAL_EXITED_ZOMBIE'
    else:
        raise ValueError('Original training driver remains alive; post-training integrity entry refused')
    return result


def execution_identity(proc=Path('/proc'), environ=None):
    env = os.environ if environ is None else environ
    status = (Path(proc) / 'self/status').read_text()
    ids = re.search(r'^Uid:\s+(.*)', status, re.M)
    need(ids is not None, 'Current CPU UID fields missing')
    uids = [int(v) for v in ids.group(1).split()]
    need(len(uids) == 4 and set(uids) == {os.getuid()}, 'Actual CPU UID fields differ')
    fields = (Path(proc) / 'self/stat').read_text().rsplit(')', 1)[1].split()
    need(len(fields) >= 20, 'Current CPU stat incomplete')
    cgroup = (Path(proc) / 'self/cgroup').read_text()
    job, step = env.get('SLURM_JOB_ID'), env.get('SLURM_STEP_ID')
    if job is not None:
        need(type(job) is str and re.fullmatch('[0-9]+', job)
             and re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', cgroup, re.M),
             'Reported new CPU job differs from actual cgroup')
    if step is not None:
        need(job is not None and type(step) is str and re.fullmatch(r'(?:[0-9]+|batch|extern)', step)
             and re.search(r'(?:^|/)job_' + re.escape(job) + r'/step_' + re.escape(step) + r'(?:/|$)', cgroup, re.M),
             'Reported new CPU step differs from actual cgroup')
    return {'pid': os.getpid(), 'start_ticks': fields[19], 'uid': os.getuid(), 'uids': uids,
        'node': socket.gethostname(), 'slurm_job_id': job,
        'slurm_step_id': step, 'cgroup': cgroup,
        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'), 'python': sys.executable,
        'python_version': sys.version, 'scope': 'ACTUAL_NEW_CPU_OBSERVER_NOT_SAVED_TRAINING3261'}


class ExactSourceFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile exact source records; never consult a project pyc or unbound fallback."""
    def __init__(self, records, prefixes=PREFIXES, namespaces=None):
        self.records = records
        self.prefixes = prefixes
        self.executed = {}
        self.namespaces = {} if namespaces is None else dict(namespaces)
        self.namespace_observations = {}
        need(not (set(self.namespaces) & set(records)), 'Namespace cannot alias source record')

    def targeted(self, name):
        return any(name == p or name.startswith(p + '.') for p in self.prefixes)

    def find_spec(self, fullname, path=None, target=None):
        if not self.targeted(fullname):
            return None
        if fullname in self.namespaces:
            path = Path(self.namespaces[fullname])
            self.namespace_observations[fullname] = physical_namespace(path)
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [str(path)]
            return spec
        need(fullname in self.records, 'Unpinned project import refused: ' + fullname)
        path = Path(self.records[fullname]['path'])
        need(path.suffix == '.py', 'Project source must be Python source')
        return importlib.util.spec_from_loader(fullname, self, origin=str(path),
                                              is_package=path.name == '__init__.py')

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        record = self.records[module.__name__]
        path = Path(record['path'])
        raw = verified_bytes(path, record['sha256'])
        module.__file__ = str(path)
        module.__cached__ = None
        if path.name == '__init__.py':
            module.__path__ = [str(path.parent)]
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
        self.executed[module.__name__] = {'physical_path': str(path), 'sha256': record['sha256'],
                                       'bytes': len(raw), 'compiled_same_verified_bytes': True}

    def install(self):
        need(not any(self.targeted(n) for n in sys.modules), 'Preloaded project module refused in fresh CPU entry')
        sys.meta_path.insert(0, self)

    def remove(self):
        if self in sys.meta_path:
            sys.meta_path.remove(self)


class AssetProofs:
    def __init__(self, shared):
        self.shared = shared
        self.files = {}
        self.roles = {}
        self.elapsed_seconds = 0.0

    def add(self, role, path, root, pin, expected_bytes=None):
        need(role not in self.roles and type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'Unique role/exact content pin required')
        need(expected_bytes is None or type(expected_bytes) is int and expected_bytes >= 0,
             'Typed nonnegative asset byte count required')
        path = str(path)
        if path in self.files:
            record = self.files[path]
            need(record['sha256'] == pin, 'Same physical path cannot have conflicting asset pins')
            self.shared.recheck_member(record, root)
        else:
            started = time.perf_counter()
            record = self.shared.stable_file(path, root, pin)[0]
            self.elapsed_seconds += time.perf_counter() - started
            self.files[path] = record
        need(expected_bytes is None or record['bytes'] == expected_bytes, 'Asset byte count differs')
        self.roles[role] = path
        return record

    def recheck(self):
        for record in self.files.values():
            self.shared.recheck_member(record, None)

    def result(self):
        return {'roles': dict(self.roles), 'files': {p: dict(r) for p, r in self.files.items()},
            'unique_files': len(self.files), 'single_stable_fd_full_hash_bytes': sum(r['bytes'] for r in self.files.values()),
            'stable_content_hash_elapsed_seconds': self.elapsed_seconds}


def logical_path(root, logical):
    need(type(logical) is str and logical and '\\' not in logical, 'Logical asset path required')
    need(not any(x in ('', '.', '..') for x in logical.split('/')), 'Invalid logical asset component')
    value = Path(logical)
    need(not value.is_absolute() and not any(x in ('.', '..') for x in value.parts), 'Asset path escapes root')
    result = Path(root) / value
    need(result.is_absolute(), 'Physical asset root required')
    return result


def asset_inventory(proofs, config, model, data, tool):
    model_root = Path(config['actor_rollout_ref']['model']['path'])
    need(model_root == Path(config['baselines']['distillation']['teacher_path']), 'Official actor/teacher roots differ')
    need(model['repo'] == 'Qwen/Qwen2.5-3B-Instruct'
         and model['revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
         and len(model['files']) == 11, 'Official identity/inventory differs')
    proofs.add('official_manifest_at_model_root', model_root / 'VERIFIED-MODEL-MANIFEST.json', model_root,
               '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44')
    for rec in model['files']:
        need(rec['revision'] == model['revision'], 'Official file revision differs')
        proofs.add('official/' + rec['file'], logical_path(model_root, rec['file']), model_root, rec['sha256'], rec['bytes'])
    data_root = Path(config['baselines']['holdout']['data_root'])
    need(set(data['splits']) == {'train', 'validation', 'stage', 'final'}, 'Four shared splits required')
    for role, rec in data['splits'].items():
        proofs.add('data/' + role, logical_path(data_root, rec['path']), data_root, rec['sha256'])
    raw = data['preparation']['raw_sources']
    need(len(raw) == 9, 'Exactly nine raw provenance files required')
    for rec in raw:
        proofs.add('raw/' + rec['source'] + '/' + rec['split'], logical_path(data_root, rec['path']), data_root, rec['sha256'])
    exclusion = data['exclusion_audit']
    proofs.add('data/exclusion_audit', logical_path(data_root, exclusion['path']), data_root, exclusion['sha256'])
    need(tool['status'] == 'FULL_TOOL_ASSET_CONTENT_VERIFIED'
         and tool['frozen_source_commit'] == '04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
         and tool['hostname'] == 'h100-10'
         and tool['assets_root'] == '/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets', 'Tool asset preparation binding differs')
    tool_root = Path(tool['assets_root'])
    results = tool['results']
    need(len(results['e5']) == 8, 'Exactly eight E5 files required')
    for i, rec in enumerate(results['e5']):
        proofs.add('tool/e5/' + str(i), rec['path'], tool_root, rec['sha256'], rec['bytes'])
    for role in ('wiki_gzip', 'index', 'corpus'):
        rec = results[role]
        proofs.add('tool/' + role, rec['path'], tool_root, rec['sha256'], rec['bytes'])


def timed(name, function, calls):
    start = time.perf_counter()
    result = function()
    calls[name] = {'actual_elapsed_seconds': time.perf_counter() - start, 'completed': True}
    return result


def validate_effective_paths(config, data_result, path_list):
    """Join the validated four physical artifacts to the actual saved inputs."""
    actual = {'train': path_list(config.data.train_files),
              'validation': path_list(config.data.val_files),
              'final': path_list(config.data.final_val_files),
              'stage': path_list(config.baselines.opsd_e1.stage_files)}
    expected = data_result['effective_paths']
    need(set(expected) == set(actual) and all(actual[k] == [expected[k]] for k in actual),
         'Saved resolved data inputs differ from validated shared artifacts')
    return actual


def validate_cpu_runtime(torch):
    need(str(torch.__version__).split('+')[0] == '2.4.0', 'Original native Torch2.4.0 environment required')
    need(torch.cuda.is_initialized() is False, 'CPU integrity process initialized CUDA')


def inspect(output):
    need(sys.flags.optimize == 0, 'Optimized Python forbidden for frozen validators')
    need(os.getuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10', 'Only own actual h100-10 UID0 CPU observer allowed')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only environment required')
    sys.dont_write_bytecode = True
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    output = Path(output)
    output.relative_to(RUN / 'evidence')
    need(not output.exists() and not output.is_symlink(), 'Exclusive actual output required')
    execution_before = execution_identity()
    origin_before = origin_driver()
    shared, manifest_record = shared_context()
    args = context_args(shared)
    calls = {}
    before = timed('load_accepted_context_before', lambda: shared.load_accepted_context(args), calls)
    closure, closure_record = shared.json_file(before['closure_path'], RUN, shared.CLOSURE_SHA)
    need(len(closure['modules']) == 210, 'Original native210 closure cardinality differs')
    finder = ExactSourceFinder(closure['modules'], namespaces={
        'reproduction': REPO / 'reproduction',
        'reproduction.baselines': REPO / 'reproduction/baselines'})
    finder.install()
    try:
        sys.path.insert(0, str(REPO))
        torch = importlib.import_module('torch')
        validate_cpu_runtime(torch)
        bootstrap = importlib.import_module('reproduction.baselines.e1_import_bootstrap')
        actual, code_count = timed('actual_cpu_import_and_live_code_verification',
            lambda: bootstrap.verify_imports(REPO, closure), calls)
        gate = importlib.import_module('search_r1.baselines.checkpoint_gate')
        timed('selected_actual_cpu_import_objects', lambda: gate.verify_import_closure(
            {'path': before['closure_path'], 'sha256': shared.CLOSURE_SHA}, RUN, REPO), calls)
        profile = importlib.import_module('search_r1.baselines.e1_profile')
        holdout = importlib.import_module('search_r1.baselines.sharedholdout')
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(before['config'])
        resolved = timed('saved_resolved_profile_semantics', lambda: profile.validate_profile(cfg), calls)
        need(resolved == before['config'], 'Independent saved config projection changed')
        anchor_records = {}
        def anchor(role):
            rec = before['anchor_files'][role]
            value, observed = shared.json_file(rec['physical_path'], RUN, rec['sha256'])
            anchor_records[role] = observed
            return value
        model, data, tool = anchor('model_manifest'), anchor('data_manifest'), anchor('tool_manifest')
        proofs = AssetProofs(shared)
        asset_inventory(proofs, before['config'], model, data, tool)
        model_root = Path(resolved['actor_rollout_ref']['model']['path'])
        timed('frozen_official_model_validator', lambda: profile.verify_official_assets(
            model_root, model_root / 'VERIFIED-MODEL-MANIFEST.json'), calls)
        data_result = timed('frozen_shared_identity_validator', lambda: holdout._validate_prebuilt(cfg), calls)
        need(data_result['train_rows'] == 169028 and data_result['holdout_rows'] == 512,
             'Shared actual train/validation sizes differ')
        effective_data = validate_effective_paths(cfg, data_result, holdout.paths)
        owned = module_bytes('post_terminal_exact_owned_runtime', FORMAL / 'owned_runtime.py', OWNED_SHA)
        timed('frozen_tool_content_validator', lambda: owned.verify_tool_assets(tool), calls)
        proofs.recheck()
        for name, record in finder.namespace_observations.items():
            need(physical_namespace(finder.namespaces[name]) == record,
                 'Namespace physical identity changed across imports')
        after = timed('load_accepted_context_after', lambda: shared.load_accepted_context(args), calls)
        need(shared.canonical(before['config']) == shared.canonical(after['config'])
             and before['anchor_files'] == after['anchor_files'] and before['workers'] == after['workers']
             and before['source_files'] == after['source_files'], 'Context/source/config anchors changed across content checks')
        proofs.recheck()
        for record in anchor_records.values():
            shared.recheck_member(record, RUN)
        validate_cpu_runtime(torch)
        execution_after = execution_identity()
        need(all(execution_after[k] == execution_before[k] for k in
                 ('pid', 'start_ticks', 'uid', 'uids', 'node', 'slurm_job_id', 'slurm_step_id', 'cgroup')),
             'Actual new CPU execution identity changed')
        origin_after = origin_driver()
        result = {'version': 1, 'kind': 'INDEPENDENT_POST_TRAINING_CPU_SOURCE_ASSET_INTEGRITY',
            'scope': 'NEW_CPU_SOURCE_CONFIG_IMPORT_MODEL_DATA_TOOL_CONTENT_LAYER_NOT_ORIGINAL_LATE_GATE_OR_TERMINAL_ACCEPTANCE',
            'passed': True, 'source_commit': SOURCE, 'training_job_id': '3261',
            'config_sha256': shared.CONFIG_SHA, 'inspected_utc': utc(),
            'inspection_execution_before': execution_before, 'inspection_execution_after': execution_after,
            'original_driver_before': origin_before, 'original_driver_after': origin_after,
            'new90_manifest': dict(manifest_record), 'native_closure': dict(closure_record),
            'accepted5': before['accepted5'], 'source_files': before['source_files'],
            'anchor_files': before['anchor_files'], 'context_before_after_matched': True,
            'asset_manifest_stable_records': {role: dict(record) for role, record in anchor_records.items()},
            'actual_cpu_imports': {'module_count': len(actual['modules']), 'function_code_count': code_count,
                'actual_loaded_modules': actual, 'exact_project_source_loader': dict(finder.executed),
                'code_free_namespace_directories': dict(finder.namespace_observations),
                'cuda_is_initialized': False, 'scope': 'NEW_CPU_PROCESS_ACTUAL_IMPORTS_NOT_EXITED_TRAINING_MEMORY'},
            'asset_stable_full_content_proofs': proofs.result(), 'actual_frozen_validator_calls': calls,
            'shared_data_identity': data_result,
            'saved_effective_data_inputs': effective_data,
            'legacy_additional_reads': {'official_11_files': 'one additional full hash pass',
                'tool_11_files': 'one additional full hash pass',
                'shared_4_parquets': 'two additional full hash passes plus actual pandas parquet reads',
                'raw9_and_exclusion': 'one additional full hash pass',
                'note': 'Frozen validators remain byte-exact. Stable FD content pass is recorded once per unique physical path; legacy timing is recorded separately.'},
            'original_late_formal_gate_replayed': False, 'source_runtime_modified': False,
            'limitations': ['No scheduler/child/writer-closure or own-resource release claim; original FAILED remains unchanged.',
                'Initial/fixed-teacher asset checks do not load HF200 or inspect Adam/RNG/DTensors; those remain separate.',
                'No final-evaluation parser, rescoring, GPU operation, HTTP request, lease, recovery, launch or signature executed.',
                'Actual CPU project modules are compiled from same pinned raw source without pyc. Native/third-party runtime modules retain ordinary library loading and their recorded bindings.',
                'Stored source file integrity and new CPU live code do not retroactively inspect every historical training in-memory state.',
                'Raw9 row counts inherit pinned manifest declarations; four shared parquet row/content/question identities are recomputed.',
                'Filesystem stability is observed at reads/rechecks; privileged changes restored between observations cannot be ruled out.']}
        return shared.exclusive_json(output, result)
    finally:
        finder.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.output), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
