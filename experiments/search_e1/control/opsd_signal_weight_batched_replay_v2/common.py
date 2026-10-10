"""Immutable, source-only inputs for a separate posthoc analysis namespace."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

STUDY_ROOT = '/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
VARIANTS = ('G_POS', 'G_NEG', 'H_POS', 'H_NEG')
ENDPOINTS = {('source100', 0)} | {(v, t) for v in VARIANTS for t in (15, 25)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def semantic_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stat(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def file_ref(path):
    path = Path(path).resolve(); before = stat(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    require(before == stat(path), 'File changed while hashing: ' + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=before[2])


def check_file(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str)
        and isinstance(ref.get('sha256'), str) and len(ref['sha256']) == 64,
        'Explicit immutable path/SHA required')
    actual = file_ref(ref['path'])
    require(actual['sha256'] == ref['sha256'] and actual['bytes'] == ref.get('bytes', actual['bytes']),
        'File identity differs: ' + ref['path'])
    return actual


def read_json(ref):
    actual = check_file(ref); before = stat(actual['path'])
    value = json.loads(Path(actual['path']).read_bytes())
    require(before == stat(actual['path']), 'JSON changed while reading')
    return value, actual


def read_rows(ref):
    actual = check_file(ref); before = stat(actual['path'])
    raw = Path(actual['path']).read_bytes()
    require(not raw or raw.endswith(b'\n'), 'Truncated JSONL')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(before == stat(actual['path']), 'Rows changed while reading')
    return rows, actual


def exclusive_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return file_ref(path)


def remote_output(path):
    actual = Path(path).resolve(); root = Path(STUDY_ROOT).resolve()
    require(os.getuid() == 1001 and actual.is_relative_to(root / 'analysis')
        and str(actual).startswith(STUDY_ROOT + '/analysis/'),
        'Large/private diagnostic outputs must be on authorized HKUST4 storage, never Mac')
    return actual


def bind_training(config):
    """Import the immutable training modules under an isolated package name.

    Their source is reused read-only. Nothing is installed into the ordinary
    training namespace; no Torch/model import is allowed in CPU callers.
    """
    manifest, proof = read_json(config['helper_source_manifest'])
    root = Path(config['paths']['helper_root']).resolve()
    require(config['paths']['study_base'] == STUDY_ROOT, 'Foreign study root')
    names = set(manifest['files'])
    actual = {str(p.relative_to(root)) for p in root.rglob('*.py')}
    require(actual == {n for n in names if n.endswith('.py')}, 'Frozen helper Python inventory differs')
    for name, expected in manifest['files'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source path')
        check_file(dict(path=str(root / rel), sha256=expected['sha256'] if isinstance(expected, dict) else expected))
    package = '_opsd_analysis_frozen_' + proof['sha256'][:20]
    if package in sys.modules:
        require(Path(sys.modules[package].__file__).resolve() == root / '__init__.py', 'Frozen package cache collision')
    finder = SourceOnlyHelper(package, root, manifest['files'])
    for name, module in list(sys.modules.items()):
        if name == package or name.startswith(package + '.'):
            path, digest, _ = finder.resolve(name)
            require(Path(getattr(module, '__file__', '')).resolve() == path
                and getattr(module, '_analysis_frozen_source_sha256', None) == digest,
                'Cached frozen helper module lacks exact compiled source identity')
    sys.meta_path[:] = [x for x in sys.meta_path if not isinstance(x, SourceOnlyHelper) or x.package != package]
    sys.meta_path.insert(0, finder)
    importlib.import_module(package)
    return {name: importlib.import_module(package + '.' + name) for name in
        ('checkpoint', 'prepare', 'gates', 'evaluation', 'evaluation_worker',
         'training_metrics', 'objective', 'sibling_scores', 'precision', 'targets')}


class SourceOnlyHelper(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Never execute a stale helper pyc, even with matching timestamp/size."""
    def __init__(self, package, root, files):
        self.package, self.root, self.files = package, Path(root).resolve(), files

    def resolve(self, fullname):
        relative = fullname[len(self.package):].lstrip('.').replace('.', '/')
        candidates = [('__init__.py', True)] if not relative else [
            (relative + '/__init__.py', True), (relative + '.py', False)]
        for name, package in candidates:
            if name in self.files:
                expected = self.files[name]; digest = expected['sha256'] if isinstance(expected, dict) else expected
                path = self.root / name
                require(path.resolve().is_relative_to(self.root), 'Frozen helper module escaped root')
                return path, digest, package
        raise ValueError('Module absent from frozen helper manifest: ' + fullname)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.package and not fullname.startswith(self.package + '.'):
            return None
        source, _, package = self.resolve(fullname)
        return importlib.util.spec_from_loader(fullname, self, origin=str(source), is_package=package)

    def create_module(self, spec): return None

    def exec_module(self, module):
        path, digest, package = self.resolve(module.__name__)
        actual = check_file(dict(path=str(path), sha256=digest)); before = stat(path); raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'Frozen compiled helper bytes differ')
        module.__file__ = str(path); module.__package__ = module.__name__ if package else module.__name__.rpartition('.')[0]
        if package: module.__path__ = [str(path.parent)]
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        module._analysis_frozen_source_sha256 = digest
        require(before == stat(path), 'Helper source changed while compiling')


def checked_endpoint(config, receipt_ref, modules):
    value, proof = read_json(receipt_ref)
    require(value.get('schema') == 'opsd-signal-batched-canonical-endpoint-v1'
        and value.get('accepted') is True and value['primary_slots'] == 512
        and value['generation_protocol'] == 'vllm-greedy-rep1p05-bf16-merged-fixed512-v1'
        and value['actual_worker_wait_returncodes'] == [0]*8
        and (value['version'], value['completed_outer']) in ENDPOINTS,
        'Only actual complete batched zero/15/25 endpoints are replay inputs')
    require(value['run_id'] == config['run_id'] and value['study_id'] == config['study_id']
        and value['config']['sha256'] == config['_config_sha256'], 'Endpoint belongs to another config')
    original, _ = read_json(value['batched_input']); summary, _ = read_json(value['batched_summary'])
    require(original['accepted'] is True and original['config']['sha256'] == value['config']['sha256']
        and original['raw_checkpoint'] == (None if value['completed_outer']==0 else value['raw_checkpoint']['manifest'])
        and summary['actual_worker_wait_returncodes']==[0]*8, 'Actual batched provenance changed')
    records, _ = read_rows(value['results']); data, _ = read_rows(config['validation_identity'])
    modules['prepare'].verify_runtime(config)
    core=importlib.import_module('search_e1.core');qa=importlib.import_module('search_r1.qa_protocol')
    try: from .adapter import canonical_rows
    except ImportError: from adapter import canonical_rows
    decoded=canonical_rows(data, records, lambda ref:read_json(ref)[0],
        lambda result,answers:modules['training_metrics'].protocol(core,qa,result,answers))
    require(decoded==records, 'Canonical response/classification changed')
    return value, proof, records, data


def cpu_only():
    require('torch' not in sys.modules and os.environ.get('CUDA_VISIBLE_DEVICES', '') == '',
        'Analysis CPU parent must not import Torch or see CUDA')
