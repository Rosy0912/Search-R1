"""Small immutable readers and source-only reuse of published stdlib helpers."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

VARIANTS = ('G_POS', 'G_NEG', 'H_POS', 'H_NEG')
ORIGINAL_SHA = 'af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314'
STUDY_ROOT = '/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def stat(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def checked_file(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str)
            and isinstance(ref.get('sha256'), str) and len(ref['sha256']) == 64,
            'Explicit immutable file path/SHA required')
    path = Path(ref['path']).resolve(); before = stat(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    require(stat(path) == before and digest.hexdigest() == ref['sha256']
            and ref.get('bytes', before[2]) == before[2], 'File changed or SHA differs: ' + str(path))
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=before[2])


def read(ref):
    proof = checked_file(ref); before = stat(proof['path'])
    raw = Path(proof['path']).read_bytes()
    require(stat(proof['path']) == before and hashlib.sha256(raw).hexdigest() == proof['sha256'],
            'JSON parsed bytes differ from pinned SHA or changed while reading')
    value = json.loads(raw)
    return value, proof


def rows(ref):
    proof = checked_file(ref); before = stat(proof['path']); raw = Path(proof['path']).read_bytes()
    require((not raw or raw.endswith(b'\n')) and stat(proof['path']) == before
            and hashlib.sha256(raw).hexdigest() == proof['sha256'], 'JSONL parsed bytes differ from pinned SHA, changed or truncated')
    return [json.loads(line) for line in raw.splitlines() if line.strip()], proof


def same(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and all(left.get(k) == right.get(k) for k in ('path', 'sha256'))


def exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return checked_file(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


class SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, package, root, files):
        self.package, self.root, self.files = package, Path(root).resolve(), files

    def resolve(self, name):
        rel = name[len(self.package):].lstrip('.').replace('.', '/')
        candidates = [('__init__.py', True)] if not rel else [(rel + '/__init__.py', True), (rel + '.py', False)]
        for path, package in candidates:
            if path in self.files:
                entry = self.files[path]
                return self.root / path, entry['sha256'], package
        raise ValueError('Module absent from bound helper source: ' + name)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.package or fullname.startswith(self.package + '.'):
            source, _, package = self.resolve(fullname)
            return importlib.util.spec_from_loader(fullname, self, origin=str(source), is_package=package)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, digest, package = self.resolve(module.__name__); checked_file(dict(path=str(path), sha256=digest))
        raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == digest, 'Compiled bytes differ')
        module.__file__ = str(path); module.__package__ = module.__name__ if package else module.__name__.rpartition('.')[0]
        if package:
            module.__path__ = [str(path.parent)]
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        module._completion_source_sha256 = digest


def bind_helper(root, manifest_ref, modules):
    manifest, mp = read(manifest_ref); root = Path(root).resolve(); files = manifest['files']
    require(set(n for n in files if n.endswith('.py')) == {str(p.relative_to(root)) for p in root.rglob('*.py')},
            'Published helper Python inventory differs')
    for name, entry in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe helper manifest path')
        checked_file(dict(path=str(root / relative), sha256=entry['sha256'], bytes=entry['bytes']))
    package = '_completion_bound_' + mp['sha256'][:20]
    finder = SourceOnly(package, root, files)
    for name, module in list(sys.modules.items()):
        if name == package or name.startswith(package + '.'):
            path, digest, _ = finder.resolve(name)
            require(Path(module.__file__).resolve() == path and module._completion_source_sha256 == digest,
                    'Cached helper source collision')
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, SourceOnly) or f.package != package]
    sys.meta_path.insert(0, finder); importlib.import_module(package)
    result = {name: importlib.import_module(package + '.' + name) for name in modules}
    require('torch' not in sys.modules, 'CPU coverage helper unexpectedly imported Torch')
    return result


def execution_packet(packet, require_no_torch=True, require_accepted_output=True):
    """Accept recorded execution only after source/publication/output fresh SHA."""
    output, op = read(packet['output']); execution, ep = read(packet['execution'])
    publication, pp = read(packet['publication']); manifest, mp = read(packet['source_manifest'])
    require(publication.get('accepted') is True and publication.get('source_commit') == publication.get('remote_readback_sha')
            and isinstance(publication.get('source_commit'), str) and len(publication['source_commit']) == 40
            and publication.get('source_manifest_sha256', publication.get('helper_source_manifest_sha256')) == mp['sha256'],
            'External helper lacks matching published source/manifest/readback')
    root = Path(packet['helper_root']).resolve()
    for name, entry in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe external source path')
        checked_file(dict(path=str(root / name), sha256=entry['sha256'], bytes=entry['bytes']))
    require(execution.get('accepted') is True and execution.get('actual_wait_returncode') == 0
            and same(execution.get('stdout'), op) and same(execution.get('source_manifest'), mp)
            and same(execution.get('publication'), pp)
            and (not require_no_torch or execution.get('torch_imported') is False),
            'External output lacks actual wait0/device-scope and exact execution/source/publication binding')
    require(not require_accepted_output or output.get('accepted') is True, 'External audit did not accept')
    return output, dict(output=op, execution=ep, publication=pp, source_manifest=mp)
