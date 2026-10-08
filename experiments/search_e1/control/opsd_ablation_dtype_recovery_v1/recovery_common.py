"""Source-only, immutable bindings for one narrow adapter-loading repair."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

ORIGINAL_COMMIT = '871dc46f9ae436e5b4115dd33e51715b858ee57d'
ORIGINAL_MANIFEST_SHA = '8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41'
ORIGINAL_CONTROLLER_SHA = 'c8aa9c8f81bb669707085d0a586843608058258a3882006b380a28c556f26b56'
ORIGINAL_TRAINER_SHA = '4dc352f5091594ab5bcae71ecc17516afd8607601b5932216cbda66befc5409c'
VARIANTS = {'C0', 'C1', 'M', 'R', 'W', 'P', 'J'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(ref):
    path = Path(ref['path']).resolve()
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) for k in fields), 'Bound file changed while reading')
    require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'Bound file SHA differs: ' + str(path))
    return data


def checked_json(ref):
    return json.loads(checked(ref))


def same_ref(left, right):
    return left['path'] == right['path'] and left['sha256'] == right['sha256']


def tree(root, manifest_ref):
    root = Path(root).resolve()
    manifest = checked_json(manifest_ref)
    for name, entry in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source inventory')
        checked(dict(path=str(root / relative), sha256=entry['sha256'] if isinstance(entry, dict) else entry))
    expected = {n for n in manifest['files'] if n.endswith('.py')}
    require(expected == {str(p.relative_to(root)) for p in root.rglob('*.py')}, 'Python source inventory differs')
    return manifest


def bind(repair_ref, config_ref, run_id, *, resume=None):
    value = checked_json(repair_ref)
    require(value['schema'] == 'opsd-ablation-dtype-repair-manifest-v1' and value['accepted'] is True,
            'Explicit immutable dtype repair manifest required')
    root = Path(value['helper_root']).resolve()
    require(Path(__file__).resolve().parent == root, 'Repair helper loaded from another directory')
    source = tree(root, value['source_manifest'])
    publication = checked_json(value['publication'])
    require(publication.get('accepted') is True and publication.get('branch') == 'search-e1'
            and publication['source_commit'] == publication['remote_readback_sha'] == value['source_commit']
            and len(value['source_commit']) == 40
            and publication['source_manifest_sha256'] == value['source_manifest']['sha256'],
            'Repair source has not been published and read back')
    refs = value['original_configs']
    require(len(refs) == 7 and len({(r['path'], r['sha256']) for r in refs}) == 7,
            'Seven distinct original immutable config references required')
    configs = [checked_json(r) for r in refs]
    require({c['variant'] for c in configs} == VARIANTS, 'Seven exact ablation variants required')
    require(all(c['schema'] == 'opsd-ablation-config-v1' and c['study_id'] == value['study_id']
                and c['source_commit'] == ORIGINAL_COMMIT
                and same_ref(c['helper_source_manifest'], value['original_helper_manifest']) for c in configs),
            'Original study/config/helper binding differs')
    roots = {str(Path(c['paths']['helper_root']).resolve()) for c in configs}
    require(len(roots) == 1 and value['original_helper_manifest']['sha256'] == ORIGINAL_MANIFEST_SHA,
            'Only the original frozen 871 training helper may be overlaid')
    original_root = Path(roots.pop())
    original = tree(original_root, value['original_helper_manifest'])
    for name, expected in [('controller.py', ORIGINAL_CONTROLLER_SHA), ('trainer.py', ORIGINAL_TRAINER_SHA)]:
        entry = original['files'][name]
        require((entry['sha256'] if isinstance(entry, dict) else entry) == expected, 'Original execution source differs')
    matches = [(r, c) for r, c in zip(refs, configs) if same_ref(r, config_ref) and c['run_id'] == run_id]
    require(len(matches) == 1, 'CLI config/run is outside the pinned seven originals')
    config = matches[0][1]
    require(os.environ.get('E1_RUN_ID') == run_id, 'Actual run marker differs')
    if resume is not None:
        require(resume == (config['variant'] == 'C0'), 'C0 resumes original output; other groups start fresh')
    return value, config, original_root, original, source


class OriginalSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile verified original bytes; never borrow a cached helper .pyc."""
    def __init__(self, root, manifest):
        self.root = root
        self.files = {Path(n).stem: (n, e) for n, e in manifest['files'].items()
                      if n.endswith('.py') and '/' not in n and n != '__init__.py'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.files:
            return None
        return importlib.util.spec_from_loader(fullname, self, origin=str(self.root / self.files[fullname][0]))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name, entry = self.files[module.__name__]
        path = self.root / name
        payload = checked(dict(path=str(path), sha256=entry['sha256'] if isinstance(entry, dict) else entry))
        module.__file__ = str(path)
        exec(compile(payload, str(path), 'exec'), module.__dict__)


def install_original(root, manifest):
    loader = OriginalSources(root, manifest)
    require(not any(name in sys.modules for name in loader.files), 'Original helper module was already imported/shadowed')
    sys.meta_path.insert(0, loader)
    sys.path.insert(0, str(root))
    return loader


def receipt(config, repair_ref, role, fields):
    directory = Path(config['paths']['receipts']) / 'dtype-recovery' / repair_ref['sha256']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (role + '-pid-' + str(os.getpid()) + '.json')
    value = dict(schema='opsd-ablation-dtype-repair-execution-v1', accepted=True,
                 repair_manifest=repair_ref, run_id=config['run_id'], variant=config['variant'],
                 pid=os.getpid(), uid=os.getuid(), **fields)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return str(path)
