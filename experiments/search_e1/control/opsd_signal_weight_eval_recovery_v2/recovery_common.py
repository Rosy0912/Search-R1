"""Immutable source bindings for a narrow named-adapter storage recovery."""
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

ORIGINAL_COMMIT = '00fec71f6e2481f9e1bc4576d956748bdda77aaa'
ORIGINAL_MANIFEST_SHA = 'af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314'
ORIGINAL_CONTROLLER_SHA = '87c767b4d1e739d0a8c4f2088c687cf36ebf1067f775b83a3775b07e6dd1763b'
ORIGINAL_TRAINER_SHA = '74b9f70e6940b03a61035224d8d22605832798967346761167cd5b9284ae25fe'
VARIANTS = {'G_POS', 'G_NEG', 'H_POS', 'H_NEG'}
CONFIG_SHAS = {
    'G_POS': 'e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661',
    'G_NEG': '2c37c851193dbc9a2d2512a72dd49310b9b8bcc14ac903159af4a491ef773a0d',
    'H_POS': '934f3e1ab8f70d7a1faa14b9806e9ddaecdbaea6afc97668ca3389a5c7237612',
    'H_NEG': 'ddb7d18dfcd5826f6aede00bcb507cebe553d8af5a28b9ef3d299c137b92c8bc',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(ref):
    path = Path(ref['path']).resolve()
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, name) == getattr(after, name) for name in fields),
            'Bound file changed while reading')
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
    expected = {name for name in manifest['files'] if name.endswith('.py')}
    require(expected == {str(path.relative_to(root)) for path in root.rglob('*.py')},
            'Python source inventory differs')
    return manifest


def bind(repair_ref, config_ref, run_id, *, cpu_supervisor=False):
    """Keep every original config/run/source byte; recovery is separate evidence."""
    value = checked_json(repair_ref)
    require(value['schema'] == 'opsd-signal-weight-evaluation-recovery-manifest-v2'
            and value['accepted'] is True
            and value['scope'] == 'evaluation_hold_retry_and_existing_named_outer_old_fp32',
            'Explicit immutable narrow recovery manifest required')
    root = Path(value['helper_root']).resolve()
    require(Path(__file__).resolve().parent == root, 'Recovery helper loaded from another directory')
    source = tree(root, value['source_manifest'])
    publication = checked_json(value['publication'])
    require(publication.get('accepted') is True and publication.get('branch') == 'search-e1'
            and publication['source_commit'] == publication['remote_readback_sha'] == value['source_commit']
            and len(value['source_commit']) == 40
            and publication['source_manifest_sha256'] == value['source_manifest']['sha256'],
            'Recovery source has not been published and read back')
    refs = value['original_configs']
    require(len(refs) == 4 and len({(ref['path'], ref['sha256']) for ref in refs}) == 4,
            'Four distinct original immutable configs required')
    configs = [checked_json(ref) for ref in refs]
    require({config['variant'] for config in configs} == VARIANTS, 'Four exact signal variants required')
    require(all(ref['sha256'] == CONFIG_SHAS[cfg['variant']] for ref, cfg in zip(refs, configs)),
            'Only the four original immutable config byte identities may be resumed')
    require(all(config['schema'] == 'opsd-signal-weight-config-v1'
                and config['study_id'] == value['study_id']
                and config['source_commit'] == ORIGINAL_COMMIT
                and same_ref(config['helper_source_manifest'], value['original_helper_manifest'])
                for config in configs), 'Original study/config/helper binding differs')
    roots = {str(Path(config['paths']['helper_root']).resolve()) for config in configs}
    require(len(roots) == 1 and value['original_helper_manifest']['sha256'] == ORIGINAL_MANIFEST_SHA,
            'Only original frozen00fec71 training helper may be overlaid')
    original_root = Path(roots.pop())
    original = tree(original_root, value['original_helper_manifest'])
    for name, expected in [('controller.py', ORIGINAL_CONTROLLER_SHA), ('trainer.py', ORIGINAL_TRAINER_SHA)]:
        entry = original['files'][name]
        require((entry['sha256'] if isinstance(entry, dict) else entry) == expected,
                'Original execution source differs')
    matches = [(ref, config) for ref, config in zip(refs, configs)
               if same_ref(ref, config_ref) and config['run_id'] == run_id]
    require(len(matches) == 1, 'CLI config/run is outside pinned four originals')
    if cpu_supervisor:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
                and os.getuid() == 1001,
                'Independent supervisor preflight requires UID1001, empty CUDA and no Torch import')
        require(os.environ.get('E1_RUN_ID') == value['supervisor_run_id'],
                'Actual supervisor marker differs from recovery binding')
    else:
        require(os.environ.get('E1_RUN_ID') == run_id, 'Actual run marker differs')
    return value, matches[0][1], original_root, original, source


class OriginalSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile verified original bytes without borrowing cached helper pyc."""
    def __init__(self, root, manifest):
        self.root = root
        self.files = {Path(name).stem: (name, entry) for name, entry in manifest['files'].items()
                      if name.endswith('.py') and '/' not in name and name != '__init__.py'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.files:
            return None
        return importlib.util.spec_from_loader(fullname, self,
                origin=str(self.root / self.files[fullname][0]))

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
    require(not any(name in sys.modules for name in loader.files),
            'Original helper module already imported or shadowed')
    sys.meta_path.insert(0, loader)
    sys.path.insert(0, str(root))
    return loader


def receipt(config, repair_ref, role, fields):
    directory = Path(config['paths']['receipts']) / 'signal-recovery' / repair_ref['sha256']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (role + '-pid-' + str(os.getpid()) + '.json')
    value = dict(schema='opsd-signal-weight-recovery-execution-v1', accepted=True,
                 repair_manifest=repair_ref, run_id=config['run_id'], variant=config['variant'],
                 pid=os.getpid(), uid=os.getuid(), **fields)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return str(path)
