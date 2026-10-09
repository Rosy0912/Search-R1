"""Independent CPU provenance/I/O. No model import at module import time."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
VARIANTS = ('L',)
RUNTIME_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
VALIDATION_SHA = '75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76'
QA_SHA = '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
SCANNER_SHA = '2a66a863f7170035daea3d9a5f1d7a34106b12d12799f337021bf63747991e26'
VERSIONS = ('L_forward99_bf16',)


class Hold(ValueError):
    """Evidence/input not ready. This is not a training failure."""


def require(condition, reason):
    if not condition: raise Hold(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def metadata(path):
    path = Path(path).resolve(); s = path.stat()
    return dict(path=str(path), device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read_file(path, expected=None):
    before = metadata(path); digest = hashlib.sha256(); raw = bytearray()
    with Path(before['path']).open('rb') as handle:
        require(os.fstat(handle.fileno()).st_ino == before['inode'], 'Artifact inode changed')
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block); raw.extend(block)
        after_fd = os.fstat(handle.fileno())
    require(metadata(path) == before and after_fd.st_size == before['bytes'], 'Artifact changed during read')
    sha = digest.hexdigest(); require(expected is None or expected == sha, 'Artifact SHA differs')
    return bytes(raw), dict(before, sha256=sha)


def fileproof(path, expected=None):
    # Streaming SHA avoids retaining multi-GB source weights in RAM.
    before = metadata(path); digest = hashlib.sha256()
    with Path(before['path']).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''): digest.update(block)
        s = os.fstat(handle.fileno())
    require(metadata(path) == before and s.st_ino == before['inode'] and s.st_size == before['bytes'], 'File changed while hashing')
    require(expected is None or digest.hexdigest() == expected, 'File SHA differs')
    return dict(before, sha256=digest.hexdigest())


def checked_json(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and ref.get('sha256'), 'Missing explicit path/SHA input')
    raw, proof = read_file(ref['path'], ref['sha256']); return json.loads(raw), proof


def rows(ref):
    raw, proof = read_file(ref['path'], ref['sha256']); require(not raw or raw.endswith(b'\n'), 'Partial JSONL record')
    return [json.loads(line) for line in raw.splitlines()], proof


def atomic_json(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not exclusive or not path.exists(), 'Existing output is never reused')
    temp = path.with_name(path.name + '.pending-' + str(os.getpid()) + '-' + str(time.time_ns()))
    with temp.open('x') as handle:
        handle.write(canonical(value).decode() + '\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return fileproof(path)


def append_json(handle, value):
    handle.write(canonical(value).decode() + '\n'); handle.flush(); os.fsync(handle.fileno())


def check_metadata(proofs):
    for proof in proofs:
        require(metadata(proof['path']) == {k: proof[k] for k in metadata(proof['path'])}, 'Bound file metadata changed after load')


def fresh_module(name, path, sha):
    raw, proof = read_file(path, sha); module = types.ModuleType(name); module.__file__ = proof['path']
    exec(compile(raw, proof['path'], 'exec'), module.__dict__); check_metadata([proof]); return module


def original_config(config):
    # This is the explicit genuine common scientific reference, not L training.
    value, proof = checked_json(config['common_config'])
    require(value['schema']=='opsd-ablation-config-v1' and value['variant']=='C0'
        and value['study_id']==config['study_id'] and value['scientific']['outer_iterations']==10,
        'Wrong actual common science reference')
    require(value['source_model_manifest']['sha256']==SOURCE_MANIFEST_SHA
        and value['runtime_manifest']['sha256']==RUNTIME_SHA
        and value['validation_identity']['sha256']==VALIDATION_SHA
        and value['common_initial_adapter_tensor_sha256']==COMMON_TENSOR_SHA,
        'L/common source/data/template identity differs')
    value['_config_sha256'] = proof['sha256']; return value, proof


def source_files(config, full=True):
    train, _ = original_config(config); manifest, mp = checked_json(train['source_model_manifest'])
    require(manifest.get('source_grpo_step') == 100, 'Source model is not the transferred GRPO100')
    require(len(manifest['files']) == 12, 'Source100 must have twelve bound native files')
    root=Path(train['paths']['model']).resolve();paths=[Path(ref['path']) for ref in manifest['files']]
    require(len({p.resolve() for p in paths})==12 and not any(p.is_symlink() for p in paths)
        and {p.resolve() for p in paths}=={p.resolve() for p in root.rglob('*') if p.is_file()},
        'Source100 actual twelve-file inventory differs/aliases')
    proofs = []
    for ref in manifest['files']:
        require(Path(ref['path']).resolve().is_relative_to(Path(train['paths']['model']).resolve()), 'Source file escapes model root')
        if full: proof = fileproof(ref['path'], ref['sha256'])
        else:
            proof = metadata(ref['path']); require(all(proof[k] == ref[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns')), 'Source metadata differs')
            proof['sha256'] = ref['sha256']
        require(proof['bytes'] == ref['bytes'], 'Source file size differs'); proofs.append(proof)
    return dict(manifest=mp, files=proofs, full_sha_read=full)


def runtime_files(config):
    train, _ = original_config(config); inventory, proof = checked_json(train['runtime_manifest'])
    inventory = inventory.get('files', inventory); require(len(inventory) == 305 and proof['sha256']==RUNTIME_SHA, 'Wrong frozen scientific inventory')
    root = Path(train['paths']['runtime']).resolve(); files = {}
    for name, value in inventory.items():
        path = root / name; require(path.resolve().is_relative_to(root), 'Runtime source escapes frozen root')
        files[name] = fileproof(path, value if isinstance(value, str) else value['sha256'])
    require(files['upstream/Search-R1/search_r1/qa_protocol.py']['sha256'] == QA_SHA, 'Wrong actual QA scorer')
    return dict(root=str(root), manifest=proof, files=files)


def science(config):
    """Only workers/completion call this, after all source hashes are checked."""
    proof = runtime_files(config); root = Path(proof['root'])
    sys.path[:0] = [str(root/'upstream/Search-R1'), str(root)]
    finder = SourceOnlyScience(proof)
    replace_bound_clone=False;scientific_modules={}
    for name, old in list(sys.modules.items()):
        if name == 'search_e1' or name.startswith('search_e1.') or name == 'search_r1' or name.startswith('search_r1.'):
            path, expected, _ = finder.resolve(name)
            require(getattr(old,'_opsd_results_source_sha256',None) == expected,
                    'Foreign/pyc/cached scientific module is not source-only bound')
            scientific_modules[name]=old
            if Path(old.__file__).resolve()!=path:
                # The sequential CPU parent moves between seven byte-identical
                # frozen runtime copies. Only our already SHA-bound modules may
                # be cold-replaced, and both old and new source bytes are checked.
                fileproof(old.__file__,expected);replace_bound_clone=True
    if replace_bound_clone:
        for name in scientific_modules:sys.modules.pop(name)
    sys.meta_path[:]=[x for x in sys.meta_path if not isinstance(x,SourceOnlyScience)]
    sys.meta_path.insert(0, finder)
    core = importlib.import_module('search_e1.core')
    qa = importlib.import_module('search_r1.qa_protocol')
    for module, key in [(core,'search_e1/core.py'), (qa,'upstream/Search-R1/search_r1/qa_protocol.py')]:
        require(Path(module.__file__).resolve() == Path(proof['files'][key]['path']), 'Scientific module path differs')
        fileproof(module.__file__, proof['files'][key]['sha256'])
    return core, qa, proof


class SourceOnlyScience(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Read SHA-bound source bytes, never SourceFileLoader/pyc."""
    def __init__(self, proof): self.proof=proof; self.root=Path(proof['root'])
    def resolve(self, fullname):
        base=fullname.replace('.', '/')
        if fullname=='search_r1' or fullname.startswith('search_r1.'): base='upstream/Search-R1/'+base
        for key, package in ((base+'/__init__.py',True),(base+'.py',False)):
            if key in self.proof['files']:
                row=self.proof['files'][key]; return Path(row['path']),row['sha256'],package
        raise Hold('Scientific dependency absent from frozen305 source: '+fullname)
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname=='search_e1' or fullname.startswith('search_e1.') or fullname=='search_r1' or fullname.startswith('search_r1.')): return None
        source, _, package=self.resolve(fullname)
        return importlib.util.spec_from_loader(fullname,self,origin=str(source),is_package=package)
    def create_module(self,spec): return None
    def exec_module(self,module):
        path, sha, package=self.resolve(module.__name__); raw,proof=read_file(path,sha)
        module.__file__=str(path); module.__package__=module.__name__ if package else module.__name__.rpartition('.')[0]
        if package: module.__path__=[str(path.parent)]
        exec(compile(raw,str(path),'exec'),module.__dict__)
        module._opsd_results_source_sha256=sha; check_metadata([proof])


def cohort(config):
    train, _ = original_config(config); require(train['validation_identity']['sha256'] == VALIDATION_SHA, 'Different validation512')
    data, fp = rows(train['validation_identity'])
    require(len(data) == len({x['question_id'] for x in data}) == len({x['question'] for x in data}) == 512,
            'Validation must have 512 unique IDs/questions')
    from collections import Counter
    require(Counter(x['data_source'] for x in data) == Counter(nq=256, hotpotqa=256), 'Validation dataset balance differs')
    require(all(set(x) == {'question_id','question','answers','data_source'} and x['answers'] for x in data), 'Validation schema differs')
    return data, dict(file=fp, semantic_sha256=[object_sha(x) for x in data], ranks=[list(range(r,512,8)) for r in range(8)])


def scanner(config):
    train, _ = original_config(config); path = Path(train['paths']['helper_root'])/'owned_process_audit.py'
    return fresh_module('online_results_ownership', path, SCANNER_SHA)


SOURCE_MANIFEST_SHA='2ef6070a328357b0f9d3a026010aaed0aa3f8840a241bc1021ed9a183e8d75f7'
COMMON_TENSOR_SHA='836590b798434f21160418c6471cbce3bdfac5bf4a9947c1e3e7723d057baf2f'
L_TENSOR_SHA='2c4a39a37d5bd47ba20b06b6e20b59dc323f86746d2d84514585bf92fedc37ce'
QUALIFICATION_SHA='3da5b18e3431901a04f146d45091134cc3f90aeeb0130b35439fd3277fe39f10'
TARGET_PATH='/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/runs/checkpoints/update-00000099-outer-00009/manifest.json'
TARGET_SHA='476c1b1003d8d8f4cc4711c462b013ed249bfbddc6d4a5235686f5e2ff946ad4'
HISTORICAL_RUN='1d1d67b60f0e46a2afc0536a36ab69b1'
