"""Pinned small JSON readers. No raw rollout, tensor, model or network imports."""
import hashlib
import json
import math
from pathlib import Path

VARIANTS = ('G_POS', 'G_NEG', 'H_POS', 'H_NEG')
ENDPOINTS = {('source100', 0)} | {(v, t) for v in VARIANTS for t in (15, 25)}
STUDY_ROOT = '/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
MAX_JSON = 32 << 20
MAX_AGGREGATE = 256 << 10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and all(a.get(k) == b.get(k) for k in ('path', 'sha256'))


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode() + b'\n'


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/invalid ' + label)
    return value


class Reader:
    """Rehash every bound input at the end; metadata alone is not evidence."""
    def __init__(self):
        self.proofs = {}

    def file(self, ref):
        require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and
                isinstance(ref.get('sha256'), str) and len(ref['sha256']) == 64,
                'Explicit immutable path/SHA required')
        path = Path(ref['path'])
        require(path.is_absolute() and not path.is_symlink() and path.is_file(), 'Absolute regular input required')
        s = path.stat()
        before = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                h.update(block)
        z = path.stat()
        require(before == (z.st_dev, z.st_ino, z.st_size, z.st_mtime_ns, z.st_ctime_ns) and
                h.hexdigest() == ref['sha256'] and ref.get('bytes', s.st_size) == s.st_size,
                'Input changed or pinned SHA differs: ' + str(path))
        proof = dict(path=str(path), sha256=h.hexdigest(), bytes=s.st_size)
        key = str(path)
        require(key not in self.proofs or self.proofs[key] == proof, 'Conflicting immutable input refs')
        self.proofs[key] = proof
        return proof

    def json(self, ref):
        proof = self.file(ref)
        require(proof['bytes'] <= MAX_JSON, 'JSON exceeds bounded renderer input budget')
        raw = Path(proof['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == proof['sha256'], 'Parsed JSON changed')
        return json.loads(raw)

    def stable_end(self):
        for proof in list(self.proofs.values()):
            self.file(proof)


def published(reader, packet):
    manifest = reader.json(packet['source_manifest'])
    pub = reader.json(packet['publication'])
    require(pub.get('accepted') is True and pub.get('branch') == 'search-e1' and
            isinstance(pub.get('source_commit'), str) and len(pub['source_commit']) == 40 and
            pub['source_commit'] == pub.get('remote_readback_sha') and
            pub.get('source_manifest_sha256', pub.get('helper_source_manifest_sha256')) == packet['source_manifest']['sha256'],
            'Source publication/readback binding missing')
    root = Path(packet['helper_root']).resolve()
    files = manifest['files']
    require(isinstance(files, dict) and files, 'Published source inventory absent')
    for name, entry in files.items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe source manifest path')
        reader.file(dict(path=str(root / rel), sha256=entry['sha256'], bytes=entry['bytes']))
    require({n for n in files if n.endswith('.py')} == {str(p.relative_to(root)) for p in root.rglob('*.py')},
            'Published Python inventory changed')
    return pub


def native_packet(reader, packet, kind):
    require(packet.get('mode') == 'native_cli_artifact', 'Native CLI artifact packet required')
    published(reader, packet)
    cli = reader.json(packet['actual_stdout'])
    exe = reader.json(packet['execution'])
    require(exe.get('accepted') is True and exe.get('actual_wait_returncode') == 0 and
            exe.get('torch_imported') is False and same(exe.get('stdout'), packet['actual_stdout']) and
            same(exe.get('source_manifest'), packet['source_manifest']) and
            same(exe.get('publication'), packet['publication']), 'Actual CLI wait/source/stdout binding invalid')
    if kind == 'train_relationship':
        complete = reader.json(cli)
        require(complete.get('accepted') is True and complete.get('complete_100_outers') is True,
                'TRAIN CLI complete artifact is partial')
        report_ref = complete['outputs']['summary']
        original_request = reader.json(complete['request'])
    else:
        require(cli.get('accepted') is True and same(cli.get('output'), packet['output']), 'CLI did not name accepted original output')
        report_ref = cli['output']
        original_request = None
    require(same(report_ref, packet['output']), 'Relabeled CLI report forbidden')
    value = reader.json(report_ref)
    require(value.get('accepted') is True, 'Report not accepted')
    if kind not in ('coverage', 'train_relationship'):
        original_request = reader.json(value['request'])
    return value, original_request
