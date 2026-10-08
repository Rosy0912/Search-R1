"""Local preparation only; never contacts a host or executes the publisher."""
import base64
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import stat

H = Path(__file__).resolve().parent
BUNDLE = H / 'final200-prerequisites-runtime-bundle-85f014e167c8'
OUT = H / 'strict200-bundle-publication-preparation-v1'
MANIFEST_SHA = '85f014e167c8ca0fbb0ea4cf74bf2065e012d8efa1ec89c38b1f7240b09a62eb'
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
ROOT = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006'
BASE = H / 'native-cpu-bundle-publication-preparation-v1' / 'publisher.py'
BASE_SHA = 'daec377ae1caa1d298975188e77d305ac301f57f97ed55cf8bc2c42d32aeefad'
REVIEW = H / 'STRICT_FINAL200_PREREQUISITES_BUNDLE_INDEPENDENT_REVIEW_V2.json'
REVIEW_SHA = '60fc31e4f9cf6d2e2879c9ba5a8d9a503cf2ba0918c0ec74637c4e3adf63c127'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def physical(path):
    if path.resolve(strict=True) != path:
        raise ValueError('A preparation input must be a physical original')
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1:
        raise ValueError('Owned single-link regular input required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(tuple(getattr(s, k) for k in fields) != tuple(getattr(before, k) for k in fields)
           for s in (opened, after_fd, after)):
        raise ValueError('Input changed while reading')
    raw = b''.join(chunks)
    return raw, {'physical_path': str(path), 'sha256': digest(raw), 'bytes': len(raw),
                 'device': before.st_dev, 'inode': before.st_ino, 'uid': before.st_uid,
                 'mode': stat.S_IMODE(before.st_mode), 'nlink': before.st_nlink,
                 'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns}


def syncdir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(name, raw):
    fd = os.open(OUT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    syncdir(OUT)


def main():
    records = {}
    manifest_raw, records['manifest'] = physical(BUNDLE / 'manifest.json')
    if digest(manifest_raw) != MANIFEST_SHA:
        raise ValueError('Actual bundle manifest drift')
    manifest = json.loads(manifest_raw)
    if set(manifest) != {'version', 'kind', 'source_commit', 'tools'} or type(manifest['version']) is not int or manifest['version'] != 1:
        raise ValueError('Original manifest schema')
    if manifest['kind'] != 'STRICT_FINAL200_PREREQUISITES_READ_ONLY_RUNTIME_BUNDLE' or manifest['source_commit'] != SOURCE or len(manifest['tools']) != 40:
        raise ValueError('Original strict200 bundle identity')
    if stat.S_IMODE(BUNDLE.stat().st_mode) != 0o555 or {p.name for p in BUNDLE.iterdir()} != set(manifest['tools']) | {'manifest.json'}:
        raise ValueError('Original immutable bundle members')
    files = {}
    for name, expected in manifest['tools'].items():
        if Path(name).name != name or name == 'manifest.json':
            raise ValueError('Unsafe original member')
        raw, records[name] = physical(BUNDLE / name)
        if digest(raw) != expected or records[name]['mode'] != 0o444:
            raise ValueError('Original member bytes/mode')
        files[name] = base64.b64encode(raw).decode('ascii')
    if manifest['tools']['common.py'] != 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581' or manifest['tools']['verify_actual200_prerequisites.py'] != '7dff811b01b302f45c729ee12ae9ee22bb561b359f24fcd0d37352192db2d075':
        raise ValueError('Semantic runtime must be strict200, not native ee31 control/common')
    review_raw, records['bundle_independent_review'] = physical(REVIEW)
    review = json.loads(review_raw)
    if digest(review_raw) != REVIEW_SHA or review['manifest_sha256'] != MANIFEST_SHA or review['status'] != 'PASS_LOCAL_IMMUTABLE_PACKAGING_AND_COMPOSITION_ONLY_ACTUAL200_BLOCKED':
        raise ValueError('Actual independent packaging qualification')
    base_raw, records['successful_base_publisher'] = physical(BASE)
    if digest(base_raw) != BASE_SHA:
        raise ValueError('Original actual-success publisher drift')
    changes = [
        ('ee31da65570318227ad6b037187cffa14e4dc489bac2a504d2bc54010c95da05', MANIFEST_SHA),
        ('IMMUTABLE_C5BF_NATIVE_CPU_PREPARATION_SOFTWARE_BUNDLE', manifest['kind']),
        ('c5bf4c8db33562a259fa227a940dc69543901b98', SOURCE),
        ("len(m['tools'])==22", "len(m['tools'])==40"),
        ("'final200-eval-native-cpu-'", "'final200-prerequisites-'"),
        ('Actual isolated c5bf CPU-preparation tool publication only; no native tool execution, actual200/model/GPU/eval/training/receipt/newallocation',
         'Actual isolated strict200 prerequisites tool publication only; no verifier/common execution, signature/receipt/scheduler/GPU/eval/training/actual200 acceptance'),
    ]
    original = base_raw.decode('utf-8')
    publisher = original
    for old, new in changes:
        if publisher.count(old) != 1:
            raise ValueError('Expected one explicit publisher difference')
        publisher = publisher.replace(old, new, 1)
    compile(publisher, str(OUT / 'publisher.py'), 'exec')
    payload = {'manifest_base64': base64.b64encode(manifest_raw).decode('ascii'),
               'manifest_sha256': MANIFEST_SHA, 'files': files}
    for record in records.values():
        _, later = physical(Path(record['physical_path']))
        if record != later:
            raise ValueError('Preparation input changed before exclusive output')
    OUT.mkdir(mode=0o700)
    syncdir(OUT.parent)
    syncdir(OUT)
    write_once('publisher.py', publisher.encode())
    write_once('payload.original.json', (json.dumps(payload, sort_keys=True) + '\n').encode())
    write_once('publisher-from-successful-base.diff', ''.join(difflib.unified_diff(original.splitlines(True), publisher.splitlines(True), fromfile=str(BASE), tofile=str(OUT / 'publisher.py'))).encode())
    record = {'version': 1, 'kind': 'LOCAL_STRICT200_BUNDLE_PUBLICATION_PREPARATION_NOT_DEPLOYMENT',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_commit': SOURCE, 'bundle_manifest_sha256': MANIFEST_SHA,
              'remote_target': ROOT + '/tools/final200-prerequisites-' + MANIFEST_SHA[:12],
              'publisher_sha256': digest(publisher.encode()), 'base_publisher_sha256': BASE_SHA,
              'literal_only_differences': changes, 'stable_input_records': records,
              'actual_remote_deployment': None, 'publisher_executed': False,
              'actual200': None, 'actual_signed_receipt': None,
              'scope': 'Local preparation and full byte comparison only; no network, main, signature, scheduler or GPU.'}
    write_once('preparation.original.json', (json.dumps(record, sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in record.items() if k != 'stable_input_records'}, sort_keys=True))


if __name__ == '__main__':
    main()
