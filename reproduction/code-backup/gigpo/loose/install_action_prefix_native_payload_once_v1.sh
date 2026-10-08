python3 work/remote.py h100-8 <<'SH'
/usr/bin/python3 - <<'PY'
import datetime, hashlib, json, os, socket, subprocess, tarfile
from pathlib import Path
root = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
archive = root / 'logs/action-prefix-native-install-code-metadata-v1.tar.gz'
receipt_path = root / 'logs/ACTION-PREFIX-NATIVE-ACTUAL-INSTALL-2262942-v1.json'
digest = lambda raw: hashlib.sha256(raw).hexdigest()
assert socket.gethostname().split('.')[0] == 'h100-8'
assert digest(archive.read_bytes()) == '079ce645031ab4af4603381db42b695b4e7f9150e991f4b347c16a83fa16768c'
assert not os.path.lexists(receipt_path)
with tarfile.open(archive, 'r:gz') as ar:
    members = ar.getmembers()
    assert all(m.isfile() and not m.issym() and not m.islnk() for m in members)
    raw = ar.extractfile('manifest.json').read()
    assert digest(raw) == '65c6197b36040e9cf1ffae34ff9108ad0f8c24250bb77a494042c9537f35e36e'
    manifest = json.loads(raw)
    assert manifest['source_commit'] == '2262942142f9df8f1666ce3f1b9ebfe9c9239241'
    assert manifest['scope'] == 'CODE_AND_BOUNDED_METADATA_ONLY_NO_PT_OR_MODEL_DATA_CACHE'
    assert len(manifest['files']) == 70
    assert {m.name for m in members} == {'manifest.json'} | {x['member'] for x in manifest['files']}
    payloads = []
    for x in manifest['files']:
        p = Path(x['path'])
        assert p.is_absolute() and str(p).startswith(str(root) + '/') and '..' not in p.parts
        assert p.suffix in {'.json', '.py', '.log', '.patch', '.sbatch'}
        assert not os.path.lexists(p) and x['bytes'] < 16 * 1024 * 1024
        data = ar.extractfile(x['member']).read()
        assert len(data) == x['bytes'] and digest(data) == x['sha256']
        payloads.append((p, data, x))
assert len({str(p) for p, _, _ in payloads}) == 70
for directory in manifest['fresh_directories']:
    p = Path(directory)
    assert str(p).startswith(str(root) + '/') and not os.path.lexists(p)
    assert p.parent.is_dir() and p.parent.resolve() == p.parent
repo = root / 'repo-raw-action-2262942'
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == manifest['source_commit']
assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
for directory in manifest['fresh_directories']:
    Path(directory).mkdir()
for p, data, x in payloads:
    p.parent.mkdir(parents=True, exist_ok=True)
    assert p.parent.resolve() == p.parent
    with p.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert p.stat().st_size == x['bytes'] and digest(p.read_bytes()) == x['sha256']
record = dict(status='ACTUAL_CODE_AND_METADATA_INSTALL_NOT_READY_NOT_GPU',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node='h100-8',
    source_commit=manifest['source_commit'], candidate_sha256=manifest['candidate_sha256'],
    archive=str(archive), archive_sha256=digest(archive.read_bytes()),
    manifest_sha256=digest(raw), installed_files=[{k:x[k] for k in ('path','bytes','sha256')} for _,_,x in payloads],
    gpu_used=False, native_submitted=False, formal_updates=0)
with receipt_path.open('x') as stream:
    stream.write(json.dumps(record, indent=2) + '\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(dict(receipt={'path':str(receipt_path),'bytes':receipt_path.stat().st_size,
    'sha256':digest(receipt_path.read_bytes())}, actual=record)))
PY
SH
