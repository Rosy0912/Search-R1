"""Authorized private HF archives to HKUST3 /backup; token stays in RAM."""
import fcntl
import hashlib
import json
import os
import pathlib
import sys
import time
import requests

ROOT = pathlib.Path(__file__).resolve().parent

def save(name, data):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(str(tmp), str(p))

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def main():
    lock = (ROOT / 'downloader.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token = sys.stdin.readline().strip()
    assert token
    headers = {'Authorization': 'Bearer ' + token}
    plan = json.loads((ROOT / 'plan.json').read_text())
    repo = plan['repo']
    api = 'https://huggingface.co/api/models/' + repo
    info = requests.get(api, headers=headers, timeout=60)
    info.raise_for_status()
    assert info.json()['private'], 'Private repository required'
    records = plan['records']
    paths = ['topd-migration-20261009/archives/' + x['group'] + '.tar.zst' for x in records]
    meta = requests.post(api + '/paths-info/main', headers=headers,
                         json={'paths': paths}, timeout=60)
    meta.raise_for_status()
    inventory = {x['path']: x for x in meta.json()}
    for record, path in zip(records, paths):
        entry = inventory[path]
        assert entry['size'] == record['archive_bytes']
        assert entry['lfs']['oid'] == record['archive_sha256']
    save('hf-source-sha-inventory.json', {'epoch': time.time(), 'status': 'ALL_HF_SOURCE_ARCHIVE_SHA_MATCH', 'inventory': inventory})
    save('controller.json', {'pid': os.getpid(), 'epoch': time.time(), 'repo': repo, 'groups': len(records), 'root': str(ROOT), 'token_persisted': False})
    for record, path in zip(records, paths):
        name = record['group']
        final = ROOT / (name + '.tar.zst')
        part = ROOT / (name + '.' + record['archive_sha256'] + '.part')
        receipt = ROOT / (name + '-download-verified.json')
        if receipt.exists() and final.exists() and digest(final) == record['archive_sha256']:
            continue
        failures = 0
        while True:
            try:
                offset = part.stat().st_size if part.exists() else 0
                assert offset <= record['archive_bytes']
                if offset < record['archive_bytes']:
                    h = dict(headers)
                    if offset:
                        h['Range'] = 'bytes=%d-' % offset
                    url = 'https://huggingface.co/' + repo + '/resolve/main/' + path
                    with requests.get(url, headers=h, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if offset:
                            assert response.status_code == 206
                            assert response.headers['Content-Range'].startswith('bytes %d-' % offset)
                        else:
                            assert response.status_code == 200
                        last = 0
                        with part.open('ab') as out:
                            for b in response.iter_content(1048576):
                                if not b:
                                    continue
                                assert offset + len(b) <= record['archive_bytes']
                                out.write(b)
                                offset += len(b)
                                if time.time() - last >= 15:
                                    out.flush()
                                    save(name + '-progress.json', {'epoch': time.time(), 'group': name, 'bytes': offset, 'total': record['archive_bytes'], 'status': 'DOWNLOADING'})
                                    last = time.time()
                            out.flush()
                            os.fsync(out.fileno())
                assert part.stat().st_size == record['archive_bytes']
                assert digest(part) == record['archive_sha256'], 'Downloaded archive SHA mismatch'
                os.replace(str(part), str(final))
                save(receipt.name, {'epoch': time.time(), 'group': name, 'bytes': record['archive_bytes'], 'sha256': record['archive_sha256'], 'status': 'SOURCE_HF_DOWNLOAD_ARCHIVE_SHA256_VERIFIED'})
                print(json.dumps({'group': name, 'status': 'DOWNLOAD_SHA_VERIFIED'}), flush=True)
                break
            except Exception as error:
                failures += 1
                save(name + '-failure.json', {'epoch': time.time(), 'group': name, 'count': failures, 'error_type': type(error).__name__})
                print(json.dumps({'group': name, 'status': 'RETRY', 'count': failures, 'error_type': type(error).__name__}), flush=True)
                time.sleep(min(180, failures * 30))
    save('download-complete.json', {'epoch': time.time(), 'status': 'ALL_PLANNED_HF_DOWNLOAD_ARCHIVES_SHA_VERIFIED', 'groups': len(records)})

if __name__ == '__main__':
    main()
