"""Official, hash-checked, one-shard-at-a-time relay to this task's single cluster."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent.parent
META = BASE / 'outputs/qwen2.5-32b-official-metadata.json'
STAGING = Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-qwen25-32b-20261003')
REMOTE = '/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/qwen2.5-32b'
REVISION = '1818d35814b8319459f4bd55ed1ac8709630f003'
STATE = BASE / 'outputs/qwen25-32b-download-state.json'
STAGING.mkdir(parents=True, exist_ok=True)
lock = (STAGING / 'download.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
metadata = json.loads(META.read_text())
assert metadata['sha'] == REVISION
names = ['config.json', 'generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
         'vocab.json', 'merges.txt', 'model.safetensors.index.json']
entries = {x['rfilename']: x for x in metadata['siblings']}
names += sorted(n for n in entries if n.endswith('.safetensors'))
assert len([n for n in names if n.endswith('.safetensors')]) == 17
assert all('/' not in n and n in entries for n in names)
state = json.loads(STATE.read_text()) if STATE.exists() else {
    'model': 'Qwen/Qwen2.5-32B', 'revision': REVISION, 'remote_directory': REMOTE,
    'source': 'Official Hugging Face; no other cluster cache', 'files': {}, 'complete': False}
assert state['revision'] == REVISION and state['remote_directory'] == REMOTE
state['pid'] = os.getpid()

def save():
    state['updated_unix_seconds'] = time.time()
    temporary = STATE.with_suffix('.json.partial')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

try:
    for name in names:
        item = entries[name]
        existing = state['files'].get(name)
        if existing and existing.get('remote_sha_verified'):
            print('ALREADY_TRANSFERRED', name, flush=True)
            continue
        destination = STAGING / name
        if shutil.disk_usage(STAGING).free < int(item['size']) + 2 * 1024**3 and not destination.exists():
            raise RuntimeError('Insufficient staging free space; preserve existing files')
        state['active_file'] = name
        save()
        expected = (item.get('lfs') or {}).get('sha256')
        if not destination.exists() or destination.stat().st_size != item['size']:
            print('DOWNLOAD_OFFICIAL', name, item['size'], flush=True)
            url = f'https://huggingface.co/Qwen/Qwen2.5-32B/resolve/{REVISION}/{name}'
            subprocess.run(['curl', '--location', '--fail', '--connect-timeout', '15',
                            '--retry', '4', '--retry-delay', '3', '--speed-time', '60',
                            '--speed-limit', '8192', '--continue-at', '-', '--output',
                            str(destination), url], check=True)
        assert destination.stat().st_size == item['size'], name
        actual = digest(destination)
        assert expected is None or actual == expected, f'Official source SHA mismatch: {name}'
        print('OFFICIAL_FILE_VERIFIED', name, actual, flush=True)
        subprocess.run([sys.executable, str(BASE / 'work/transfer_file.py'), str(destination),
                        REMOTE + '/' + name], cwd=BASE, check=True)
        state['files'][name] = {'size': item['size'], 'sha256': actual,
                              'official_lfs_sha256': expected, 'remote_sha_verified': True}
        save()
        destination.unlink()
        print('VERIFIED_RELAY_FILE_REMOVED', name, flush=True)
    state['complete'] = True
    state['active_file'] = None
    save()
    subprocess.run([sys.executable, str(BASE / 'work/transfer_file.py'), str(STATE),
                    REMOTE + '/official-download-manifest.json'], cwd=BASE, check=True)
    print('OFFICIAL_MODEL_DOWNLOAD_AND_TRANSFER_COMPLETE_NOT_EVALUATED', flush=True)
except BaseException as exc:
    state['last_error'] = type(exc).__name__ + ': ' + str(exc)
    save()
    raise
