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
from concurrent.futures import ThreadPoolExecutor

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
state['download_concurrency'] = 3
state.pop('last_error', None)

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

def download(name):
    item = entries[name]
    destination = STAGING / name
    existing_bytes = destination.stat().st_size if destination.exists() else 0
    if shutil.disk_usage(STAGING).free < int(item['size']) - existing_bytes + 2 * 1024**3:
        raise RuntimeError('Insufficient staging free space; preserve existing files')
    expected = (item.get('lfs') or {}).get('sha256')
    if existing_bytes != item['size']:
        print('DOWNLOAD_OFFICIAL', name, item['size'], flush=True)
        url = f'https://huggingface.co/Qwen/Qwen2.5-32B/resolve/{REVISION}/{name}'
        for attempt in range(32):
            # A new curl invocation reads the current partial size. Internal
            # retries reset to its original offset and discard new progress.
            result = subprocess.run(['curl', '--socks5-hostname', '127.0.0.1:7892', '--http1.1', '--silent', '--show-error',
                        '--location', '--fail', '--connect-timeout', '15', '--retry', '0',
                        '--speed-time', '60', '--speed-limit', '8192', '--continue-at', '-',
                        '--output', str(destination), url])
            if result.returncode == 0:
                break
            print('RESUME_FROM_CURRENT_PARTIAL', name, attempt, destination.stat().st_size if destination.exists() else 0, result.returncode, flush=True)
            time.sleep(3)
        else:
            raise RuntimeError('Official download retries exhausted, partial retained: ' + name)
    assert destination.stat().st_size == item['size'], name
    actual = digest(destination)
    assert expected is None or actual == expected, f'Official source SHA mismatch: {name}'
    print('OFFICIAL_FILE_VERIFIED', name, actual, flush=True)
    return actual

try:
    pending = [name for name in names if not state['files'].get(name, {}).get('remote_sha_verified')]
    # At most three files, each under4GB. Reserve their remaining full sizes
    # before launching; transfers remain serial and completed relays are removed.
    needed = sum(max(0, entries[name]['size'] - ((STAGING/name).stat().st_size if (STAGING/name).exists() else 0))
                 for name in pending[:3])
    assert shutil.disk_usage(STAGING).free >= needed + 2 * 1024**3
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for index, name in enumerate(pending):
            for queued in pending[index:index+3]:
                if queued not in futures:
                    futures[queued] = pool.submit(download, queued)
            state['active_file'] = name
            state['inflight_files'] = list(futures)
            save()
            actual = futures.pop(name).result()
            destination = STAGING / name
            subprocess.run([sys.executable, str(BASE / 'work/transfer_file.py'), str(destination),
                            REMOTE + '/' + name], cwd=BASE, check=True)
            state['files'][name] = {'size': entries[name]['size'], 'sha256': actual,
                                  'official_lfs_sha256': (entries[name].get('lfs') or {}).get('sha256'),
                                  'remote_sha_verified': True}
            save()
            destination.unlink()
            print('VERIFIED_RELAY_FILE_REMOVED', name, flush=True)
    state['complete'] = True
    state['active_file'] = None
    state['inflight_files'] = []
    save()
    subprocess.run([sys.executable, str(BASE / 'work/transfer_file.py'), str(STATE),
                    REMOTE + '/official-download-manifest.json'], cwd=BASE, check=True)
    print('OFFICIAL_MODEL_DOWNLOAD_AND_TRANSFER_COMPLETE_NOT_EVALUATED', flush=True)
except BaseException as exc:
    state['last_error'] = type(exc).__name__ + ': ' + str(exc)
    save()
    raise
