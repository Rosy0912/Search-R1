"""Single-owner remote HKUST3 -> HPC3 archive relay; no Mac data or credentials."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REMOTE = '/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/migration-flash-build-20261009'
SSH = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
       '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
       '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
       '-oConnectTimeout=8', '-oServerAliveInterval=15', '-oServerAliveCountMax=3',
       'jxu957@10.120.81.30']
SHA = '46367eb890d271a68b50520127f23b3cdeee690991e5bb27ac07a9ef595d7ffe'
SIZE = 6930216


def save(name, data):
    p = ROOT / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(t, p)


def receiver(action, timeout=60):
    command = 'python3 ' + shlex.quote(REMOTE + '/cosearch_flash_hpc_receiver.py') + ' ' + action
    r = subprocess.run(SSH + [command], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        if 'AssertionError' in r.stderr:
            raise AssertionError('HPC receiver integrity rejection')
        raise RuntimeError('HPC receiver exit ' + str(r.returncode))
    return json.loads(r.stdout)


def main():
    lock = (ROOT / 'hpc-forward.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save('hpc-forward-controller.json', {'pid': os.getpid(), 'epoch': time.time(),
         'route': 'HKUST3 -> HPC3', 'mac_data_relay': False})
    receipt = ROOT / 'forward-verified.json'
    while not receipt.exists():
        save('hpc-forward-progress.json', {'status': 'WAITING_HKUST3_ARCHIVE_SHA', 'epoch': time.time()})
        time.sleep(15)
    accepted = json.loads(receipt.read_text())
    assert accepted['status'] == 'HKUST2_HKUST3_ARCHIVE_SHA256_VERIFIED'
    assert accepted['sha256'] == SHA and accepted['bytes'] == SIZE
    source = ROOT / 'flash-attn-281-build-source.tar.zst'
    h = hashlib.sha256()
    with source.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    assert source.stat().st_size == SIZE and h.hexdigest() == SHA
    attempt = 0
    while True:
        attempt += 1
        try:
            state = receiver('state')
            if not state['final_exists']:
                offset = state['offset']
                assert 0 <= offset <= SIZE
                save('hpc-forward-progress.json', {'status': 'FORWARDING', 'epoch': time.time(),
                     'resumed_from': offset, 'attempt': attempt})
                command = 'python3 ' + shlex.quote(REMOTE + '/cosearch_flash_hpc_receiver.py') + ' receive ' + str(offset)
                with source.open('rb') as f, (ROOT / 'hpc-forward-transport.log').open('ab') as log:
                    f.seek(offset)
                    child = subprocess.Popen(SSH + [command], stdin=subprocess.PIPE,
                                             stdout=log, stderr=log)
                    try:
                        while True:
                            b = f.read(1024 * 1024)
                            if not b:
                                break
                            child.stdin.write(b)
                        child.stdin.close()
                        if child.wait(timeout=90) != 0:
                            raise RuntimeError('Receiver failed; preserve partial')
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                            child.wait(timeout=15)
                        raise
            save('hpc-forward-progress.json', {'status': 'VERIFYING_ARCHIVE_AND_ORIGINALS',
                 'epoch': time.time(), 'attempt': attempt})
            result = receiver('verify-extract', timeout=3600)
            assert result['status'] == 'SUPPLEMENT_ALL_ORIGINAL_FILES_FRESH_SHA_VERIFIED'
            save('hpc-target-fresh-verified.json', result)
            return
        except AssertionError:
            save('hpc-forward-integrity-failure.json', {'epoch': time.time(), 'status': 'INTEGRITY_HARDSTOP'})
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            save('hpc-forward-failure.json', {'epoch': time.time(), 'attempt': attempt,
                 'error_type': type(error).__name__})
            time.sleep(min(120, 15 * attempt))


if __name__ == '__main__':
    main()
