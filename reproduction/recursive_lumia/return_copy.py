"""Bounded resumable transport; never changes checkpoint content or submits work."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def digest(path):
    h = hashlib.sha256()
    if path.exists():
        with path.open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(b)
    return h


def copy_file(group, item, root, transport, update, stop, deadline,
              max_attempts=20, retry_delay=10, idle_seconds=600):
    root = Path(root)
    name = Path(item['path']).name
    dest, part = root / name, root / (name + '.part')
    key = group + '/' + name
    with (root / (name + '.transfer.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if dest.exists():
            if dest.is_symlink() or dest.stat().st_size != item['bytes'] or digest(dest).hexdigest() != item['sha256']:
                raise ValueError('Existing final file mismatch: ' + key)
        else:
            for attempt in range(1, max_attempts + 1):
                if stop.is_set() or time.monotonic() >= deadline:
                    raise RuntimeError('Transport stopped or deadline reached: ' + key)
                if part.is_symlink():
                    raise ValueError('Symlink partial: ' + key)
                count = part.stat().st_size if part.exists() else 0
                if count > item['bytes']:
                    raise ValueError('Oversized partial: ' + key)
                h = digest(part)
                if count == item['bytes']:
                    if h.hexdigest() != item['sha256']:
                        raise ValueError('Complete partial SHA mismatch: ' + key)
                    part.rename(dest)
                    break
                update(key, state='CONNECTING', attempt=attempt, bytes=count, total=item['bytes'], at=time.time())
                proc = subprocess.Popen(transport + ['read ' + group + ' ' + name + ' ' + str(count)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                done = threading.Event()
                progress = [time.monotonic()]

                def watchdog():
                    while not done.wait(1):
                        if stop.is_set() or time.monotonic() >= deadline or time.monotonic() - progress[0] > idle_seconds:
                            if proc.poll() is None:
                                proc.terminate()
                            return

                watcher = threading.Thread(target=watchdog, daemon=True)
                watcher.start()
                try:
                    last = 0
                    with part.open('ab') as f:
                        # read1 returns available bytes; slow streams do not wait to fill a4MiB block.
                        while True:
                            block = proc.stdout.read1(1024 * 1024)
                            if not block:
                                break
                            if count + len(block) > item['bytes']:
                                raise ValueError('Source exceeded expected size: ' + key)
                            f.write(block)
                            h.update(block)
                            count += len(block)
                            progress[0] = time.monotonic()
                            if time.time() - last > 10:
                                f.flush()
                                update(key, state='COPYING', attempt=attempt, bytes=count, total=item['bytes'], at=time.time())
                                last = time.time()
                        f.flush()
                        os.fsync(f.fileno())
                    err = proc.stderr.read()[-1000:].decode(errors='replace')
                    rc = proc.wait(timeout=15)
                    if count == item['bytes']:
                        if h.hexdigest() != item['sha256']:
                            raise ValueError('Complete file SHA mismatch: ' + key)
                        part.rename(dest)
                        break
                    update(key, state='CONNECTION_RETRY_PENDING', attempt=attempt, bytes=count, rc=rc, error=err, at=time.time())
                finally:
                    done.set()
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
                    proc.stdout.close()
                    proc.stderr.close()
                    watcher.join(timeout=2)
                if attempt == max_attempts:
                    raise RuntimeError('Connection retry budget exhausted: ' + key)
                stop.wait(min(60, retry_delay * attempt))
            else:
                raise RuntimeError('No completed transfer: ' + key)
        result = dict(state='ACTUAL_SHA_VERIFIED', group=group, name=name,
                      bytes=item['bytes'], sha256=item['sha256'], at=time.time())
        update(key, **result)
        return result


def main(directory):
    d = Path(directory)
    config = json.loads((d / 'copy-config.json').read_text())
    transport = [x.replace('ConnectTimeout=15', 'ConnectTimeout=90')
                 .replace('ServerAliveInterval=20', 'ServerAliveInterval=60')
                 .replace('ServerAliveCountMax=3', 'ServerAliveCountMax=10') for x in config['transport']]
    state = dict(started=time.time(), pid=os.getpid(), state='COPYING', files={},
                 training_submitted=False, parallel_streams=4, max_attempts_per_file=20,
                 deadline_hours=16, protocol='20261017-inspected-network-repair-v1')
    mutex, stop = threading.Lock(), threading.Event()

    def save():
        tmp = d / 'copy-status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(d / 'copy-status.json')

    def update(key, **values):
        with mutex:
            state['files'][key] = values
            save()

    with (d / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        deadline = time.monotonic() + 16 * 3600
        try:
            rows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = []
                for group in ('generation-2', 'budget-5'):
                    for item in config['files'][group]:
                        futures.append(pool.submit(copy_file, group, item, config['targets'][group],
                                                   transport, update, stop, deadline))
                try:
                    for future in concurrent.futures.as_completed(futures):
                        rows.append(future.result())
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            for group in config['files']:
                receipt = Path(config['targets'][group]) / 'direct-backup-receipt.json'
                with receipt.open('x') as f:
                    json.dump(dict(at=time.time(), status='ALL_NINE_ACTUAL_SHA_VERIFIED', source_kept=True,
                                   files=[x for x in rows if x['group'] == group]), f, indent=2)
            state.update(state='ALL_18_FILES_ACTUAL_SHA_VERIFIED', finished=time.time())
        except BaseException as exc:
            state.update(state='FAILED_PRESERVE_PARTIAL_NO_AUTO_RETRY', error=repr(exc))
            raise
        finally:
            save()


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
