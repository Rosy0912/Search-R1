"""Mac-only closed native USTAR to gzip; never a training dependency."""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def load_operation(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('Published source-operation SHA differs')
    ns = dict(__name__='_closed_raw_operation', __file__=str(path))
    exec(compile(raw, str(path), 'exec'), ns)
    return raw, ns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scope', 'scope-sha256', 'inventory', 'destination',
                'source-operation', 'operation-sha256', 'source-helper'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    operation, m = load_operation(a.source_operation, a.operation_sha256)
    b = m['primitives'](a.source_helper)
    scope_raw, local_scope = b['fileproof'](Path(a.scope), dict(path=str(Path(a.scope)),
                                            sha256=a.scope_sha256), retain=True)
    scope = json.loads(scope_raw)
    m['scope_shape'](scope)
    raw, inventory_ref = b['fileproof'](Path(a.inventory), retain=True)
    value = json.loads(raw)
    m['validate_inventory'](b, scope, a.scope_sha256, value)
    destination = Path(a.destination)
    m['require'](destination.is_absolute() and not destination.exists(), 'Destination must be new')
    b['checked_path'](str(destination.parent), directory=True)
    members = sum(512 + ((f['bytes'] + 511) // 512) * 512 for f in value['files'])
    tar_upper = ((members + 1024 + 10239) // 10240) * 10240
    m['require'](shutil.disk_usage(destination.parent).free > tar_upper + tar_upper // 100 + (5 << 30),
                 'Insufficient archive upper bound plus1percent and5GiB; backup backlog only')
    destination.mkdir(mode=0o700); b['fsync_dir'](destination.parent)
    ssh = ['/usr/bin/ssh', '-oBatchMode=yes', '-oConnectTimeout=10', 'hkust1']
    remote = shlex.join(['env', '-u', 'E1_RUN_ID', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
        '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python', '-B', '-c',
        operation.decode(), 'stream', '--scope-base64', base64.b64encode(scope_raw).decode('ascii'),
        '--scope-sha256', a.scope_sha256])
    partial = destination / 'closed-native512.tar.gz.partial'
    with partial.open('xb') as out, (destination / 'transfer.stderr.txt').open('xb') as err:
        child = subprocess.Popen(ssh + [remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err)
        b['write_json'](destination / 'transfer-launch.json', dict(pid=child.pid, parent_pid=os.getpid(),
            started_unix=time.time(), inventory_sha256=inventory_ref['sha256'], scope=local_scope,
            scope_transport='inline-base64', operation_sha256=a.operation_sha256, ssh=ssh,
            backup_is_training_dependency=False))
        error = None
        try:
            child.stdin.write(raw); child.stdin.close()
            with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=out, mtime=0) as gz:
                for block in iter(lambda: child.stdout.read(8 << 20), b''):
                    gz.write(block)
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            if not child.stdin.closed: child.stdin.close()
            child.stdout.close(); rc = child.wait(); out.flush(); os.fsync(out.fileno())
            b['write_json'](destination / 'transfer-terminal.json', dict(actual_returncode=rc,
                local_error=error, ended_unix=time.time(), partial_bytes=partial.stat().st_size,
                stderr_bytes=(destination / 'transfer.stderr.txt').stat().st_size,
                job_status_not_inferred=True))
    m['require'](rc == 0 and error is None, 'Actual transport failed; preserve partial')
    b['stable'](local_scope); b['stable'](inventory_ref)
    print(json.dumps(dict(actual_returncode=rc, archive_partial=str(partial))))


if __name__ == '__main__':
    main()
