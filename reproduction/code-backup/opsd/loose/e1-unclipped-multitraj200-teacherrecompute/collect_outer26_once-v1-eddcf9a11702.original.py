"""Root's single read-only transfer; exact reviewed remote bytes, no deployment."""
import base64
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

H = Path(__file__).resolve().parent
C = H / 'paired-outer26-candidate'
PINS = {
    'collect_outer26.py': 'b097bc410bd6be4cfe5f4203feef8f1cd5bfe233977f54aa1e267597a99906f5',
    'AUTHORITY-v2.original.json': 'b64e39864be651423dee39b423191a3d87a6c401d496356330c37d186e6b0fd9',
}
REVIEW = '18eaf20db1432f4e55504d8393cbb042f0ad529272af873f202cfd498fc6f45a'
TRANSPORT = 'a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pin(path, pin):
    assert path.resolve(strict=True) == path
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        raw = f.read()
        after = os.fstat(f.fileno())
    attrs = ('st_dev', 'st_ino', 'st_uid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    sig = lambda value: tuple(getattr(value, key) for key in attrs)
    assert sig(before) == sig(after) == sig(path.stat()) and sha(raw) == pin
    return raw


def save(path, raw):
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def main():
    source = read_pin(C / 'collect_outer26.py', PINS['collect_outer26.py'])
    authority = read_pin(C / 'AUTHORITY-v2.original.json', PINS['AUTHORITY-v2.original.json'])
    read_pin(H / 'OBSERVED_OUTER26_COLLECTION_ANALYSIS_INDEPENDENT_SOFTWARE_REVIEW.json', REVIEW)
    transport = read_pin(H / 'transport.py', TRANSPORT)
    helper = importlib.util.module_from_spec(importlib.util.spec_from_loader('own_h100_transport_bytes', loader=None))
    helper.__file__ = str(H / 'transport.py')
    exec(compile(transport, helper.__file__, 'exec'), helper.__dict__)
    remote = ('import base64,hashlib;source=base64.b64decode(' + repr(base64.b64encode(source).decode()) +
              ');assert hashlib.sha256(source).hexdigest()==' + repr(PINS['collect_outer26.py']) +
              ';exec(compile(source,"stdin-readonly-outer26","exec"))')
    argv = helper.route('python3 -B -c ' + shlex.quote(remote))
    directory = H / 'outer26-collection-once-v1-original'
    directory.mkdir(exist_ok=False)
    for path in (directory.parent, directory):
        fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    save(directory / 'caller-source.original.py', Path(__file__).read_bytes())
    save(directory / 'collector-source.original.py', source)
    save(directory / 'authority.original.json', authority)
    save(directory / 'transport-source.original.py', transport)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    intent = dict(kind='ACTUAL_READ_ONLY_OUTER26_COLLECTION_INTENT', utc=start, argv=argv,
                  runtime_sha256=sha(source), authority_sha256=sha(authority), transport_sha256=sha(transport),
                  remote_host='h100-10', scope='STDOUT_COPY_ONLY_NO_REMOTE_WRITES_GPU_OR_JOB')
    save(directory / 'command.original.json', (json.dumps(intent, sort_keys=True) + '\n').encode())
    waited, returncode, local_pid = False, None, None
    with (directory / 'stdout.original.gz').open('xb') as out, (directory / 'stderr.original').open('xb') as err:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=out, stderr=err)
        local_pid = process.pid
        process.stdin.write(authority)
        process.stdin.close()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                returncode = process.wait(timeout=min(30, max(.001, deadline - time.monotonic())))
                waited = True
                break
            except subprocess.TimeoutExpired:
                pass
        out.flush()
        err.flush()
        os.fsync(out.fileno())
        os.fsync(err.fileno())
    digest = hashlib.sha256()
    with (directory / 'stdout.original.gz').open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            digest.update(block)
    result = dict(kind='ACTUAL_READ_ONLY_OUTER26_COLLECTION_WAIT', waited=waited, returncode=returncode,
                  runtime_sha256=sha(source), authority_sha256=sha(authority), transport_sha256=sha(transport),
                  caller_sha256=sha(Path(__file__).read_bytes()), remote_host='h100-10', local_ssh_pid=local_pid,
                  start_utc=start, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  compressed_stdout_sha256=digest.hexdigest(), compressed_bytes=(directory / 'stdout.original.gz').stat().st_size,
                  unknown_policy='Preserve originals; never retry same once or infer remote success')
    save(directory / 'wait.original.json', (json.dumps(result, sort_keys=True) + '\n').encode())
    print(json.dumps(dict(directory=str(directory), waited=waited, returncode=returncode,
                          compressed_bytes=result['compressed_bytes'], wait_sha256=sha((directory / 'wait.original.json').read_bytes()))))
    if not waited or returncode != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
