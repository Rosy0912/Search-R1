# Independent Mac transfer. No GPU/model/HTTP/Node writes or training dependency.
import argparse, gzip, hashlib, json, os, pathlib, shlex, shutil, subprocess, time

p = argparse.ArgumentParser()
for k in ('inventory', 'destination', 'source-operation', 'operation-sha256'):
    p.add_argument('--' + k, required=True)
a = p.parse_args()
D = pathlib.Path(a.destination)
assert D.is_dir() and not D.is_symlink()
raw = pathlib.Path(a.inventory).read_bytes()
x = json.loads(raw)
assert x['schema'] == 'c1-closed5-9-raw-fixed-inventory-v1' and x['closed_outers'] == [5,6,7,8,9]
members_upper=sum(512+((f['bytes']+511)//512)*512 for f in x['files'])
tar_upper=((members_upper+1024+10239)//10240)*10240
assert shutil.disk_usage(D).free > tar_upper + tar_upper//100 + (5 << 30)
operation = pathlib.Path(a.source_operation).read_bytes()
assert hashlib.sha256(operation).hexdigest() == a.operation_sha256
ssh = ['/usr/bin/ssh', '-B', 'en0', '-oProxyCommand=none', '-oProxyJump=none', '-oControlMaster=no', '-S', 'none', '-oBatchMode=yes', '-oConnectTimeout=10', 'hkust1']
remote = shlex.join(['env', '-u', 'E1_RUN_ID', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', '/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python', '-B', '-c', operation.decode(), 'stream'])
partial = D / 'closed-raw.tar.gz.partial'
def save(name, value):
    with (D / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
with partial.open('xb') as out, (D / 'transfer.stderr.txt').open('xb') as err, pathlib.Path(a.inventory).open('rb') as stdin:
    child = subprocess.Popen(ssh + [remote], stdin=stdin, stdout=subprocess.PIPE, stderr=err)
    save('transfer-launch.json', dict(pid=child.pid, parent_pid=os.getpid(), started_unix=time.time(), inventory_sha256=hashlib.sha256(raw).hexdigest(), operation_sha256=a.operation_sha256, ssh=ssh, backup_is_training_dependency=False))
    print(json.dumps(dict(actual_started=True, pid=child.pid, destination=str(D))), flush=True)
    error = None
    try:
        with gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=out, mtime=0) as gz:
            for block in iter(lambda: child.stdout.read(8 << 20), b''):
                gz.write(block)
    except BaseException as e:
        error = type(e).__name__ + ': ' + str(e)
        raise
    finally:
        child.stdout.close()
        rc = child.wait()
        out.flush()
        os.fsync(out.fileno())
        save('transfer-terminal.json', dict(actual_returncode=rc, local_error=error, ended_unix=time.time(), partial_bytes=partial.stat().st_size, stderr_bytes=(D / 'transfer.stderr.txt').stat().st_size, job_status_not_inferred=True))
assert rc == 0 and error is None
print(json.dumps(dict(actual_returncode=rc, archive_partial=str(partial))), flush=True)
