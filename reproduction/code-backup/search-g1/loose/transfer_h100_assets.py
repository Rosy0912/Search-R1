"""Same-H100-cluster asset relay; preserve verified staging until target passes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

root = Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/g1-common8-h100-assets')
source = '/dataset/shaoshitong/search-r1-rl/tree-grpo'
target = '/dataset/shaoshitong/Search-G1-common8'
status = Path('/Users/shaoshitong/Documents/Search-G1/preparation/assets-transfer-status.json')
paths = ['wheelhouse', 'instruct300/assets/qwen2.5-3b-instruct-aa8e725',
    'env/lib/python3.10/site-packages/flash_attn',
    'env/lib/python3.10/site-packages/flash_attn-2.7.4.post1.dist-info',
    'env/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so']


def publish(stage, **fields):
    value = dict(stage=stage, unix=time.time(), pid=os.getpid(),
        source_host='h100-9', target_host='h100-8', **fields)
    tmp = status.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(status)
    print(stage, flush=True)


def remote(host, command, **kwargs):
    return subprocess.run(control.route(host, command), check=True, **kwargs)


try:
    publish('READ_SOURCE_HASHES')
    py = 'from pathlib import Path; import hashlib,json; root=Path('+repr(source)+'); names='+repr(paths)+'; files=sorted({p for name in names for p in ([root/name] if (root/name).is_file() else (root/name).rglob("*")) if p.is_file()}); print(json.dumps({str(p.relative_to(root)):hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in files}))'
    # Source system Python3.10 lacks file_digest: use streaming explicitly.
    py = py.replace('hashlib.file_digest(p.open("rb"),"sha256").hexdigest()', 'hashlib.sha256(p.read_bytes()).hexdigest()')
    hashes = json.loads(remote('h100-9', 'python3 -c '+shlex.quote(py),
        capture_output=True, text=True, timeout=240).stdout)
    (root/'source-hashes.json').write_text(json.dumps(hashes, indent=2))
    publish('DOWNLOADING_SAME_CLUSTER_BUNDLE', file_count=len(hashes))
    archive = root/'assets.tar'
    with archive.open('wb') as f:
        remote('h100-9', 'tar -C '+shlex.quote(source)+' -cf - '+' '.join(map(shlex.quote, paths)),
            stdout=f, timeout=7200)
    publish('UPLOADING_TARGET', archive_bytes=archive.stat().st_size)
    with archive.open('rb') as f:
        remote('h100-8', 'umask 077; mkdir -p '+target+'/bootstrap-assets; tar -xf - -C '+target+'/bootstrap-assets',
            stdin=f, timeout=7200)
    publish('VERIFYING_TARGET')
    py = 'from pathlib import Path; import hashlib,json; root=Path('+repr(target+'/bootstrap-assets')+'); expected='+repr(hashes)+'; actual={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in expected}; assert actual==expected; print(json.dumps(dict(accepted=True,file_count=len(actual))))'
    proof = json.loads(remote('h100-8', 'python3 -c '+shlex.quote(py),
        capture_output=True, text=True, timeout=240).stdout)
    (root/'target-proof.json').write_text(json.dumps(proof, indent=2))
    archive.unlink()  # Only this transfer's large staging, after all hashes pass.
    publish('COMPLETE', **proof)
except Exception as exc:
    publish('FAILED_RETAIN_STAGING', error=type(exc).__name__+': '+str(exc))
    raise
