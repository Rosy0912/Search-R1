"""Reuse immutable same-node source; upload only new code/audit metadata."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

WORK = Path(__file__).resolve().parent
sys.path.insert(0,str(WORK))
from remote import route


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--build-sha',required=True)
    args = parser.parse_args()
    build_bytes = args.build.read_bytes()
    assert hashlib.sha256(build_bytes).hexdigest() == args.build_sha
    build = json.loads(build_bytes)
    new = Path(build['local_package'])
    assert new.parent.resolve() == WORK and re.fullmatch('single-tp1-window-[0-9a-f]{8}-[0-9a-f]{8}',new.name)
    old = WORK/'probability-probe-49a6b35-29822f9-r3'
    prior = json.loads((old/'package-manifest.json').read_bytes())
    manifest = json.loads((new/'package-manifest.json').read_bytes())
    assert sha(new/'package-manifest.json') == build['manifest_sha256']
    copied, changed = [], {}
    files = dict(manifest['files'])
    files['package-manifest.json'] = build['manifest_sha256']
    for name,expected in files.items():
        path = new/name
        assert path.resolve() == path and path.is_file()
        source_bytes = path.read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == expected
        assert path.suffix not in {'.pt','.safetensors','.parquet','.bin','.so'}
        if prior['files'].get(name) == expected:
            copied.append(name)
        else:
            assert path.suffix in {'.py','.sh','.json','.md','.txt','.diff','.yaml','.yml'}, 'New source metadata only'
            changed[name] = base64.b64encode(source_bytes).decode()
    remote_base = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')
    assert build['remote_package'] == str(remote_base/new.name)
    packet = dict(source=str(remote_base/old.name),target=build['remote_package'],
                  source_manifest_sha256=sha(old/'package-manifest.json'),
                  manifest_sha256=build['manifest_sha256'],copy_files=copied,changed=changed)
    payload = gzip.compress(json.dumps(packet,sort_keys=True).encode())
    journal = WORK/'tree-vllm-window-boundaries3295/SOURCE_UPLOAD_PREPARED.json'
    assert not journal.exists(), 'No implicit upload retry'
    journal.write_text(json.dumps(dict(status='SINGLE_WINDOW_CODE_UPLOAD_ATTEMPT_PREPARED_NO_GPU',
        build_sha256=args.build_sha,packet_sha256=hashlib.sha256(payload).hexdigest(),
        copied_same_node_files=len(copied),changed_code_metadata_files=len(changed),compressed_bytes=len(payload),
        tensor_assets_transferred=False,gpu_submitted=False),indent=2)+'\n')
    remote = r'''
import base64,gzip,hashlib,json,os,re,shutil,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
p=json.loads(gzip.decompress(base64.b64decode(sys.stdin.read())))
base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates')
source=Path(p['source']);target=Path(p['target'])
assert source==base/'probability-probe-49a6b35-29822f9-r3'
assert target.parent==base and re.fullmatch('single-tp1-window-[0-9a-f]{8}-[0-9a-f]{8}',target.name)
assert hashlib.sha256((source/'package-manifest.json').read_bytes()).hexdigest()==p['source_manifest_sha256']
prior=json.loads((source/'package-manifest.json').read_bytes())
assert not target.exists(),'Immutable source destination already exists; reconcile without replacing'
target.mkdir()
for name in p['copy_files']:
 n=Path(name);assert not n.is_absolute() and '..' not in n.parts
 s=source/n;d=target/n
 assert s.resolve()==s and s.is_file() and hashlib.sha256(s.read_bytes()).hexdigest()==prior['files'][name]
 d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d)
for name,value in p['changed'].items():
 n=Path(name);assert not n.is_absolute() and '..' not in n.parts
 assert n.suffix in {'.py','.sh','.json','.md','.txt','.diff','.yaml','.yml'}
 d=target/n;d.parent.mkdir(parents=True,exist_ok=True)
 with d.open('xb') as stream:
  stream.write(base64.b64decode(value));stream.flush();os.fsync(stream.fileno())
b=(target/'package-manifest.json').read_bytes()
assert hashlib.sha256(b).hexdigest()==p['manifest_sha256']
m=json.loads(b)
actual={f.relative_to(target).as_posix() for f in target.rglob('*') if f.is_file() and f.name!='package-manifest.json'}
assert actual==set(m['files'])
for name,wanted in m['files'].items():
 f=target/name;assert f.resolve()==f and hashlib.sha256(f.read_bytes()).hexdigest()==wanted
print(json.dumps(dict(status='SINGLE_WINDOW_SOURCE_BYTES_VERIFIED_NO_DISPATCH_OR_GPU_ACCEPTANCE',
 remote_package=str(target),manifest_sha256=p['manifest_sha256'],
 copied_same_node_files=len(p['copy_files']),changed_code_metadata_files=len(p['changed']),
 tensor_assets_transferred=False,gpu_submitted=False),sort_keys=True))
'''
    command = 'env PYTHONDONTWRITEBYTECODE=1 /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -B -c '+shlex.quote(remote)
    answer = subprocess.run(route(command),input=base64.b64encode(payload),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    sys.stdout.buffer.write(answer.stdout)
    sys.stderr.buffer.write(answer.stderr)
    sys.exit(answer.returncode)


if __name__ == '__main__':
    main()
