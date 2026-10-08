#!/usr/bin/env python3
"""Download each published branch and verify the committed source manifest."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

def run(args, cwd=None):
    p = subprocess.run(args, cwd=cwd, capture_output=True, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15'))
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])
    return p.stdout

class Objects:
    def __init__(self, repo):
        self.p = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def read(self, name):
        self.p.stdin.write((name+'\n').encode()); self.p.stdin.flush()
        header = self.p.stdout.readline().split()
        if len(header)!=3 or header[1]!=b'blob': raise RuntimeError('Missing committed blob: '+name)
        raw = self.p.stdout.read(int(header[2])); self.p.stdout.read(1)
        return raw
    def close(self):
        self.p.stdin.close(); self.p.wait()

def verify(result):
    if result['status'] not in {'pushed_remote_ref_verified', 'remote_download_all_files_verified'}:
        raise RuntimeError('Branch is not yet confirmed pushed: '+result['label'])
    label = result['label']; dest = ROOT/'remote-readback'/label
    if not dest.exists():
        run(['git','clone','--quiet','--depth','1','--single-branch','--branch',result['branch'],'--no-checkout','git@github.com:shaoshitong/Search-R1.git',str(dest)])
    sha = run(['git','-C',str(dest),'rev-parse','HEAD']).decode().strip()
    if sha != result['commit']:
        run(['git','-C',str(dest),'fetch','--quiet','--depth','1','origin',result['branch']])
        sha = run(['git','-C',str(dest),'rev-parse','FETCH_HEAD']).decode().strip()
    if sha != result['commit']: raise RuntimeError('Downloaded branch SHA differs from source backup')
    objects = Objects(dest)
    raw = objects.read(sha+':reproduction/code-backup/'+label+'/FILES-SHA256.json')
    manifest = json.loads(raw)
    for f in manifest['files']:
        dat = objects.read(sha+':'+f['destination'])
        if len(dat)!=f['bytes'] or hashlib.sha256(dat).hexdigest()!=f['sha256']:
            raise RuntimeError('Remote downloaded blob hash mismatch: '+f['destination'])
    objects.close()
    source_tree = run(['git','-C',result['repo'],'rev-parse',sha+'^{tree}']).decode().strip()
    remote_tree = run(['git','-C',str(dest),'rev-parse',sha+'^{tree}']).decode().strip()
    if source_tree != remote_tree: raise RuntimeError('Remote full source tree identity differs')
    result.update(status='remote_download_all_files_verified', remote_readback_repo=str(dest), remote_readback_sha256_count=len(manifest['files']), remote_full_tree_sha=remote_tree, manifest_sha256=hashlib.sha256(raw).hexdigest(), verified_utc=datetime.now(timezone.utc).isoformat())
    (ROOT/(label+'-result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result

if __name__ == '__main__':
    results = [json.loads(p.read_text()) for p in sorted(ROOT.glob('*-result.json'))]
    results = [r for r in results if r['status'] in {'pushed_remote_ref_verified', 'remote_download_all_files_verified'}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(verify,results))
    (ROOT/'REMOTE-VERIFICATION.json').write_text(json.dumps(dict(verified_utc=datetime.now(timezone.utc).isoformat(),count=len(rows),experiments=rows),ensure_ascii=False,indent=2)+'\n')
