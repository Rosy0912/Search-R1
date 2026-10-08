"""Download immutable HF assets on the target; verify upstream LFS SHA256."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/search-r1')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def valid(path, a):
    if not path.exists() or (a['size'] and path.stat().st_size != a['size']):
        return False
    if a.get('sha256'):
        return digest(path) == a['sha256']
    h = hashlib.sha1()
    h.update(('blob '+str(path.stat().st_size)+'\0').encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest() == a['git_blob_sha1']

def fetch(a):
    dest = BASE/'assets'/a['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if valid(dest, a):
        print('EXISTS', a['path'], flush=True)
        return True
    prefix = '/datasets/' if a['kind'] == 'dataset' else '/'
    # Canonical metadata and hashes were fetched from huggingface.co on Mac.
    url = 'https://hf-mirror.com'+prefix+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
    temp = dest.with_name(dest.name+'.partial')
    if temp.exists() and a['size'] and temp.stat().st_size >= a['size']:
        if valid(temp, a):
            temp.replace(dest)
            return True
        rejected = temp.with_name(temp.name+'.rejected.'+str(time.time_ns()))
        temp.replace(rejected)
        aria_state = temp.with_name(temp.name+'.aria2')
        if aria_state.exists():
            aria_state.replace(rejected.with_name(rejected.name+'.aria2'))
        print('RETAINED_REJECTED_PARTIAL', rejected, flush=True)
    print('START', a['path'], a['size'], flush=True)
    if (a['size'] or 0) > 8*1024**3:
        command = ['aria2c','--continue=true','--file-allocation=none',
            '--max-connection-per-server=8','--split=8','--min-split-size=64M',
            '--connect-timeout=10','--timeout=60','--lowest-speed-limit=1024',
            '--max-tries=3','--retry-wait=3','--summary-interval=60',
            '--console-log-level=warn','--auto-file-renaming=false',
            '--dir='+str(temp.parent),'--out='+temp.name,url]
    else:
        command = ['curl','-f','-L','-sS','--connect-timeout','10',
            '--max-time','3600','--speed-limit','1024','--speed-time','60',
            '--retry','2','--retry-delay','3','-C','-','-o',str(temp),url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print('RELAY_NEEDED', a['path'], 'curl_exit', result.returncode, flush=True)
        return False
    if a['size'] and temp.stat().st_size != a['size']:
        print('SIZE_MISMATCH', a['path'], flush=True)
        return False
    if not valid(temp, a):
        print('SHA_MISMATCH', a['path'], flush=True)
        return False
    temp.replace(dest)
    print('VERIFIED', a['path'], dest.stat().st_size, flush=True)
    return True

if __name__ == '__main__':
    manifest = json.loads((BASE/'assets.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, manifest))
    (BASE/'logs/assets-status.json').write_text(json.dumps({
        'time':time.time(), 'complete':all(results),
        'failed':[a['path'] for a, ok in zip(manifest, results) if not ok]},indent=2))
    sys.exit(0 if all(results) else 1)
