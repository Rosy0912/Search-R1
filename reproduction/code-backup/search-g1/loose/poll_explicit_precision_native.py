"""Observe the existing immutable native smoke; never submit or restart it."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
import control

prep = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
p = argparse.ArgumentParser()
p.add_argument('--retry', action='store_true')
p.add_argument('--package', action='store_true')
args = p.parse_args()
assert not (args.retry and args.package)
stem = 'explicit-precision-native-package' if args.package else ('explicit-precision-native-retry' if args.retry else 'explicit-precision-native')
handle = json.loads((prep / (stem + '-launch.json')).read_text())
code = r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
h=json.load(sys.stdin);b=Path(h['base']);job=h['job_id']
r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
root=b/'logs/model-smoke-initial'; log=b/'logs'/('slurm-'+job+'.log')
files={p.name:dict(value=json.loads(p.read_text()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.glob('*.json')}
manifest=b/'logs/run-manifest.json'
source={n:hashlib.sha256((b/'repo'/n).read_bytes()).hexdigest() for n in h['source_sha256']}
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if ('SEARCH_G1_BASE='+str(b)).encode() in (p/'environ').read_bytes().split(b'\0'):
   stat=(p/'stat').read_text().rsplit(')',1)[1].split()
   owned.append(dict(pid=int(p.name),start_ticks=int(stat[19]),state=stat[0]))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print(json.dumps(dict(checked_unix=time.time(),job=r.stdout,scontrol_returncode=r.returncode,scontrol_stderr=r.stderr,
 files=files,source_sha256=source,source_matches=source==h['source_sha256'],
 manifest=json.loads(manifest.read_text()) if manifest.exists() else None,
 owned_processes=owned,log_tail=log.read_text(errors='replace')[-12000:] if log.exists() else '')))
'''
r = subprocess.run(control.route('h100-8', 'python3 -c ' + shlex.quote(code)),
                   input=json.dumps(handle), capture_output=True, text=True, timeout=55)
if r.returncode:
    raise RuntimeError(r.stderr[-2000:])
observed = json.loads(r.stdout)
(prep / (stem + '-observation.json')).write_text(json.dumps(observed, indent=2))
text = {k: observed[k] for k in ('checked_unix', 'job', 'source_matches', 'owned_processes')}
text['evidence_files'] = list(observed['files'])
text['log_tail'] = observed['log_tail'][-2200:]
print(json.dumps(text, indent=2))
