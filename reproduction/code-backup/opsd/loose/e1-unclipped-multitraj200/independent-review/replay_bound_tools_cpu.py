"""Local CPU fixtures against exact bound modules. No production entrypoints."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

H = Path(__file__).resolve().parents[1]
R = H.parent / 'opsd-unclipped-multitraj200'
B = H / 'runtime-tools-bound-667c68e0f91b.json'
spec = json.loads(B.read_bytes())
out = H / 'independent-review/bound-tools-independent-cpu'
out.mkdir()
paths = [B]
for role in ('formal', 'native', 'first5'):
    row = spec[role]
    paths += [Path(row['local_directory']) / name for name in (*row['files'], row['manifest_name'])]
hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before = hashes()
program = r'''
import importlib.util,json,pathlib,sys
h=pathlib.Path(sys.argv[1]);b=json.loads((h/'runtime-tools-bound-667c68e0f91b.json').read_bytes())
repo=h.parent/'opsd-unclipped-multitraj200';sys.path.insert(0,str(repo))
for role in ('native','first5','formal'):sys.path.insert(0,b[role]['local_directory'])
for name,role in [('formal_supervisor','formal'),('resource_sample','formal'),('owned_runtime','formal'),('receipt_payload','first5'),('hf_cpu_reload','first5')]:
    p=pathlib.Path(b[role]['local_directory'])/(name+'.py');m=importlib.util.module_from_spec(importlib.util.spec_from_file_location(name,p));sys.modules[name]=m
    exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),m.__dict__)
    assert pathlib.Path(m.__file__)==p
import pytest
raise SystemExit(pytest.main(['-q',str(h/'formal-tools-candidate/test_formal_tools.py'),str(h/'formal-tools-candidate/test_resource_exit_races.py'),str(h/'first5-receipt-candidate/test_first5_tools.py'),str(pathlib.Path(b['native']['local_directory'])/'test_inspector.py')]))
'''
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
command = [sys.executable, '-B', '-c', program, str(H)]
started = datetime.now(timezone.utc).isoformat()
with (out/'stdout.original').open('xb') as stdout, (out/'stderr.original').open('xb') as stderr:
    process = subprocess.run(command, cwd=R, env=env, stdout=stdout, stderr=stderr)
after = hashes()
record = {'version':1,'scope':'Local CPU fixtures execute exact source-bound modules; no actual native GPU/first5/job receipt/signature/deployment',
          'utc_started':started,'utc_finished':datetime.now(timezone.utc).isoformat(),
          'command':command,'cwd':str(R),'returncode':process.returncode,'before_sha256':before,'after_sha256':after,
          'all_bundle_bytes_stable':before==after,'runtime_binding_sha256':hashlib.sha256(B.read_bytes()).hexdigest()}
with (out/'execution.original.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'returncode':process.returncode,'stable':before==after,'output':str(out)}))
print((out/'stdout.original').read_text())
print((out/'stderr.original').read_text())
if process.returncode or before!=after:raise SystemExit(1)
