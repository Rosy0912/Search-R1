"""One owned existing-run CPU runner deployment/readiness check, no automatic start."""
from pathlib import Path
import hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent;E=H.parent
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns);remote=ns['remote']
R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/formal-3210';D=R+'/evidence/first5-cpu-inspector-preparation';raw=(H/'inspect_first5_cpu.py').read_bytes();pin=hashlib.sha256(raw).hexdigest()
code='''import pathlib,hashlib,json,sys,os
p=pathlib.Path(DIR);assert p.parent.resolve(strict=True)==p.parent and os.getuid()==0
if not p.exists():p.mkdir()
f=p/'inspect_first5_cpu.py';data=sys.stdin.buffer.read();assert hashlib.sha256(data).hexdigest()==PIN
if f.exists():assert f.read_bytes()==data
else:
 with f.open('xb') as out:out.write(data)
assert f.resolve(strict=True)==f and hashlib.sha256(f.read_bytes()).hexdigest()==PIN
print(json.dumps({'path':str(f),'sha256':PIN,'uid':os.getuid()}))
'''.replace('DIR',repr(D)).replace('PIN',repr(pin))
p=remote('python3 -c '+shlex.quote(code),raw);(H/'deployment-original.json').write_bytes(p.stdout)
cmd='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python '+shlex.quote(D+'/inspect_first5_cpu.py')+' --status-only'
p=subprocess.run(ns['route'](cmd),capture_output=True,timeout=60);(H/'readiness-stdout-original.txt').write_bytes(p.stdout);(H/'readiness-stderr-original.txt').write_bytes(p.stderr);(H/'readiness-exit-original.json').write_text(json.dumps({'returncode':p.returncode})+'\n');print(p.stdout.decode());print('status_exit',p.returncode)
