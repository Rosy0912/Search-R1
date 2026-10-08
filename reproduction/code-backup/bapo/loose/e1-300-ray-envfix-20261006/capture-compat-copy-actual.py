import subprocess,pathlib,shlex,json
b=pathlib.Path(__file__).resolve().parent.parent
code="""import pathlib,json,hashlib,base64
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');names=['cpu-gate.log','cpu-gate-log-compatibility-copy.json','create-cpu-log-path-compatibility.py','cpu-gate-durable.log','cpu-gate-durable-exit.json','launch-approval.json','tools-manifest.json','../runs/bapo-instruct300-hardware-3287/slurm-terminal.txt','../runs/bapo-instruct300-hardware-3287/actual-fullhash-inventory.json'];out={}
for name in names:
 p=(r/'logs'/name).resolve()
 if not p.exists():continue
 raw=p.read_bytes();out[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
prov=json.loads((r/'logs/cpu-gate-log-compatibility-copy.json').read_text())
for row in prov.values():
 if type(row)is dict and 'path'in row:
  p=pathlib.Path(row['path'])
  if p.is_file():
   raw=p.read_bytes();out[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps(out))
"""
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(code)
r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(b/'h100_exec.py'),'--socks','--timeout','60',cmd],capture_output=True,timeout=75)
(b/'evidence/ray-envfix-hardware3287-compat-copy-actual-capture.json').write_bytes(r.stdout);(b/'evidence/ray-envfix-hardware3287-compat-copy-actual-capture.stderr').write_bytes(r.stderr);assert r.returncode==0 and not r.stderr;print('Actual capture saved',len(r.stdout))
