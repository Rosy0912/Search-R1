import pathlib,subprocess,shlex,json
B=pathlib.Path(__file__).resolve().parent.parent
code="""import pathlib,json,base64,hashlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');run=root/'runs/bapo-instruct300-formal300-3292';out={}
for p in [run/'FORMAL3292-RESUME25-INDEPENDENT-PERMISSION-REVIEW.md',run/'recovery-permission.json',run/'recovery-permission-check.json',root/'logs/formal3292-actual-permission-gate-execution.json',run/'phase.log']:
 if p.is_file():
  raw=p.read_bytes();out[str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(out))
"""
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(code)
r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','60',cmd],capture_output=True,timeout=75)
(B/'evidence/ray-envfix-formal3292-permission-outcome-capture.json').write_bytes(r.stdout);(B/'evidence/ray-envfix-formal3292-permission-outcome-capture.stderr').write_bytes(r.stderr);print(json.dumps(dict(exit_code=r.returncode,bytes=len(r.stdout),stderr=r.stderr.decode())));assert r.returncode==0 and not r.stderr
