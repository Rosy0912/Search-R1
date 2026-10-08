import pathlib,subprocess,shlex,json
D=pathlib.Path(__file__).resolve().parent;B=D.parent
code="""import pathlib,json,hashlib,base64
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');out={}
for name in ['hardware-acceptance.json','HARDWARE3287-FINAL-INDEPENDENT-REVIEW.md','hardware3287-final-frozen-gate-execution.json']:
 p=r/'logs'/name;raw=p.read_bytes();out[str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode())
print(json.dumps(out))
"""
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(code)
r=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','60',cmd],capture_output=True,timeout=75)
(B/'evidence/ray-envfix-hardware3287-published-gate-original-capture.json').write_bytes(r.stdout);(B/'evidence/ray-envfix-hardware3287-published-gate-original-capture.stderr').write_bytes(r.stderr);assert r.returncode==0 and not r.stderr;print('Original published report/receipt/gate captured',len(r.stdout))
