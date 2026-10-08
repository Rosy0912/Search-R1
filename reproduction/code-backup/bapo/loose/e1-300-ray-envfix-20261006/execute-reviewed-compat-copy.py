import hashlib,pathlib,subprocess,json
B=pathlib.Path(__file__).resolve().parent.parent
D=B/'e1-300-ray-envfix-20261006'
script='''import pathlib,hashlib,sys,subprocess,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');p=root/'logs/create-cpu-log-path-compatibility.py';raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()=='1cdaebb49a3f69c01ca2913abc81600ae520d02bbbe98f4a8e861efabef30776';assert not p.exists()
with p.open('xb')as f:f.write(raw)
assert p.read_bytes()==raw
r=subprocess.run([str(root/'env/bin/python'),'-B',str(p),'--job-id','3287'],capture_output=True,text=True)
print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,tool_path=str(p),tool_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
raise SystemExit(r.returncode)
'''
import shlex
command='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(script)
result=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','60','--stdin-file',str(D/'create-cpu-log-path-compatibility.py'),command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=75)
(B/'evidence/ray-envfix-hardware3287-compat-copy-actual.stdout').write_bytes(result.stdout);(B/'evidence/ray-envfix-hardware3287-compat-copy-actual.stderr').write_bytes(result.stderr);print(json.dumps(dict(ssh_exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())))
assert result.returncode==0
