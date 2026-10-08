import pathlib,subprocess,shlex,json
D=pathlib.Path(__file__).resolve().parent;B=D.parent
code="""import pathlib,hashlib,sys,subprocess,json,os
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()=='82ccb52ae94b2e05f4fd9dbafb66443631ca0ff77eb0978fe481ee82b082834b';live=subprocess.check_output(['scontrol','show','job','-dd','3292'],text=True,timeout=15)
if 'JobState=RUNNING'not in live or not(r/'runs/bapo-instruct300-formal300-3292/WAITING_FOR_INDEPENDENT_RECOVERY_PERMISSION').is_file():
 print(json.dumps(dict(status='WAIT_SAME_JOB_ONLY',job_id=3292,slurm=live)));raise SystemExit(0)
p=r/'logs/capture-formal3292-allocation.py';assert not p.exists()and not p.is_symlink()
with p.open('xb')as f:f.write(raw)
assert p.read_bytes()==raw
res=subprocess.run([str(r/'env/bin/python'),'-B',str(p),'--job-id','3292','--hardware-receipt-sha','e8625dabce8c9cf5a97498cb0989b399937fd9719acfa186357b1d29cebe1e2c'],cwd=r/'repo',env=dict(os.environ,PYTHONPATH=str(r/'repo'),PYTHONDONTWRITEBYTECODE='1'),capture_output=True)
sys.stdout.buffer.write(res.stdout);sys.stderr.buffer.write(res.stderr);raise SystemExit(res.returncode)
"""
cmd='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/env/bin/python -B -c '+shlex.quote(code)
p=subprocess.run(['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B',str(B/'h100_exec.py'),'--socks','--timeout','60','--stdin-file',str(D/'capture-formal-allocation.py'),cmd],capture_output=True,timeout=75)
(B/'evidence/ray-envfix-formal3292-allocation-transfer.json').write_bytes(p.stdout);(B/'evidence/ray-envfix-formal3292-allocation-transfer.stderr').write_bytes(p.stderr);print(json.dumps(dict(ssh_exit_code=p.returncode,stdout_bytes=len(p.stdout),stderr=p.stderr.decode())));assert p.returncode==0 and not p.stderr
