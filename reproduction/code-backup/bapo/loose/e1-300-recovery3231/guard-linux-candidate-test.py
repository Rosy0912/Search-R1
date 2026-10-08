import json,base64,hashlib,pathlib,sys,subprocess,os,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');p=json.load(sys.stdin)
assert set(p)=={'resource_watch.py','test_resource_watch_recovery.py'}
D=ROOT/'logs/guard-linux-candidate-5b2aeb43';assert not D.exists();D.mkdir()
for name,item in p.items():
 b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['sha256'];(D/name).write_bytes(b)
start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([str(ROOT/'env/bin/python'),'-m','unittest','discover','-s',str(D),'-p','test_resource_watch_recovery.py','-v'],env=env,text=True,capture_output=True)
(D/'stdout.log').write_text(r.stdout);(D/'stderr.log').write_text(r.stderr)
print(r.stdout);print(r.stderr,file=sys.stderr)
print(json.dumps(dict(scope='ACTUAL_TARGET_LINUX_CPU_PIDFD_AND_MOCK_RESOURCE_TESTS_NOT_GPU_ACCEPTANCE',exit_code=r.returncode,wall_seconds=time.monotonic()-start,files={n:v['sha256']for n,v in p.items()},directory=str(D))))
raise SystemExit(r.returncode)
