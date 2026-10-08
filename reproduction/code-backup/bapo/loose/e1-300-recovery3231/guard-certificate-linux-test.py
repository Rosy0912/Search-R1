import json,base64,hashlib,pathlib,sys,subprocess,os,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');p=json.load(sys.stdin)
assert set(p)=={'resource_watch.py','test_resource_watch_recovery.py'}
assert p['resource_watch.py']['sha256']=='1dffc1ac91a625201ff166ae50f427e9a1da6aa46e276401221b1ac1703d3ca5'
D=ROOT/'logs/guard-linux-candidate-1dffc1ac';assert ROOT.resolve()==ROOT and (ROOT/'logs').resolve()==ROOT/'logs' and not D.exists();D.mkdir()
for name,item in p.items():
 b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['sha256'];
 with (D/name).open('xb')as f:f.write(b)
start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1');r=subprocess.run([str(ROOT/'env/bin/python'),'-m','unittest','discover','-s',str(D),'-p','test_resource_watch_recovery.py','-v'],env=env,text=True,capture_output=True,timeout=60)
(D/'stdout.log').write_text(r.stdout);(D/'stderr.log').write_text(r.stderr)
print(r.stdout);print(r.stderr,file=sys.stderr)
print(json.dumps(dict(scope='ACTUAL_TARGET_LINUX_CPU_NEW_CERTIFICATE_PIDFD_AND_MOCK_RESOURCE_TESTS_NOT_GPU_ACCEPTANCE',host=os.uname().nodename,cuda_visible_devices='',exit_code=r.returncode,wall_seconds=time.monotonic()-start,files={n:v['sha256']for n,v in p.items()},directory=str(D))))
raise SystemExit(r.returncode)
