"""Staging and CPU test runner for independent candidate directory only."""
import json,base64,hashlib,pathlib,sys,subprocess,os,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005')
p=json.load(sys.stdin);expected={'guard_journal.py','test_guard_journal.py','resource_watch.py'};assert set(p)==expected
assert p['guard_journal.py']['sha256']=='6d28fd7ddb029b7ce062d4a058ade36d170cdcadf52ef7b3203a9de612a465eb'
assert p['resource_watch.py']['sha256']=='1dffc1ac91a625201ff166ae50f427e9a1da6aa46e276401221b1ac1703d3ca5'
D=ROOT/'logs/guard-journal-linux-candidate-6d28fd7d'
assert ROOT.resolve()==ROOT and (ROOT/'logs').resolve()==ROOT/'logs' and not D.exists();D.mkdir()
for name,item in p.items():
 b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['sha256']
 with (D/name).open('xb')as f:f.write(b)
code="""import unittest,sys,os,signal,subprocess
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
def forbidden(*a,**k):raise AssertionError('CPU journal tests prohibit process/GPU/signal calls')
os.kill=forbidden
if hasattr(signal,'pidfd_send_signal'):signal.pidfd_send_signal=forbidden
subprocess.run=forbidden;subprocess.Popen=forbidden
suite=unittest.defaultTestLoader.discover(sys.argv[1],pattern='test_guard_journal.py')
r=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not r.wasSuccessful())
"""
start=time.monotonic();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
r=subprocess.run([str(ROOT/'env/bin/python'),'-c',code,str(D)],env=env,text=True,capture_output=True,timeout=60)
for name,text in [('stdout.log',r.stdout),('stderr.log',r.stderr)]:
 with (D/name).open('x')as f:f.write(text)
record=dict(scope='ACTUAL_TARGET_LINUX_CPU_24_READONLY_JOURNAL_TESTS_MOCK_RESOURCES_NOT_GPU',exit_code=r.returncode,wall_seconds=time.monotonic()-start,directory=str(D),host=os.uname().nodename,python=str(ROOT/'env/bin/python'),cuda_visible_devices='',gpu_or_signal_calls_allowed=False,files={n:hashlib.sha256((D/n).read_bytes()).hexdigest()for n in p})
with (D/'record.json').open('x')as f:json.dump(record,f,indent=2)
print(r.stdout);print(r.stderr,file=sys.stderr);print(json.dumps(record));raise SystemExit(r.returncode)
