"""Copy independently reviewed future acceptance helpers only; do not run a publisher."""
import json,sys,base64,hashlib,pathlib,ast,subprocess,os
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
D=ROOT/'evidence-tools/first5-3231'
assert ROOT.resolve()==ROOT and D.parent.resolve()==D.parent and D.parent.is_dir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_original_tools():
 expected=json.loads((ROOT/'logs/tools-manifest.json').read_text())['files']
 assert {p.name:sha(p)for p in (ROOT/'tools').iterdir()if p.is_file()}==expected
check_original_tools()
assert sha(ROOT/'evidence-tools/step5-extract.py')=='a27674a26f4958f9514c19be6f9f5ba5fab9bfd640c586b5bf293266da29ee74'
payload=json.load(sys.stdin)
allowed={'first5_guard3231.py','validate-publish-first5-release3231.py','archive-first5-extraction3231.py','test_first5_guard3231.py','test_first5_independent3231.py','FIRST5-3231-LOCAL-CANDIDATE-MANIFEST.json','FIRST5-3231-LOCAL-SELF-REVIEW.md','FIRST5-3231-INDEPENDENT-TOOL-REVIEW.md','first5-tool-approval3231.json','first5-3231-independent-tool-tests-final.log'}
assert set(payload)==allowed
raw={}
for name,item in payload.items():
 assert set(item)=={'base64','sha256'}
 b=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==item['sha256'];raw[name]=b
approval=json.loads(raw['first5-tool-approval3231.json'])
assert hashlib.sha256(raw['first5-tool-approval3231.json']).hexdigest()=='784ae61cebe3da2f52b592c703552de29d98c3bfc81b970847b9d95a2da924d1'
assert approval['status']=='PASS_INDEPENDENT_FIRST5_TOOLS_REVIEW'and approval['job_id']==3231 and approval['root']==str(ROOT) and approval['source_commit']=='39cdcdd1404213b897bb0de99cc52d72707682f0'
for item in approval['files']+[approval['independent_report'],approval['tests']]:
 name=pathlib.Path(item['path']).name
 assert name in raw and len(raw[name])==item['bytes']and hashlib.sha256(raw[name]).hexdigest()==item['sha256']
D.mkdir(exist_ok=True);assert D.resolve()==D and not D.is_symlink()
for name,b in raw.items():
 p=D/name
 if p.exists():assert p.is_file()and not p.is_symlink()and p.read_bytes()==b
 else:
  with p.open('xb')as f:f.write(b)
 if name.endswith('.py'):ast.parse(b,filename=str(p))
check_original_tools()
# Only mock tests execute: checkpoint/proc/Slurm subprocesses are intercepted, all outputs in temp dirs.
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
r=subprocess.run([str(ROOT/'env/bin/python'),'-m','unittest','-v','test_first5_guard3231','test_first5_independent3231'],cwd=str(D),env=env,text=True,capture_output=True)
log=D/'target-linux-mock-tests.log'
with log.open('x')as f:f.write(r.stdout+r.stderr)
assert r.returncode==0 and 'Ran 25 tests' in log.read_text()and '\nOK\n' in log.read_text()
check_original_tools()
print(json.dumps({'status':'REVIEWED_FUTURE_TOOLS_STAGED_LINUX25_MOCK_PASS','job_id':3231,'directory':str(D),'log_sha256':sha(log),'publisher_executed':False,'actual_first5_accepted':False,'original22tools_unchanged':True}))
