from pathlib import Path
import hashlib,json,shlex
H=Path(__file__).resolve().parent;E=H.parent;V=E/'first5-payload-tools-v5-json-exact-candidate'
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns);remote=ns['remote']
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';TOOLS=ROOT+'/tools/first5-native-v5-c72f84da5735';PREP=ROOT+'/runs/formal-3210/evidence/first5-cpu-inspector-v5-preparation'
m=json.loads((V/'candidate-manifest.json').read_text());assert hashlib.sha256((V/'candidate-manifest.json').read_bytes()).hexdigest()=='c72f84da5735936502da6c8a4e329bde7146365e70707f011044a2966d08e84d'
files={}
for n,r in m['files'].items():
 data=(V/n).read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256'];files[TOOLS+'/'+n]=data
files[TOOLS+'/candidate-manifest.json']=(V/'candidate-manifest.json').read_bytes();files[PREP+'/inspect_first5_cpu.py']=(H/'inspect_first5_cpu.py').read_bytes()
results=[]
for path,data in files.items():
 pin=hashlib.sha256(data).hexdigest();code='''import pathlib,hashlib,sys,json,os
p=pathlib.Path(PATH);assert os.getuid()==0 and p.parent.parent.resolve(strict=True)==p.parent.parent
if not p.parent.exists():p.parent.mkdir(mode=0o700)
assert p.parent.resolve(strict=True)==p.parent
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==PIN
if p.exists():assert p.resolve(strict=True)==p and p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
assert hashlib.sha256(p.read_bytes()).hexdigest()==PIN
print(json.dumps({'path':str(p),'sha256':PIN,'uid':os.getuid()}))
'''.replace('PATH',repr(path)).replace('PIN',repr(pin))
  # An unknown transport outcome is not retried; exact existing bytes are never replaced.
 q=remote('python3 -c '+shlex.quote(code),data);results.append(json.loads(q.stdout))
(H/'deployment-original.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n');print(json.dumps({'tools':TOOLS,'runner':PREP,'files':len(results),'runner_sha256':hashlib.sha256((H/'inspect_first5_cpu.py').read_bytes()).hexdigest()}))
