from pathlib import Path
import hashlib,json,shlex
H=Path(__file__).resolve().parent;E=H.parent;V=E/'first5-tools-a096-raydedup-candidate'
ns={'__file__':str(E/'deploy_formal_reviewed.py')};s=Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns);remote=ns['remote']
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';D=ROOT+'/tools/first5-e4d8bd9e088e';PIN='e4d8bd9e088e49fc92f335abfaa47737830d1ba9320cbe206d1bd130876518f7';m=json.loads((V/'manifest.json').read_text());assert hashlib.sha256((V/'manifest.json').read_bytes()).hexdigest()==PIN
results=[]
for n in list(m['tools'])+['manifest.json']:
 raw=(V/n).read_bytes();pin=hashlib.sha256(raw).hexdigest();assert n=='manifest.json' or pin==m['tools'][n]
 code='''import pathlib,hashlib,json,sys,os
p=pathlib.Path(PATH);assert os.getuid()==0 and p.parent.parent.resolve(strict=True)==p.parent.parent
if not p.parent.exists():p.parent.mkdir(mode=0o700)
assert p.parent.resolve(strict=True)==p.parent
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==PIN
if p.exists():assert p.resolve(strict=True)==p and p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
assert hashlib.sha256(p.read_bytes()).hexdigest()==PIN
print(json.dumps({'path':str(p),'sha256':PIN,'uid':os.getuid()}))
'''.replace('PATH',repr(D+'/'+n)).replace('PIN',repr(pin))
 q=remote('python3 -c '+shlex.quote(code),raw);results.append(json.loads(q.stdout))
(H/'external-deployment-original.json').write_text(json.dumps(results,sort_keys=True,indent=2)+'\n');print(json.dumps({'tools':D,'manifest_sha256':PIN,'files':len(results),'scope':'Deployment only; no receipt or actual acceptance'}))
