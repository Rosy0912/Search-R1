"""Mac-side task transfer; each unknown outcome is left for inspection, not retried."""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess,sys
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;out=Path(a.evidence).resolve();out.mkdir(parents=True,exist_ok=True)
deploy=(here/'deploy.py').read_text()
raw=Path(a.bundle).read_bytes();bundle_sha=hashlib.sha256(raw).hexdigest()
parts=[raw[i:i+44*1024]for i in range(0,len(raw),44*1024)]
def call(name,body,timeout=90):
 if list(out.glob(name+'.*')):raise RuntimeError('Existing request/outcome artifacts: inspect before intentionally repeating '+name)
 script=out/(name+'.remote.py')
 with script.open('x')as f:f.write(body)
 result=out/(name+'.actual.json')
 command=[sys.executable,str(here/'transport.py'),'--python-file',str(script),'--output',str(result),'--timeout',str(timeout)]
 subprocess.run(command,check=True)
 return json.loads(result.read_text())
def program(action):return 'import json\nnamespace={"__name__":"task_deployment"}\nexec('+repr(deploy)+',namespace)\nprint(json.dumps('+action+'))\n'
for i,part in enumerate(parts):
 call('upload-source-'+str(i),program('namespace["upload"]('+str(i)+','+repr(base64.b64encode(part).decode())+','+repr(hashlib.sha256(part).hexdigest())+')'))
value=call('setup-source',program('namespace["setup"]('+str(len(parts))+','+repr(bundle_sha)+')'),180)
assert value['source']==SOURCE and value['bundle_sha256']==bundle_sha
files=[]
for path in sorted(here.rglob('*')):
 if not path.is_file() or '__pycache__' in path.parts or path.name.endswith('.pyc'):continue
 blob=path.read_bytes();files.append((str(path.relative_to(here)),base64.b64encode(blob).decode(),hashlib.sha256(blob).hexdigest()))
# Bound each tool separately to keep all management/SSH arguments below OS limits.
for i,item in enumerate(files):call('install-tool-'+str(i),program('namespace["install_tools"]('+repr([item])+')'))
manifest={'source':SOURCE,'bundle_sha256':bundle_sha,'parts':len(parts),'tools':[{'path':x[0],'sha256':x[2]}for x in files],'scope':'SOURCE_AND_TOOLS_TRANSFER_NOT_TRAINING'}
(out/'transfer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
