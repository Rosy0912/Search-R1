import base64, hashlib, json, pathlib, shlex, subprocess
base=pathlib.Path(__file__).resolve().parent
remote=json.loads((base/'tool-assets-verifier-deployment.json').read_text())['remote_dir']
code="import pathlib,json,base64; p=pathlib.Path("+repr(remote)+"); names=['raw-stdout.log','exit-code.txt','wrapper-pid.txt','evidence/manifest.json','evidence/identity.json','evidence/failure.json']; print(json.dumps({n:base64.b64encode((p/n).read_bytes()).decode() for n in names if (p/n).is_file()}))"
r=subprocess.run(['python3',str(base.parent/'remote.py'),'/usr/bin/python3 -c '+shlex.quote(code)],capture_output=True,text=True)
if r.returncode: raise RuntimeError(r.stderr)
files=json.loads(r.stdout)
if 'exit-code.txt' not in files: raise RuntimeError('Not terminal; no local final manifest written')
out=base/'tool-assets-full-evidence'; out.mkdir(exist_ok=False)
record={}
for name,enc in files.items():
    raw=base64.b64decode(enc,validate=True); p=out/name; p.parent.mkdir(exist_ok=True,parents=True); p.write_bytes(raw); record[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
if int(base64.b64decode(files['exit-code.txt']))!=0: raise RuntimeError('Actual verifier failed; evidence retained')
if 'evidence/manifest.json' not in files: raise RuntimeError('Exit0 missing actual manifest')
raw=base64.b64decode(files['evidence/manifest.json']); m=json.loads(raw)
if m['status']!='FULL_TOOL_ASSET_CONTENT_VERIFIED': raise RuntimeError('Wrong terminal status')
(base/'tool-assets-full-manifest.json').write_bytes(raw)
(base/'tool-assets-full-evidence-pins.json').write_text(json.dumps({'remote_dir':remote,'scope':'CPU tool content only','manifest_sha256':hashlib.sha256(raw).hexdigest(),'files':record},indent=2)+'\n')
print(json.dumps({'actual_exit':0,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'index':m['results']['index']['sha256'],'corpus':m['results']['corpus']['sha256']},indent=2))
