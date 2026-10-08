import base64,hashlib,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]); target=sys.argv[2]; data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest()
command="python3 - <<'PYREMOTE'\nimport base64,hashlib,os,subprocess\nfrom pathlib import Path\np=Path("+repr(target)+")\ntemp=p.with_name(p.name+'.partial')\ndata=base64.b64decode("+repr(base64.b64encode(data).decode())+")\nassert hashlib.sha256(data).hexdigest()=="+repr(sha)+"\ntemp.write_bytes(data)\n"+("compile(data,str(temp),'exec')\n" if p.suffix=='.py' else "subprocess.run(['bash','-n',str(temp)],check=True)\n" if p.suffix=='.sh' else "")+"os.replace(temp,p)\nprint('VERIFIED_ATOMIC_UPLOAD',p.name,"+repr(sha)+")\nPYREMOTE"
raise SystemExit(subprocess.run([sys.executable,str(Path(__file__).with_name('remote.py')),command]).returncode)
