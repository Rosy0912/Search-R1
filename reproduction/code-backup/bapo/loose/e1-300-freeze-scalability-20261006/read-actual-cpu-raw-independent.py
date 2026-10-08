import subprocess,shlex,json,base64,hashlib
from pathlib import Path
R='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006'
code="""from pathlib import Path
import json,base64,hashlib,subprocess
r=Path(%r)
names=['logs/cpu-gate.json','logs/cpu-gate.log','logs/cpu-gate-source.txt','source-expected.txt','repo/verl/trainer/config/ppo_bapo_e1_300.yaml']
files={}
for name in names:
 p=r/name;assert p.is_file()and not p.is_symlink()and p.resolve()==p;b=p.read_bytes();files[name]=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode())
print(json.dumps(dict(root=str(r),head=subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True),files=files)))
"""%R
q=subprocess.run(['python3','work/h100_exec.py','--socks','--timeout','60','python3 -B -c '+shlex.quote(code)],capture_output=True,timeout=70)
Path('work/evidence/freeze-scalability-b711-independent-cpu-raw-read.stderr').write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode();d=json.loads(q.stdout)
base=Path('work/evidence/freeze-scalability-b711-independent-cpu-raw');base.mkdir(exist_ok=True)
for name,item in d['files'].items():
 b=base64.b64decode(item.pop('data'));assert len(b)==item['bytes']and hashlib.sha256(b).hexdigest()==item['sha256'];p=base/name.replace('/','__');p.write_bytes(b);item['local_path']=str(p.resolve())
(base/'inventory.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(dict(head=d['head'],status=d['status'],files={k:v['sha256']for k,v in d['files'].items()}),indent=2))
