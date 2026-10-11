"""Only two small immutable historical G controller exits move between HKUST nodes."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
import entry
SSH=['ssh','-o','BatchMode=yes','-o','ProxyCommand=none','-o','ProxyJump=none']
def main(output):
 refs=entry.G_TERMINALS
 read='''import json,pathlib,hashlib,base64,sys
refs=json.load(sys.stdin);out={}
for group,ref in refs.items():
 p=pathlib.Path(ref['path']);raw=p.read_bytes();assert len(raw)<65536 and hashlib.sha256(raw).hexdigest()==ref['sha256'];v=json.loads(raw);assert v['actual_wait_returncode']==0
 out[group]=dict(reference=ref,data=base64.b64encode(raw).decode())
print(json.dumps(out))'''
 r=subprocess.run(SSH+['hkust4','python3 -c '+shlex.quote(read)],input=json.dumps(refs).encode(),capture_output=True,check=True)
 install='''import json,pathlib,hashlib,base64,sys,os
assert os.uname().nodename=='xie-lab-4' and os.getuid()==1001
values=json.load(sys.stdin);base=pathlib.Path('/home/shaoshitong/search-e1-migration-20261011');out=[]
for g,v in values.items():
 ref=v['reference'];raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==ref['sha256'];p=base/ref['path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb') as f:f.write(raw)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];out.append(dict(group=g,reference=ref,bytes=len(raw),actual_controller_wait0=json.loads(raw)['actual_wait_returncode']==0))
print(json.dumps(dict(accepted=True,files=out,source_preserved=True,no_large_local_files=True)))'''
 t=subprocess.run(SSH+['hkust5','python3 -c '+shlex.quote(install)],input=r.stdout,capture_output=True)
 proof=dict(actual_source_wait=r.returncode,actual_target_wait=t.returncode,stderr=t.stderr.decode(),result=json.loads(t.stdout) if t.returncode==0 else t.stdout.decode())
 Path(output).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));assert t.returncode==0
if __name__=='__main__':main(sys.argv[1])
