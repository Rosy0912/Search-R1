"""Publish small immutable code only; CPU deployment never uses GPUs or signals."""
import base64,hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
SSH=['ssh','-o','BatchMode=yes','-o','ProxyCommand=none','-o','ProxyJump=none','hkust5']
REMOTE='''import base64,hashlib,json,os,pathlib,sys,time
p=json.load(sys.stdin);assert os.uname().nodename=='xie-lab-4' and os.getuid()==1001
s=pathlib.Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
h=s/'source'/('hkust5-hf-eval-'+p['commit'][:7])/'opsd_signal_weight_hkust5_hf_eval_v1';h.mkdir(parents=True,exist_ok=False)
for name,data in p['files'].items():(h/name).write_bytes(base64.b64decode(data))
m=h/'source-manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==p['manifest_sha256']
for name,v in json.loads(m.read_bytes())['files'].items():assert hashlib.sha256((h/name).read_bytes()).hexdigest()==v['sha256']
virtual='/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/source/'+h.parent.name+'/'+h.name
pub=dict(accepted=True,branch='search-e1',source_commit=p['commit'],remote_readback_sha=p['commit'],source_manifest_sha256=p['manifest_sha256'],deployed_helper=virtual,gpu_started=False,all_deployed_files_read_back=True,checked_unix=time.time())
f=s/'receipts'/('hkust5-hf-evaluator-publication-'+p['commit'][:7]+'.json')
with f.open('x') as out:json.dump(pub,out,indent=2,sort_keys=True);out.write('\\n')
print(json.dumps(dict(accepted=True,helper=virtual,publication=dict(path='/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/'+f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()),manifest_sha256=p['manifest_sha256'],gpu_started=False)))
'''
def main(commit,output):
 root=Path(__file__).resolve().parent;repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
 assert subprocess.check_output(['git','rev-parse','FETCH_HEAD'],text=True).strip()==commit
 records=[];files={}
 for f in sorted(root.iterdir()):
  if not f.is_file():continue
  raw=f.read_bytes();published=subprocess.check_output(['git','show',commit+':'+str(f.relative_to(repo))]);assert raw==published
  files[f.name]=base64.b64encode(raw).decode();records.append(dict(path=str(f.relative_to(repo)),sha256=hashlib.sha256(raw).hexdigest()))
 payload=dict(commit=commit,files=files,manifest_sha256=hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest())
 r=subprocess.run(SSH+['python3 -c '+shlex.quote(REMOTE)],input=json.dumps(payload).encode(),capture_output=True)
 proof=dict(actual_wait_returncode=r.returncode,stderr=r.stderr.decode(),result=json.loads(r.stdout) if r.returncode==0 else r.stdout.decode(),remote_sha=commit,files=records,large_local_products=0)
 Path(output).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));assert r.returncode==0
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
