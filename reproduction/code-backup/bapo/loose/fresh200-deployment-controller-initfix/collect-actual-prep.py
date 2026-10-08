"""Read-only actual newROOT preparation download through the existing H100 route.
Writes only a new local evidence directory; never creates remote files or jobs.
"""
import argparse,base64,hashlib,json,shlex,subprocess
from pathlib import Path
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006'
code="""
import base64,hashlib,json,os,pathlib,subprocess,time,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
source='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert sys.platform=='linux' and os.uname().nodename=='h100-5' and sys.prefix==str(root/'env')
assert root.resolve()==root and not root.is_symlink()
assert (root/'source-expected.txt').read_text().strip()==source
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==source
names=['target-cpu-gate.json','target-cpu-gate.raw.log','context-budget-audit.json','runtime-tests.actual.json','runtime-tests.actual.log','runtime-executable.json','isolation-setup.json','tools-manifest.json','frozen-policy-safety-config.json','user-fixed-policy-thresholds.json','latest-human-goal-behavior.json']
files={};missing=[]
for name in names:
 p=root/'logs'/name
 if not p.exists():missing.append(name);continue
 assert p.is_file() and not p.is_symlink() and p.resolve()==p
 raw=p.read_bytes();stable=p.read_bytes()==raw
 files[name]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),stable_during_read=stable,base64=base64.b64encode(raw).decode())
print(json.dumps(dict(scope='ACTUAL_H1005_READ_ONLY_NEW_SOURCE_PREPARATION_NOT_GPU_OR_LAUNCH',source_commit=source,root=str(root),host=os.uname().nodename,observed_unix_seconds=time.time(),files=files,missing=missing)))
"""
def collect(output):
 b=Path(__file__).resolve().parents[1]
 if output.exists():raise ValueError('New local evidence directory required')
 output.mkdir()
 command=ROOT+'/env/bin/python -B -c '+shlex.quote(code)
 result=subprocess.run(['python3',str(b/'h100_exec.py'),'--socks','--timeout','60',command],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 (output/'read-route.raw.log').write_bytes(result.stdout+b'\nSTDERR\n'+result.stderr)
 if result.returncode:raise RuntimeError('Actual read route failed; raw response preserved')
 value=json.loads(result.stdout)
 assert value['source_commit']==SOURCE and value['root']==ROOT and value['host']=='h100-5'
 for name,item in value['files'].items():
  assert '/'not in name and item['path']==ROOT+'/logs/'+name and item['stable_during_read']is True
  raw=base64.b64decode(item.pop('base64'),validate=True)
  assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256']
  with (output/name).open('xb')as f:f.write(raw)
 value['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (output/'actual-preparation-records.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(dict(output=str(output),files=len(value['files']),missing=value['missing'],source_commit=SOURCE,scope=value['scope'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();collect(args.output.resolve())
