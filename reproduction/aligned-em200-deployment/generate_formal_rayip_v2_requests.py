"""Generate immutable, split, local request artifacts; never contacts a target."""
from pathlib import Path
import argparse, ast, base64, hashlib, json, zlib

ROOT='/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009'
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
HERE=Path(__file__).resolve().parent
NAMES=('formal.sbatch','formal_worker.sh','formal_inner.sh','formal_driver.py','formal_service_proof.py','formal_admission.py','submit_formal.py','ray_cli_compat.py','bind_retrieval_proof.py','test_formal_service_proof.py','test_retry_namespace.py','isolation-manifest.json')
EVIDENCE=('aligned_final_phase2.py','aligned_first5_math.py','aligned_final_evidence.py','aligned_first5_cpu_evidence.py','aligned-first5-math-provenance.json','collect_hardware_evidence.py','publish_first5_release.py')
PRELUDE=r'''
from pathlib import Path
import base64,datetime,fcntl,hashlib,json,os,stat,subprocess,sys,zlib
R=Path(PLAN['root']); D=R/'tools/deployment/formal-rayip-v2'; S=R/'scheduler-head/formal-rayip-v2'
def need(ok,msg):
 if not ok: raise RuntimeError(msg)
def sha(p):
 p=Path(p);need(p.is_file() and not p.is_symlink() and p.resolve()==p,'ordinary canonical file required: '+str(p))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def directory(p):need(p.is_dir() and not p.is_symlink() and p.resolve()==p,'canonical directory required: '+str(p))
def exclusive(p,b):
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
def jwrite(p,v):exclusive(p,(json.dumps(v,sort_keys=True,indent=2)+'\n').encode())
def originals():
 result={p:sha(p) for p in PLAN['old33']};need(result==PLAN['old33'],'old33 originals changed');return result
need(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'),'optimized checks forbidden')
need(sys.platform=='linux' and os.uname().nodename=='h100-5','wrong target')
for p in (R,R/'tools',R/'tools/deployment',R/'scheduler-head',R/'repo',R/'env'):directory(p)
need(subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()==PLAN['source'],'source080 required')
need(not subprocess.check_output(['git','-C',str(R/'repo'),'status','--porcelain'],text=True).strip(),'dirty source')
locks=[]
for name in ('submission-global.lock','execution-global.lock'):
 p=R/'scheduler-head'/name
 need(p.is_file() and not p.is_symlink() and p.resolve()==p,'existing global lock required')
 f=p.open('r+b');st=os.fstat(f.fileno());need((st.st_dev,st.st_ino)==(p.stat().st_dev,p.stat().st_ino),'lock identity race');need(st.st_nlink==1 and st.st_uid==os.getuid(),'lock ownership');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
before=originals()
'''
BEGIN=r'''
need(not D.exists() and not D.is_symlink() and not S.exists() and not S.is_symlink(),'unknown existing retry namespace: no overwrite or retry')
S.mkdir();D.mkdir();(S/'install-parts').mkdir()
need(originals()==before,'old33 changed during namespace creation')
jwrite(S/'install-intent.json',dict(plan=PLAN,before_old33=before,status='INSTALL_INCOMPLETE_NO_SUBMISSION'))
print(json.dumps(dict(status='INSTALL_NAMESPACE_CREATED_NOT_LAUNCH',plan_sha256=PLAN_SHA)))
'''
CHECK=r'''
for p in (D,S,S/'install-parts'):directory(p)
sha(S/'install-intent.json');intent=json.loads((S/'install-intent.json').read_bytes())
need(intent['plan']==PLAN and intent['before_old33']==before,'different install intent')
'''
CHUNK=r'''
need(not (S/'install-complete.json').exists(),'installation already complete')
raw=base64.b64decode(PAYLOAD,validate=True);need(hashlib.sha256(raw).hexdigest()==PART_SHA,'part hash')
exclusive(S/'install-parts'/PART,raw)
need(originals()==before,'old originals changed during upload')
print(json.dumps(dict(status='APPEND_ONLY_PART_INSTALLED',part=PART,sha256=PART_SHA)))
'''
FINISH=r'''
need(not (S/'install-complete.json').exists(),'existing completion receipt forbidden')
need(set(p.name for p in (S/'install-parts').iterdir())==set(PLAN['parts']),'partial or unknown part inventory')
for part,expected in PLAN['parts'].items():need(sha(S/'install-parts'/part)==expected,'part bytes differ')
need(not list(D.iterdir()),'unknown candidate content before final assembly')
packed=b''.join((S/'install-parts'/part).read_bytes() for part in PLAN['parts'])
need(len(packed)==PLAN['compressed_size'] and hashlib.sha256(packed).hexdigest()==PLAN['compressed_sha256'],'compressed bundle differs')
codec=zlib.decompressobj();decoded=codec.decompress(packed,1048577)
need(len(decoded)<=1048576 and codec.eof and not codec.unconsumed_tail and not codec.unused_data,'oversized/trailing/incomplete compressed bundle')
need(len(decoded)==PLAN['uncompressed_size'] and hashlib.sha256(decoded).hexdigest()==PLAN['uncompressed_sha256'],'bundle bytes differ')
bundle=json.loads(decoded);need(isinstance(bundle,dict) and set(bundle)==set(PLAN['files']),'bundle inventory differs')
for name,item in PLAN['files'].items():
 relative=Path(name);need(not relative.is_absolute() and '..' not in relative.parts and name in PLAN['files'],'unsafe bundle path')
 raw=base64.b64decode(bundle[name],validate=True)
 need(len(raw)==item['size'] and hashlib.sha256(raw).hexdigest()==item['sha256'],'file payload differs')
 p=D/name
 for ancestor in reversed(p.parents):
  if ancestor==D or D in ancestor.parents:
   if not ancestor.exists():ancestor.mkdir()
   directory(ancestor)
 exclusive(p,raw)
 need(sha(p)==item['sha256'],'installed bytes differ')
after=originals();need(after==before,'old originals changed')
jwrite(S/'install-complete.json',dict(status='APPEND_ONLY_INSTALLED_NOT_LAUNCHED',plan_sha256=PLAN_SHA,files=PLAN['files'],before_old33=before,after_old33=after,source=PLAN['source']))
print(json.dumps(dict(status='APPEND_ONLY_INSTALLED_NOT_LAUNCHED',plan_sha256=PLAN_SHA,files=len(PLAN['files']),old_originals_unchanged=len(after))))
'''
VERIFY=r'''
sha(S/'install-complete.json')
complete=json.loads((S/'install-complete.json').read_bytes());need(complete['plan_sha256']==PLAN_SHA and complete['files']==PLAN['files'],'different installed plan')
for name,item in PLAN['files'].items():need(sha(D/name)==item['sha256'],'candidate bytes differ')
need(not (S/'actual-linux-cpu-verification.json').exists() and not (S/'actual-linux-cpu-verification.json').is_symlink(),'existing verification result: no retry')
jwrite(S/'actual-linux-cpu-verification-intent.json',dict(plan_sha256=PLAN_SHA,scope='CPU_TESTS_AND_READ_ONLY_ADMISSION_NO_GPU_HEAD_SBATCH'))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'repo')+':'+str(D),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
python=str(R/'env/bin/python');results={};status='CPU_VERIFICATION_FAILED_NOT_ACCEPTED'
def run(name,args,timeout=120):
 p=subprocess.run([python,'-B']+args,cwd=R/'repo',env=env,capture_output=True,text=True,timeout=timeout)
 results[name]=dict(command=[python,'-B']+args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
 return p
try:
 identity=run('identity',['-c',"import sys,os,json,torch;assert sys.platform=='linux';assert os.uname().nodename=='h100-5';assert sys.prefix=="+repr(str(R/'env'))+";assert os.environ['CUDA_VISIBLE_DEVICES']=='';assert not torch.cuda.is_available();assert not torch.cuda.is_initialized();print(json.dumps(dict(prefix=sys.prefix,executable=sys.executable,host=os.uname().nodename,platform=sys.platform,cuda_initialized=torch.cuda.is_initialized())))"])
 need(identity.returncode==0,'actual isolated CPU identity failed')
 syntax=run('compile',['-c',"from pathlib import Path; root=Path("+repr(str(D))+"); files=list(root.rglob('*.py')); [compile(p.read_bytes(),str(p),'exec') for p in files];print(len(files))"])
 need(syntax.returncode==0,'syntax failed')
 for name in ('formal.sbatch','formal_worker.sh','formal_inner.sh'):
  p=subprocess.run(['bash','-n',str(D/name)],capture_output=True,text=True,timeout=30);results['bash_'+name]=dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr);need(p.returncode==0,'shell syntax failed')
 tests=run('pytest',['-m','pytest','-q','-p','no:cacheprovider',str(D/'test_formal_service_proof.py'),str(D/'test_retry_namespace.py')])
 need(tests.returncode==0 and '29 passed' in tests.stdout,'expected all 29 pure tests')
 helpresult=run('ray_cli_help',[str(D/'ray_cli_compat.py'),'--help']);need(helpresult.returncode==0,'CLI help failed')
 admission=run('actual_admission',[str(D/'formal_admission.py'),'verify'],300)
 need(admission.returncode==0 and 'FORMAL_BOUND_ADMISSION_PASS_NOT_TRAINING_COMPLETION' in admission.stdout,'read-only actual admission failed')
 for name,item in PLAN['files'].items():need(sha(D/name)==item['sha256'],'candidate changed during verification')
 status='ACTUAL_LINUX_29_CPU_TESTS_AND_BOUND_ADMISSION_PASS_NOT_GPU_OR_TRAINING'
except BaseException:
 status='CPU_VERIFICATION_FAILED_NOT_ACCEPTED';raise
finally:
 after=originals();need(after==before,'old33 changed during verification')
 jwrite(S/'actual-linux-cpu-verification.json',dict(status=status,scope='CPU_TESTS_AND_READ_ONLY_ADMISSION_NO_GPU_HEAD_SBATCH',source=PLAN['source'],plan_sha256=PLAN_SHA,results=results,before_old33=before,after_old33=after))
 print(json.dumps(dict(status=status,results=results,old_originals_unchanged=len(after))))
'''

def build(candidate):
 manifest=json.loads((candidate/'isolation-manifest.json').read_bytes())
 old=manifest['original_failure_files']
 if len(old)!=33 or manifest['root']!=ROOT or manifest['source']!=SOURCE:raise ValueError('fixed root/source/old33 required')
 tree=ast.parse((candidate/'formal_admission.py').read_bytes())
 literals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PREVIOUS_FAILURE_FILES','FORMAL_FILES')}
 if literals['PREVIOUS_FAILURE_FILES']!=old:raise ValueError('manifest/runtime old33 differs')
 for name,digest in literals['FORMAL_FILES'].items():
  if hashlib.sha256((candidate/name).read_bytes()).hexdigest()!=digest:raise ValueError('runtime formal pin differs: '+name)
 files={};parts={};bundle={}
 for name in NAMES+tuple('evidence-tools/'+n for n in EVIDENCE):
  p=candidate/name
  if not p.is_file() or p.is_symlink():raise ValueError('ordinary candidate required: '+name)
  raw=p.read_bytes();bundle[name]=base64.b64encode(raw).decode()
  files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
 decoded=json.dumps(bundle,sort_keys=True,separators=(',',':')).encode()
 if len(decoded)>1048576:raise ValueError('candidate bundle exceeds fixed 1MiB limit')
 packed=zlib.compress(decoded,9)
 for offset in range(0,len(packed),12000):parts[f'{len(parts):04d}.part']=packed[offset:offset+12000]
 plan=dict(root=ROOT,source=SOURCE,old33=old,files=files,parts={p:hashlib.sha256(b).hexdigest() for p,b in parts.items()},compressed_size=len(packed),compressed_sha256=hashlib.sha256(packed).hexdigest(),uncompressed_size=len(decoded),uncompressed_sha256=hashlib.sha256(decoded).hexdigest())
 return plan,parts

def generate(candidate,out):
 plan,parts=build(candidate); encoded=json.dumps(plan,sort_keys=True,separators=(',',':'));digest=hashlib.sha256(encoded.encode()).hexdigest()
 out.mkdir() # Unknown outcome or prior request artifacts must never be overwritten.
 def write(name,body,extra=''):
  code='PLAN='+repr(plan)+'\nPLAN_SHA='+repr(digest)+'\n'+extra+PRELUDE+body
  compile(code,name,'exec')
  with (out/name).open('x') as f:f.write(code)
 write('000-install-begin.py',BEGIN)
 for index,(part,raw) in enumerate(parts.items(),1):
  write(f'{index:03d}-install-part.py',CHECK+CHUNK,'PART='+repr(part)+'\nPART_SHA='+repr(plan['parts'][part])+'\nPAYLOAD='+repr(base64.b64encode(raw).decode())+'\n')
 write(f'{len(parts)+1:03d}-install-finish.py',CHECK+FINISH)
 write('verify-actual-linux.py',CHECK+VERIFY)
 with (out/'request-manifest.json').open('x') as f:json.dump(dict(plan_sha256=digest,plan=plan,requests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.py'))},scope='LOCAL_REQUESTS_ONLY_NOT_EXECUTED'),f,indent=2)
 return out
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 print(generate(HERE/'formal-rayip-v2',args.output))
