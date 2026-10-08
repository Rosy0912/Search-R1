"""Exclusive own-root code install and CUDA-hidden CPU33 only. No collector invocation."""
import argparse,hashlib,io,json,os,re,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
NONCE='bab8981d9fd14afc98923eb4ebb89e33';JOB=3315
RUN=ROOT/'runs'/('formal200-'+NONCE+'-3315')
DEST=ROOT/'first5-evidence-tools-3315-v4'
TOOLS_SHA='fbd0098b946e23a8eb7304912cad01b8f339e249f627c919ddec8313c2337859'
MANIFEST_SHA='e593b8eb4290e3717db0ede230cc9455b6b06df5edd8640ddfa046bd4b81e576'
TAR_SHA='7d47e9259930b7dbec5e0a31dea19befc3f334815ffe477b2e55ad93bbdb7ff1'
def need(ok,message):
 if not ok:raise ValueError(message)
def strict(raw):
 def pairs(items):
  d={}
  for k,v in items:need(k not in d,'Duplicate JSON key');d[k]=v
  return d
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(p):
 p=Path(p);need(p.is_absolute()and p.resolve()==p and not p.is_symlink()and ROOT in p.parents and p.is_file(),'Canonical own-root regular file required');return p
def package(path):
 need(not Path(path).is_symlink()and Path(path).is_file(),'Regular package required')
 blob=Path(path).read_bytes();need(hashlib.sha256(blob).hexdigest()==TAR_SHA,'Frozen package bytes changed')
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:')as t:
  members=t.getmembers();need(len(members)==14 and len({m.name for m in members})==14 and all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts and str(Path(m.name))==m.name for m in members),'Exact14 regular unique package members')
  files={m.name:t.extractfile(m).read()for m in members}
 need('package-manifest.json'in files and hashlib.sha256(files['package-manifest.json']).hexdigest()==MANIFEST_SHA,'Exact frozen manifest required')
 m=strict(files['package-manifest.json']);need(m.get('source_commit')==SOURCE and type(m.get('job_id'))is int and m['job_id']==JOB and m.get('target_fixture_tests')==33 and m.get('native_acceptance')is False and m.get('production_release')is False and len(m['files'])==13 and set(files)==set(m['files'])|{'package-manifest.json'},'Wrong package scope/inventory')
 for n,h in m['files'].items():need(hashlib.sha256(files[n]).hexdigest()==h,'Member digest changed '+n)
 return files,m

def identity(receipt,start,event):
 comment='BAPO_FRESH200:formal200:'+SOURCE+':'+NONCE
 for r in (receipt,start):
  need(r.get('phase')=='formal200'and r.get('source')==SOURCE and r.get('nonce')==NONCE and r.get('job_id')==str(JOB)and r.get('comment')==comment,'Exact existing actual3315 receipt/worker identity required')
 need(start.get('run')==str(RUN)and start.get('host')=='h100-5'and type(start.get('pid'))is int and start['pid']>0,'Actual run worker identity required')
 need(event.get('run')==str(RUN)and event.get('slurm_job_id')==str(JOB),'Historical UUID sample run/job differs')
 a=event.get('allocated_gpus');need(isinstance(a,dict)and len(a)==8,'Eight historical allocated GPUs required')
 uuids=[x.get('uuid')for x in a.values()];need(all(type(u)is str and u.startswith('GPU-')and len(u)>4 for u in uuids)and len(set(uuids))==8,'Distinct actual UUID binding required')
 return sorted(uuids)
def review_document(r,files,m,self_sha):
 need(r.get('status')=='PASS_CODE_ONLY_3315_FIRST5_V4_NATIVE_FORMAT_PACKAGE_REVIEW'and r.get('mustfix')==[]and r.get('native_acceptance')is False and r.get('source_commit')==SOURCE and type(r.get('job_id'))is int and r['job_id']==JOB,'Exact independent package code review required')
 need(r.get('package_manifest_sha256')==MANIFEST_SHA and r.get('package_sha256')==TAR_SHA and r.get('installer_sha256')==self_sha,'Review must bind installer/archive/manifest')
 need(all(r.get('files',{}).get(n)==h for n,h in m['files'].items()),'Review must bind every13 payload bytes')
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def exclusive_files(destination,files):
 need(not destination.exists()and not destination.is_symlink(),'Existing installation retained; no overwrite')
 destination.mkdir(mode=0o700)
 for n,raw in files.items():
  p=destination/n
  if not p.parent.exists():p.parent.mkdir(mode=0o700)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 for d in [p for p in destination.rglob('*')if p.is_dir()]+[destination,destination.parent]:fsyncdir(d)
def execute(args,files,m):
 need(not sys.flags.optimize and sys.platform=='linux'and os.uname().nodename=='h100-5'and sys.prefix==str(ROOT/'env'),'Own lexical target Linux environment required')
 canonical(args.package);canonical(Path(__file__))
 need(ROOT.resolve()==ROOT and not ROOT.is_symlink(),'Canonical ownROOT required')
 review=canonical(args.review);need(re.fullmatch('[0-9a-f]{64}',args.expected_review_sha or '')and sha(review)==args.expected_review_sha,'Exact independent review bytes required')
 review_document(strict(review.read_bytes()),files,m,sha(Path(__file__)))
 need(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Frozen clean source required')
 production=canonical(ROOT/'logs/tools-manifest.json');need(sha(production)==TOOLS_SHA,'Production21 manifest differs')
 p=strict(production.read_bytes());need(p.get('source_commit')==SOURCE and len(p['files'])==21,'Exact production21 required')
 for n,h in p['files'].items():need(sha(canonical(ROOT/'tools'/n))==h,'Frozen production tool changed '+n)
 receipt=canonical(ROOT/'scheduler-head/formal200-receipt.json');start=canonical(RUN/'worker-start.json');journal=canonical(RUN/'resource-watch.jsonl')
 # Bounded first complete historical sample only: no journal fullhash/replay, no NVML/proc mutation.
 with journal.open('rb')as f:raw=f.readline(4194305)
 need(len(raw)<=4194304 and raw.endswith(b'\n'),'Complete bounded first allocation sample required')
 uuids=identity(strict(receipt.read_bytes()),strict(start.read_bytes()),strict(raw))
 log=ROOT/'logs/first5-v4-linux33-tests.actual.log';record=ROOT/'logs/first5-v4-linux33-tests.actual.json'
 need(not log.exists()and not log.is_symlink()and not record.exists()and not record.is_symlink(),'Existing test evidence retained; no overwrite/retry')
 exclusive_files(DEST,files)
 env=dict(os.environ,PYTHONPATH=str(ROOT/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1');env.pop('PYTHONOPTIMIZE',None)
 wall=time.perf_counter()
 with log.open('xb')as f:
  result=subprocess.run([str(ROOT/'env/bin/python'),'-B','-m','unittest','discover','-s',str(DEST/'tests'),'-v'],cwd=ROOT/'repo',env=env,stdout=f,stderr=subprocess.STDOUT);f.flush();os.fsync(f.fileno())
 output=log.read_text();ok=result.returncode==0 and bool(re.search(r'^Ran 33 tests in [\d.]+s$',output,re.M))and bool(re.search(r'^OK$',output,re.M))and 'skipped='not in output
 value=dict(scope='ACTUAL_LINUX33_CPU_TOOL_FIXTURES_NOT_NATIVE_FIRST5_OR_HF',source_commit=SOURCE,job_id=JOB,run_root=str(RUN),code_directory=str(DEST),package_sha256=TAR_SHA,package_manifest_sha256=MANIFEST_SHA,review_sha256=args.expected_review_sha,installer_sha256=sha(Path(__file__)),receipt_sha256=sha(receipt),worker_start_sha256=sha(start),historical_first_allocation_sample_sha256=hashlib.sha256(raw).hexdigest(),allocated_uuids=uuids,uuid_scope='Historical first actual3315 sample identity only; not current GPU or lease revalidation',host=os.uname().nodename,sys_prefix=sys.prefix,interpreter=str(ROOT/'env/bin/python'),cwd=str(ROOT/'repo'),exit_code=result.returncode,passed=33 if ok else None,wall_seconds=time.perf_counter()-wall,raw_log_sha256=sha(log),native_acceptance=False,production_release=False,actual_final_evidence=False)
 with record.open('x')as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 fsyncdir(record.parent);need(ok,'Linux CPU fixtures failed; original log/record retained');print(json.dumps(value))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--execute-worker',action='store_true');p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha');a=p.parse_args()
 need(not sys.flags.optimize and a.source_commit==SOURCE,'Explicit fixed source required');files,m=package(a.package)
 if a.execute_worker:
  need(a.review is not None and a.expected_review_sha is not None,'Independent package review required');execute(a,files,m)
 else:print(json.dumps(dict(scope='LOCAL_PACKAGE_PRECHECK_ONLY_NO_EXECUTION',source_commit=SOURCE,payload_files=13,archive_members=14,native_acceptance=False)))
