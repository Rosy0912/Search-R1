"""Parent-operated exclusive separate-code install and CUDA-hidden Linux25 only."""
import argparse,hashlib,io,json,os,re,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
TOOLS_SHA='fbd0098b946e23a8eb7304912cad01b8f339e249f627c919ddec8313c2337859'
DEST=ROOT/'first5-evidence-tools-3315-v3'
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def package(path,expected):
 need(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'Trusted exact package SHA required')
 blob=Path(path).read_bytes();need(hashlib.sha256(blob).hexdigest()==expected,'Package bytes changed')
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:')as t:
  ms=t.getmembers();need(len(ms)==10 and len({m.name for m in ms})==10 and all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in ms),'Exact10 ordinary unique package members')
  files={m.name:t.extractfile(m).read()for m in ms}
 manifest=json.loads(files['package-manifest.json']);need(manifest['source_commit']==SOURCE and len(manifest['files'])==9 and set(files)==set(manifest['files'])|{'package-manifest.json'},'Exact9 payload plus manifest required')
 for n,h in manifest['files'].items():need(hashlib.sha256(files[n]).hexdigest()==h,'Package member hash changed '+n)
 return files,manifest
def execute(args,files,manifest):
 need(not sys.flags.optimize and sys.platform=='linux' and os.uname().nodename=='h100-5' and sys.prefix==str(ROOT/'env'),'Actual ownenv target Linux CPU required')
 need(ROOT.resolve()==ROOT and not ROOT.is_symlink()and not DEST.exists()and not DEST.is_symlink(),'New canonical owned code directory required')
 review=Path(args.review);need(review.is_absolute()and ROOT in review.parents and review.resolve()==review and review.is_file() and type(args.expected_review_sha)is str and re.fullmatch('[0-9a-f]{64}',args.expected_review_sha) and sha(review)==args.expected_review_sha,'Exact independent code review bytes required')
 r=json.loads(review.read_text());need(r.get('status')=='PASS_CODE_ONLY_3315_FIRST5_V3_PORTABLE_PACKAGE_REVIEW' and r.get('package_manifest_sha256')==hashlib.sha256(files['package-manifest.json']).hexdigest() and r.get('mustfix')==[] and r.get('native_acceptance')is False and r.get('source_commit')==SOURCE and r.get('job_id')==3315,'Unresolved/wrong independent review')
 need(all(r.get('files',{}).get(n)==h for n,h in manifest['files'].items()),'Review must bind all9 payload bytes')
 need(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'Frozen clean repo required')
 need(sha(ROOT/'logs/tools-manifest.json')==TOOLS_SHA,'Frozen production21 manifest required')
 DEST.mkdir(mode=0o700);(DEST/'tests').mkdir();(DEST/'reference').mkdir()
 for n,raw in files.items():
  with (DEST/n).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 for d in (DEST,DEST/'tests',DEST/'reference',ROOT):
  fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
 log=ROOT/'logs/first5-v3-linux25-tests.actual.log';record=ROOT/'logs/first5-v3-linux25-tests.actual.json';need(not record.exists(),'Existing CPU record cannot be replaced')
 env=dict(os.environ,PYTHONPATH=str(ROOT/'repo'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1');env.pop('PYTHONOPTIMIZE',None)
 with log.open('xb')as f:
  result=subprocess.run([str(ROOT/'env/bin/python'),'-B','-m','unittest','discover','-s',str(DEST/'tests'),'-v'],env=env,stdout=f,stderr=subprocess.STDOUT);f.flush();os.fsync(f.fileno())
 raw=log.read_text();ok=result.returncode==0 and re.search(r'^Ran 25 tests in [\d.]+s$',raw,re.M) and re.search(r'^OK$',raw,re.M) and 'skipped='not in raw
 value=dict(scope='ACTUAL_LINUX25_CPU_TOOL_SCHEMA_NOT_NATIVE_FIRST5',source_commit=SOURCE,job_id=3315,root=str(ROOT),code_directory=str(DEST),package_sha256=args.expected_package_sha,review_sha256=args.expected_review_sha,host=os.uname().nodename,sys_prefix=sys.prefix,interpreter=str(ROOT/'env/bin/python'),exit_code=result.returncode,passed=25 if ok else None,raw_log_sha256=sha(log),native_acceptance=False,production_release=False)
 with record.open('x')as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(record.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 need(ok,'Actual Linux25 failed; raw/record preserved, no overwrite/retry');print(json.dumps(value))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--expected-package-sha',required=True);p.add_argument('--source-commit',required=True);p.add_argument('--execute-worker',action='store_true');p.add_argument('--review',type=Path);p.add_argument('--expected-review-sha');a=p.parse_args()
 need(not sys.flags.optimize and a.source_commit==SOURCE,'Explicit exact source commit required');files,manifest=package(a.package,a.expected_package_sha)
 if a.execute_worker:
  need(a.review is not None,'Independent package code review required');execute(a,files,manifest)
 else:print(json.dumps(dict(scope='LOCAL_EXACT_SELF_CONTAINED_PACKAGE_PRECHECK_ONLY',files=10,source_commit=SOURCE,mutated=False,native_acceptance=False)))
