from pathlib import Path
import io,tarfile,hashlib,json,os,sys,subprocess,time,re
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');S='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert os.uname().nodename=='h100-5'and sys.prefix==str(R/'env')and not sys.flags.optimize
assert subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()==S
assert not subprocess.check_output(['git','-C',str(R/'repo'),'status','--porcelain'],text=True).strip()
assert hashlib.sha256((R/'logs/tools-manifest.json').read_bytes()).hexdigest()=='fbd0098b946e23a8eb7304912cad01b8f339e249f627c919ddec8313c2337859'
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()=='8e6d66775e80fd0e6aa029a868f574fefde5645c696748ffaffa1c9cffb1e681'
D=R/'first5-evidence-tools-3315-v2';assert not D.exists();D.mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(b),mode='r:')as t:
 seen=set()
 for m in t.getmembers():
  p=Path(m.name);assert m.isfile()and not p.is_absolute()and '..'not in p.parts and m.name not in seen;seen.add(m.name)
  q=D/p;q.parent.mkdir(parents=True,exist_ok=True)
  with q.open('xb')as f:f.write(t.extractfile(m).read())
v=json.loads((D/'package-manifest.json').read_text());assert v['source_commit']==S and seen==set(v['files'])|{'package-manifest.json'}
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in v['files'].items())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(R/'repo'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('PYTHONOPTIMIZE',None)
p=R/'logs/first5-tools-3315-v2-target-tests.actual.log';started=time.monotonic()
with p.open('x')as f:x=subprocess.run([str(R/'env/bin/python'),'-B','-m','unittest','discover','-s',str(D/'tests'),'-v'],cwd=str(D),env=env,stdout=f,stderr=subprocess.STDOUT)
text=p.read_text();assert x.returncode==0 and re.search(r'Ran 16 tests',text)and text.rstrip().endswith('OK')and 'skipped='not in text
out={'source_commit':S,'job_id':3315,'package_sha256':hashlib.sha256(b).hexdigest(),'host':os.uname().nodename,'sys_prefix':sys.prefix,'passed':16,'skipped':0,'exit_code':x.returncode,'seconds':time.monotonic()-started,'raw_log_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'ACTUAL_LINUX_FIRST5_EVIDENCE_TOOL_TESTS_NOT_NATIVE_OR_RELEASE'}
with(R/'logs/first5-tools-3315-v2-target-tests.actual.json').open('x')as f:json.dump(out,f,indent=2)
print(json.dumps(out))
