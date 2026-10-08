import pathlib,subprocess,shlex,json,hashlib,io,tarfile
H=pathlib.Path(__file__).resolve().parent;L=H.parent/'opsd-e1-300';R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004';old='89beac273f783d8e48d7d3e34da040f753514d90';new=subprocess.check_output(['git','-C',str(L),'rev-parse','HEAD'],text=True).strip();assert new!=old and not subprocess.check_output(['git','-C',str(L),'status','--porcelain'],text=True).strip()
REPO=R+'/repo-'+new[:12];bundle=H/('source-'+new[:12]+'.bundle');subprocess.run(['git','-C',str(L),'bundle','create',str(bundle),'HEAD','^'+old],check=True)
ns={'__file__':str(H/'deploy_formal_reviewed.py')};s=pathlib.Path(ns['__file__']).read_text();exec(s[:s.index('m=json.loads')],ns);base_remote=ns['remote']
def remote(c,data=None):
 try:return base_remote(c,data)
 except subprocess.CalledProcessError as e:
  (H/'source-a0968c2-transfer-error-original.txt').write_bytes(e.stdout+b'\n'+e.stderr);print(e.stderr.decode());raise RuntimeError('Remote operation failed; original stderr saved') from None
p=R+'/tools/'+bundle.name
pin=hashlib.sha256(bundle.read_bytes()).hexdigest()
code=f'''import pathlib,subprocess,hashlib,os,json
r=pathlib.Path({REPO!r});b=pathlib.Path({p!r});assert not r.exists();assert hashlib.sha256(b.read_bytes()).hexdigest()=={pin!r};assert b.stat().st_uid==0
subprocess.run(['git','clone','--no-hardlinks','--no-checkout',{R+'/repo-'+old[:12]!r},str(r)],check=True)
subprocess.run(['git','-C',str(r),'fetch',str(b),{new!r}],check=True);subprocess.run(['git','-C',str(r),'checkout','--detach',{new!r}],check=True)
assert subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip()=={new!r};assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip()
print(json.dumps({{'source_commit':{new!r},'source_repo':str(r),'bundle_sha256':{pin!r},'uid':os.getuid()}}))
'''
r=remote('python3 -c '+shlex.quote(code));(H/'source-a0968c2-native-deployment.json').write_bytes(r.stdout)
producer=R+'/tools/native-closure-1f9012d41c33/native_import_closure_producer.py';out=R+'/native-source-inventory-'+new[:12]+'.json';native='/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
code=f'''import pathlib,subprocess,os,hashlib,json
r=pathlib.Path({REPO!r});env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(r),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
p=pathlib.Path({producer!r});assert hashlib.sha256(p.read_bytes()).hexdigest()=='1f9012d41c33f1caed1f231cf2a811bce35547fcd54ff7501eaa25d3924893e1'
q=subprocess.run([{native!r},str(p),'--repo',str(r),'--checker-sha256','82433ba203fa8b8ed02cc3c4d0be4d0d0ceb7de096ebe657561f1fd726194164','--output',{out!r}],env=env,capture_output=True);print(q.stdout.decode());print(q.stderr.decode(),file=__import__('sys').stderr);assert q.returncode==0
h=subprocess.run([{native!r},str(r/'reproduction/baselines/e1-300/cpu_hydra_entry_regression.py')],env=env,capture_output=True);(pathlib.Path({R!r})/('native-hydra-entry-'+{new[:12]!r}+'.json')).write_bytes(h.stdout);print(h.stdout.decode());print(h.stderr.decode(),file=__import__('sys').stderr);assert h.returncode==0
assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip();print(json.dumps({{'inventory_path':{out!r},'inventory_sha256':hashlib.sha256(pathlib.Path({out!r}).read_bytes()).hexdigest(),'source_commit':{new!r},'source_repo':str(r),'records':len(json.loads(pathlib.Path({out!r}).read_bytes())['modules']),'scope':'NativeCPU imported-main config path and source inventory only'}}))
'''
r=remote('python3 -c '+shlex.quote(code));(H/'source-a0968c2-native-cpu-stdout.txt').write_bytes(r.stdout);(H/'source-a0968c2-native-cpu-stderr.txt').write_bytes(r.stderr);print(r.stdout.decode()[-1300:])
r=remote('cat '+shlex.quote(out));(H/('native-source-inventory-'+new[:12]+'-ACTUAL.json')).write_bytes(r.stdout)
