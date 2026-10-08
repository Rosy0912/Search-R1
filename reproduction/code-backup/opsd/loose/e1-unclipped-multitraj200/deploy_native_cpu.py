"""Produce and inspect new native CPU source closure, without a GPU context."""
import pathlib,json,hashlib,shlex,subprocess,datetime
from transport import remote
H=pathlib.Path(__file__).resolve().parent
spec=json.loads((H/'BLOCKED_BOUND_SPEC_FILENAME').read_text())
source=spec['source_commit'];repo=spec['remote_repo'];root=spec['remote_root']
old='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004'
producer=old+'/tools/native-closure-1f9012d41c33/native_import_closure_producer.py'
native='/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
output=root+'/native-source-inventory-'+source[:12]+'.json'
check=spec['source_files']['reproduction/baselines/e1_import_bootstrap.py']
code=f'''import pathlib,subprocess,os,json,hashlib,sys
repo=pathlib.Path({repo!r});p=pathlib.Path({producer!r});out=pathlib.Path({output!r})
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=={source!r}
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1f9012d41c33f1caed1f231cf2a811bce35547fcd54ff7501eaa25d3924893e1'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(repo),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPSD_E1_COMMON_DATA_ROOT={old+'/common-data'!r},OPSD_E1_COMMON_MANIFEST={old+'/common-data/manifest.json'!r},OPSD_E1_OFFICIAL_MODEL={old+'/assets/qwen2.5-3b-instruct-aa8e725'!r},OPSD_E1_RETRIEVER_URL='http://127.0.0.1:18084/retrieve',SEARCH_R1_BASELINE_RUN_ROOT={root+'/runs/unlaunched-native-cfg-only'!r},OPSD_E1_BINDING_MANIFEST={root+'/runs/unlaunched-native-cfg-only/bindings/manifest.json'!r},OPSD_E1_BINDING_MANIFEST_SHA256='UNLAUNCHED_CFG_ONLY_NO_BINDING',OPSD_E1_REVIEW_PUBLIC_KEY={root+'/review-signing/public.json'!r},OPSD_E1_REVIEW_PUBLIC_KEY_SHA256='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c')
if not out.exists():
 q=subprocess.run([{native!r},str(p),'--repo',str(repo),'--checker-sha256',{check!r},'--output',str(out)],env=env,capture_output=True)
 print(q.stdout.decode());print(q.stderr.decode(),file=sys.stderr);assert q.returncode==0
else:
 assert out.resolve(strict=True)==out and out.stat().st_uid==os.getuid()==0
 print(json.dumps(dict(existing_immutable_inventory_readonly=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
closure=json.loads(out.read_bytes());assert closure['modules']['reproduction.baselines.e1_import_bootstrap']['sha256']=={check!r}
verify="import sys,json,pathlib,torch;sys.path.insert(0,"+repr(str(repo))+");from reproduction.baselines import e1_import_bootstrap as b;assert not torch.cuda.is_initialized();a,n=b.verify_imports(pathlib.Path("+repr(str(repo))+"),json.loads(pathlib.Path("+repr(str(out))+").read_bytes()));assert not torch.cuda.is_initialized();print(json.dumps(dict(module_count=len(a['modules']),function_count=n,cuda_is_initialized=False,actual_loaded_modules=a),sort_keys=True))"
v=subprocess.run([{native!r},'-c',verify],env=env,cwd=str(repo),capture_output=True);print(v.stdout.decode());print(v.stderr.decode(),file=sys.stderr);assert v.returncode==0
c=subprocess.run([{native!r},str(repo/'reproduction/baselines/e1_import_bootstrap.py'),'--profile','multitraj200','--mode','cfg'],env=env,cwd=str(repo),capture_output=True);print(c.stdout.decode());print(c.stderr.decode(),file=sys.stderr);assert c.returncode==0
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
print(json.dumps(dict(source_commit={source!r},inventory_path=str(out),inventory_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),records=len(closure['modules']),scope='Native CPU imports and profile only, no GPU or fresh5 proof')))
'''
try:q=remote('python3 -c '+shlex.quote(code),timeout=180)
except subprocess.CalledProcessError as e:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 (H/('native-cpu-failed-'+stamp+'.stdout')).write_bytes(e.stdout);(H/('native-cpu-failed-'+stamp+'.stderr')).write_bytes(e.stderr);raise
(H/'native-cpu-original.stdout').write_bytes(q.stdout);(H/'native-cpu-original.stderr').write_bytes(q.stderr)
row=json.loads(q.stdout.decode().strip().splitlines()[-1]);(H/'native-cpu-actual.json').write_text(json.dumps(row,indent=2)+'\n')
raw=remote('cat '+shlex.quote(output)).stdout;assert hashlib.sha256(raw).hexdigest()==row['inventory_sha256']
(H/('native-source-inventory-'+source[:12]+'-ACTUAL.json')).write_bytes(raw)
print(json.dumps(row))
