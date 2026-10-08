"""Authorized read-only actual native source/import/config review; no GPU/job."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,shlex
H=Path(__file__).resolve().parent;B=H.parent
p=argparse.ArgumentParser();p.add_argument('--inventory-sha256',required=True);a=p.parse_args();sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
new=B/'native-source-inventory-a0968c2d24be-ACTUAL.json';old=B/'native-source-inventory-89beac273f78-ACTUAL.json'
assert sha(new)==a.inventory_sha256 and sha(old)=='1170bfd11f9df324506f64add056551248f96d2748cac25412b86e45d3846f22'
z=json.loads(new.read_text())['modules'];o=json.loads(old.read_text())['modules'];assert set(z)==set(o) and len(z)==210
libraries={'hydra.main','torch.utils._contextlib','contextlib','dataclasses'};changed=[]
for name,rec in z.items():
 if name in libraries:assert rec==o[name]
 else:
  assert rec['path']==o[name]['path'].replace('/repo-89beac273f78/','/repo-a0968c2d24be/')
  if rec['sha256']!=o[name]['sha256']:changed.append(name)
assert changed==['search_r1.baselines.actor','search_r1.baselines.checkpoint_gate'],changed
assert z['search_r1.baselines.actor']['sha256']=='a9e5005f9d5e48f2cba652ee88b2ff2f3b0bb671dde8ce70ea7653eeb82c5ed9'
assert z['search_r1.baselines.checkpoint_gate']['sha256']=='74c3c13929c4f5cb784b90b7fd568ee80dfe1d37c8efaa28f97aae60b3492632'
assert z['reproduction.baselines.e1_import_bootstrap']['sha256']=='82433ba203fa8b8ed02cc3c4d0be4d0d0ceb7de096ebe657561f1fd726194164'
code=r'''
import sys,os,pathlib,json,hashlib,subprocess,socket
sys.dont_write_bytecode=True
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004');repo=root/'repo-a0968c2d24be';source='a0968c2d24beeaa0400f60e4e34677fe6deef9c9';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname()=='h100-10' and os.getuid()==0 and os.environ['CUDA_VISIBLE_DEVICES']=='' and repo.resolve()==repo
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
p=repo/'reproduction/baselines/e1_import_bootstrap.py';assert sha(p)=='82433ba203fa8b8ed02cc3c4d0be4d0d0ceb7de096ebe657561f1fd726194164'
raw=(root/'native-source-inventory-a0968c2d24be.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==PIN
sys.path.insert(0,str(repo));from reproduction.baselines import e1_import_bootstrap as checker
assert pathlib.Path(checker.__file__).resolve()==p
import torch
assert torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized()
actual,count=checker.verify_imports(repo,checker.unique_json(raw));assert not torch.cuda.is_initialized()
q=subprocess.run([sys.executable,str(repo/'reproduction/baselines/e1-300/cpu_hydra_entry_regression.py')],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo)),timeout=20)
assert q.returncode==0,q.stderr;config=json.loads(q.stdout);assert config['cuda_is_initialized'] is False and config['repo']==str(repo)
c=config['resolved'];assert c['trainer']['total_training_steps']==300 and c['actor_rollout_ref']['actor']['optim']['lr_warmup_steps']==57
assert c['data']['train_batch_size']==512 and c['actor_rollout_ref']['rollout']['n_agent']==5 and c['actor_rollout_ref']['actor']['distillation']['component_clip'] is None and c['actor_rollout_ref']['actor']['kl_loss_coef']==0 and c['actor_rollout_ref']['actor']['baseline_method']=='opsd'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip();assert not torch.cuda.is_initialized()
print(json.dumps({'status':'INDEPENDENT_NATIVE_CPU_IMPORT_AND_IMPORTED_HYDRA_CONFIG_PASS_NOT_GPU','node':socket.gethostname(),'uid':os.getuid(),'pid':os.getpid(),'torch':torch.__version__,'source_commit':source,'source_clean_before_after':True,'inventory_sha256':PIN,'inventory_records':210,'checker_sha256':sha(p),'actual_loaded_modules':actual,'actual_module_count':len(actual['modules']),'verified_function_count':count,'hydra_child_stdout_sha256':hashlib.sha256(q.stdout.encode()).hexdigest(),'hydra_child_stderr':q.stderr,'hydra':config,'cuda_visible_devices':'','cuda_is_initialized':False,'scope':'Actualnative CPU fullrequired import/livecode and importedmain configcomposition only; no asset/model/taskbody/GPU/job/first5/10 acceptance'},sort_keys=True))
'''.replace('PIN',repr(a.inventory_sha256))
command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/repo-a0968c2d24be /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python -c '+shlex.quote(code)
r=subprocess.run([sys.executable,str(B.parent/'remote.py'),command],capture_output=True,timeout=70)
with (H/'native-original.json').open('xb') as f:f.write(r.stdout)
with (H/'native-stderr.txt').open('xb') as f:f.write(r.stderr)
assert r.returncode==0,r.stderr.decode();v=json.loads(r.stdout)
with (H/'local-inventory-delta.json').open('x') as f:json.dump({'inventory_records':210,'only_changed_modules':changed,'native_libraries_identical':True,'source_checker_unchanged':True,'actual_new_inventory_sha256':a.inventory_sha256},f,indent=2)
print(json.dumps({'passed_native_cpu':True,'actual_modules':v['actual_module_count'],'actual_functions':v['verified_function_count'],'native_original_sha256':sha(H/'native-original.json')}))
