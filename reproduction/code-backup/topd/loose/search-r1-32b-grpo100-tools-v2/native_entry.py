"""One exact-source no-GRES CPU contract run; no bytecode mutation before imports."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,os,pathlib,subprocess,time
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def verify(root,manifest_name):
 root=pathlib.Path(root);manifest=root/manifest_name;records=json.loads(manifest.read_text())['files'];actual=set()
 for directory,dirs,files in os.walk(root):
  for name in dirs+files:
   if pathlib.Path(directory,name).is_symlink():raise ValueError('Source alias')
  actual.update(str(pathlib.Path(directory,name).relative_to(root)) for name in files)
 if actual!=set(records)|{manifest_name}:raise ValueError('Source exact file set differs')
 for name,r in records.items():
  p=root/name
  if p.stat().st_size!=r['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Source bytes differ '+name)
 return hashlib.sha256(manifest.read_bytes()).hexdigest()
def main():
 source=pathlib.Path(os.environ['R1_SOURCE']);result_root=pathlib.Path(os.environ['R1_CPU_RESULT']);result_root.mkdir(mode=0o700,exist_ok=False)
 shas={'source':verify(source,'r1-files.json'),'tools':verify(HERE,'files.json')}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(HERE)+':'+str(ROOT/'test-tools'))
 tests=[str(HERE/'test_standard_r1.py'),str(HERE/'test_final_cleanup.py')]
 cmd=[str(ROOT/'env/bin/python'),'-m','pytest','-q',*tests,'--basetemp',str(result_root/'tmp'),'-o','cache_dir='+str(result_root/'cache')]
 with open(result_root/'pytest.log','x') as log:code=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);log.flush();os.fsync(log.fileno())
 text=(result_root/'pytest.log').read_text()
 from pipeline import atomic
 expected=31
 if code==0 and (str(expected)+' passed') not in text:code=1
 record={'scope':'STANDARD_R1_SOURCE_CPU_CONTRACTS_ONLY_NO_GPU_PROOF','job':os.environ.get('SLURM_JOB_ID'),'returncode':code,'expected_count':expected,'zero_skip':'skipped' not in text,'source_manifests':shas,'log_sha256':hashlib.sha256((result_root/'pytest.log').read_bytes()).hexdigest(),'completed_epoch':time.time(),'cuda_visible_devices':''}
 if not record['zero_skip']:code=1;record['returncode']=code
 atomic(result_root/'result.json',record);print(json.dumps(record));sys.exit(code)
if __name__=='__main__':main()
