from pathlib import Path
import subprocess,json,hashlib,tarfile,os
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd'); review=root/'tools/corrected-integration-review-9fb'; src=review/'source-v3'; parent=root/'repo'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent,text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a'
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=parent,text=True).strip()==''
assert not src.exists()
subprocess.run(['git','clone','--shared','--no-checkout',str(parent),str(src)],check=True)
subprocess.run(['git','fetch',str(review/'base.bundle'),'reproduce/opsd'],cwd=src,check=True)
subprocess.run(['git','checkout','--detach','9fb3a74c3249a0a7a688701ce4439764605a9e1f'],cwd=src,check=True)
archive=review/'v3-files.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='26d353a4b429f9b334c62a67ad40a46e629ddc91e7980a1261e6203102ea38ce'
extract=review/'v3-files';extract.mkdir()
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 tar.extractall(extract)
m=extract/'candidate-manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='40adc6398d1df94a95a1fa68b4b62a9e2d7a3e0a43e0738ec924dd9f8d20c036'
manifest=json.loads(m.read_text());import shutil
for name,sha in manifest['file_sources'].items():
 rel=name.split('integration-candidate-9fb-v3/',1)[1];p=extract/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 if rel.startswith('repo/'):
  dest=src/rel[5:];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SEARCH_R1_REQUIRE_REAL_DATAPROTO='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(src))
python=str(root/'env/bin/python'); selections=['tests/reproduction/test_corrected_evaluation.py','tests/reproduction/test_corrected_runtime_capture.py','tests/reproduction/test_reward_and_validation.py','tests/reproduction/test_terminal_answer_protocol.py','tests/baselines'];results=[]
for title,cwd,args in [('repo',src,selections),('tools',extract/'tools',['test_strict_stage_prerequisites.py','test_pre_stage_runtime.py','test_hf_capture.py'])]:
 log=review/(title+'-v3-native-cpu.txt')
 with log.open('wb') as f:r=subprocess.run([python,'-m','pytest','-q',*args],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(scope=title,exit_code=r.returncode,log=str(log),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),tail=log.read_text()[-2200:]))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent,text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a'
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=parent,text=True).strip()==''
print(json.dumps(dict(candidate_manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),results=results,parent_training_source_unchanged=True)))
