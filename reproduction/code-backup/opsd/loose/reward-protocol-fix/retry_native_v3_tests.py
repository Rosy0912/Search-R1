from pathlib import Path
import subprocess,json,hashlib,os
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');review=root/'tools/corrected-integration-review-9fb';src=review/'source-v3';parent=root/'repo';extract=review/'v3-files';m=extract/'candidate-manifest.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()=="40adc6398d1df94a95a1fa68b4b62a9e2d7a3e0a43e0738ec924dd9f8d20c036"
manifest=json.loads(m.read_text())
for name,sha in manifest["file_sources"].items():
 rel=name.split("integration-candidate-9fb-v3/",1)[1];p=(src/rel[5:]) if rel.startswith("repo/") else extract/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',SEARCH_R1_REQUIRE_REAL_DATAPROTO='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',PYTHONPATH=str(src)+":"+str(root/"tools/pytest-lib"))
python=str(root/'env/bin/python'); selections=['tests/reproduction/test_corrected_evaluation.py','tests/reproduction/test_corrected_runtime_capture.py','tests/reproduction/test_reward_and_validation.py','tests/reproduction/test_terminal_answer_protocol.py','tests/baselines'];results=[]
for title,cwd,args in [('repo',src,selections),('tools',extract/'tools',['test_strict_stage_prerequisites.py','test_pre_stage_runtime.py','test_hf_capture.py'])]:
 log=review/(title+'-v3-native-cpu-retry.txt')
 with log.open('wb') as f:r=subprocess.run([python,'-m','pytest','-q',*args],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(scope=title,exit_code=r.returncode,log=str(log),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),tail=log.read_text()[-2200:]))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent,text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a'
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=parent,text=True).strip()==''
print(json.dumps(dict(candidate_manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),results=results,parent_training_source_unchanged=True)))
