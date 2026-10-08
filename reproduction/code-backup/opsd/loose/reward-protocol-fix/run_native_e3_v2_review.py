from pathlib import Path
import hashlib,json,tarfile,subprocess,os
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');review=root/'tools/corrected-integration-review-9fb';src=review/'source-v3';dest=review/'e3-v2-tools';assert not dest.exists()
archive=review/'e3-v2.tar.gz'
with archive.open('xb') as f:
 for i in range(3):f.write((review/('e3-v2.part'+str(i))).read_bytes())
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='32c33e31b1b567dd5b31e7a56cf8246fcd8acc25132e219719cc7a7fca65ce5f';dest.mkdir()
with tarfile.open(archive) as t:
 for member in t.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 t.extractall(dest)
m=dest/'candidate-manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()=='027ec1f3256450f6bf32e5c48b60c4e582c8e279d1f1d33e71289b2d8d238434'
for name,sha in json.loads(m.read_text())['files'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha
layout=dest/'integration-candidate-9fb-v3';layout.mkdir();(layout/'repo').symlink_to(src,target_is_directory=True)
env=dict(os.environ,PYTHONPATH=str(src)+':'+str(root/'tools/pytest-lib'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',SEARCH_R1_REQUIRE_REAL_DATAPROTO='1',HF_HUB_OFFLINE='1');python=str(root/'env/bin/python');results=[]
for scope,cwd in [('new-eval',dest),('strict-hf-v6',dest/'strict-tools')]:
 log=review/('e3-v2-'+scope+'-native.txt')
 with log.open('wb') as f:r=subprocess.run([python,'-m','pytest','-q',*sorted(p.name for p in cwd.glob('test_*.py'))],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(scope=scope,returncode=r.returncode,log=str(log),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),tail=log.read_text()[-2200:]))
repo=root/'repo';assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='8e60976d364ab7d7ae51714415813c1359dafe9a';assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()==''
print(json.dumps(dict(candidate_manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),scope='native CPU software tests only; noactual500-HF/RayGPU',results=results,parent_training_source_unchanged=True)))
