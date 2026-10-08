from pathlib import Path
r=Path('work/rewardfix/remote')
root='/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097'
source='b09736d9d1b487a5c134e7a358761e04613cea32'
script='''import pathlib,subprocess,os,hashlib,json,time
root=pathlib.Path(ROOT)
source=SOURCE
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo',text=True).strip()==source
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root/'repo',text=True).strip()
(root/'tools/cpu-tests.py').write_text(CPU_TESTS)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH=str(root/'repo')+':'+str(root/'repo/tests/baselines'))
start=time.time()
with (root/'logs/environment-acceptance.json').open('w') as f:
 subprocess.run([str(root/'env/bin/python'),'reproduction/h20/check_environment.py'],cwd=root/'repo',env=env,stdout=f,check=True)
with (root/'logs/cpu-full-b097.txt').open('w') as f:
 subprocess.run([str(root/'env/bin/python'),str(root/'tools/cpu-tests.py')],cwd=root/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
text=(root/'logs/cpu-full-b097.txt').read_text()
assert '174 passed' in text and 'skipped' not in text, text[-2000:]
(root/'logs/cpu-gate-source.txt').write_text(source+'\\n')
record=dict(source_commit=source,wall_seconds=time.time()-start,tests_sha256=hashlib.sha256((root/'logs/cpu-full-b097.txt').read_bytes()).hexdigest(),environment_sha256=hashlib.sha256((root/'logs/environment-acceptance.json').read_bytes()).hexdigest(),cuda_visible_devices='',scope='TARGET_LINUX_CPU_ONLY')
(root/'logs/cpu-gate-b097.json').write_text(json.dumps(record,indent=2)+'\\n')
(root/'logs/environment.ready').touch();(root/'logs/cpu-gate.ready').touch()
print(json.dumps(record));print(text[-500:])
'''.replace('ROOT',repr(root)).replace('SOURCE',repr(source)).replace('CPU_TESTS',repr(Path('work/remote/cpu-tests.py').read_text()))
(r/'cpu-gate.py').write_text(script)
