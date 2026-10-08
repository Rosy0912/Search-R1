import pathlib,subprocess,os,hashlib,json,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric')
source='1cceda2f7e201de3f68c876bb9a3b895563c20ce'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo',text=True).strip()==source
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root/'repo',text=True).strip()
(root/'tools/cpu-tests.py').write_text('"""Run repository tests without installed packages shadowing its namespace."""\nfrom pathlib import Path\nimport sys\nimport os\nimport types\nimport pytest\nroot = Path.cwd()\n# CPU tests assert default composition and use local loopback for Gloo ranks.\nos.environ.pop("SEARCH_R1_ASSETS", None)\nos.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"\nfor name, directory in [(\'tests\', root/\'tests\'), (\'tests.baselines\', root/\'tests/baselines\')]:\n    module = types.ModuleType(name)\n    module.__path__ = [str(directory)]\n    sys.modules[name] = module\nif __name__ == \'__main__\':\n    raise SystemExit(pytest.main([\'-q\', \'tests/baselines\', \'tests/reproduction\']))\n')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONPATH=str(root/'repo')+':'+str(root/'repo/tests/baselines'))
start=time.time()
with (root/'logs/environment-acceptance.json').open('w') as f:
 subprocess.run([str(root/'env/bin/python'),'reproduction/h20/check_environment.py'],cwd=root/'repo',env=env,stdout=f,check=True)
with (root/'logs/cpu-full-numeric.txt').open('w') as f:
 subprocess.run([str(root/'env/bin/python'),str(root/'tools/cpu-tests.py')],cwd=root/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
text=(root/'logs/cpu-full-numeric.txt').read_text()
assert '183 passed' in text and 'skipped' not in text, text[-2000:]
(root/'logs/cpu-gate-source.txt').write_text(source+'\n')
record=dict(source_commit=source,wall_seconds=time.time()-start,tests_sha256=hashlib.sha256((root/'logs/cpu-full-numeric.txt').read_bytes()).hexdigest(),environment_sha256=hashlib.sha256((root/'logs/environment-acceptance.json').read_bytes()).hexdigest(),cuda_visible_devices='',scope='TARGET_LINUX_CPU_ONLY')
(root/'logs/cpu-gate-numeric.json').write_text(json.dumps(record,indent=2)+'\n')
(root/'logs/environment.ready').touch();(root/'logs/cpu-gate.ready').touch()
print(json.dumps(record));print(text[-500:])
