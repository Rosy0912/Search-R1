import pathlib,json,subprocess,datetime
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-resume5-8860aab-20261011')
assert (base/'preparation/asset-binding.json').is_file()
assert not (base/'preparation/native-submit-intent.json').exists() and not (base/'preparation/native-submit-receipt.json').exists()
p=subprocess.run([str(base/'env/bin/python'),'-B','-m','reproduction.common8.submit_mixed_native','--base',str(base),'--source-commit','8860aabe91cf28b5cc74fd99c09c7cf2efc13a86','--source-manifest',str(base/'preparation/published-source-files.json'),'--restore-receipt','/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/diagnostics/checkpoint5-cpu-validation-once-v1/RESULT.json'],cwd=base/'repo',env={'PATH':'/usr/local/bin:/usr/bin:/bin','PYTHONPATH':str(base/'repo'),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},capture_output=True,text=True,timeout=45)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
assert p.returncode==0
