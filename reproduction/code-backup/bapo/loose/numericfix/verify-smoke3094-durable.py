import pathlib,subprocess,json,os,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric');p=root/'logs/smoke3094-actual-receipt-gate-durable.json';log=root/'logs/smoke3094-actual-receipt-gate-durable.log'
assert not p.exists() and not log.exists()
start=time.time()
with log.open('w') as f:r=subprocess.run([str(root/'env/bin/python'),str(root/'tools/verify_smoke_receipt-gpu64.py'),'3094'],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
p.write_text(json.dumps(dict(exit_code=r.returncode,wall_seconds=time.time()-start,job_id=3094,scope='actual fixed1cc smoke receipt gate'),indent=2)+'\n');print(p.read_text());print(log.read_text()[-300:]);raise SystemExit(r.returncode)
