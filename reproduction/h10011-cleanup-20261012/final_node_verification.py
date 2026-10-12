import pathlib,json,time,subprocess
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');run=r/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359'
ranks=[run/('checkpoints/fresh200_state/outer_200_COMPLETE/ranks/rank_%d.pt'%i) for i in range(8)]
models=sorted((run/'checkpoints/actor/global_step_200').glob('model-*.safetensors'))
assert len(models)==3 and all(p.is_file() and p.stat().st_size>0 for p in ranks+models)
receipt=json.loads((r/'cleanup-receipt-20261012.json').read_text());assert all(not pathlib.Path(p).exists() for p in receipt['deleted'])
log=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/logs/slurm-3387.log')
print(json.dumps(dict(epoch=time.time(),df=subprocess.check_output(['df','-B1','/dataset'],text=True),queue=subprocess.check_output(['squeue','-h','-o','%i|%j|%T|%N'],text=True),latest200_ranks=[dict(path=str(p),size=p.stat().st_size) for p in ranks],latest200_models=[dict(path=str(p),size=p.stat().st_size) for p in models],source_exists=(r/'repo/verl').is_dir(),deleted_missing=True,rba_log_tail=subprocess.check_output(['tail','-c','6000',str(log)],text=True) if log.is_file() else None)))
