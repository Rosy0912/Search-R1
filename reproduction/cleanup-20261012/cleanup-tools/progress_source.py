import pathlib,json,re,time,subprocess
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/runs/tree-branpo-fresh200-3360');f=r/'stage.log'
with f.open('rb') as h:h.seek(max(0,f.stat().st_size-500000));text=h.read().decode(errors='replace')
nums=re.findall(r'(?<![a-zA-Z_])step:(\d+)',text);o={'epoch':time.time(),'step':max(map(int,nums)) if nums else None,'log_bytes':f.stat().st_size,'rolling':[p.name for p in (r/'checkpoints/training_state/rolling-completed').iterdir()],'tail':text[-5000:],'queue':subprocess.check_output(['squeue','-h','-j','3360','-o','%i|%T|%N'],text=True)}
print(json.dumps(o))
