import os,json,pathlib,subprocess,time
r=pathlib.Path('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012');logs=r/'logs'; result={'epoch':time.time(),'df':subprocess.getoutput('df -B1 /mnt/data')}
xs=[]
for p in logs.glob('formal-inline-optimizer-step-*-rank-*.json'):
 try:x=json.loads(p.read_text());xs.append(x)
 except:pass
if xs:
 step=max(x['logical_update'] for x in xs);result['step']=step;result['ranks']=[{'rank':x['rank'],'writes':x['optimizer_writes'],'finite':x['all_parameters_and_adam_finite']} for x in xs if x['logical_update']==step]
p=logs/'trainer.log'
with p.open('rb') as f:f.seek(max(0,p.stat().st_size-15000));result['trainer_tail']=f.read().decode(errors='replace')
result['failure_files']=[str(p) for p in logs.glob('*failure*')]
print(json.dumps(result))
