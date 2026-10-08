import sys, subprocess, shlex, pathlib, json
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
code=r'''import pathlib,subprocess,json,socket,datetime,os
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006');run=root/'runs/gigpo-e1-fresh200-live-job3323-1791355362986358231'
out={'hostname':socket.gethostname(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'3323','run':str(run)}
for n,c in [('scheduler',['scontrol','show','job','3323','-o']),('queue',['squeue','-h','-w','h100-8','-o','%i %j %T']),('gpu',['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])]:
 a=subprocess.run(c,capture_output=True,text=True,timeout=20);out[n]={'returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr}
out['files']=[{'name':x.name,'bytes':x.stat().st_size,'mtime':x.stat().st_mtime} for x in run.iterdir() if x.is_file()] if run.exists() else []
out['metadata']={}
for n in ['live-progress.json','run-result.json','driver-result.json','probability-identity-ready.json','probability-identity-admitted.json','live-first5-production-ready.json']:
 p=run/n
 if p.is_file():out['metadata'][n]=json.loads(p.read_bytes())
out['logs']={}
for n,p in [('slurm',root/'logs/fresh200-live-3323.log'),('calibration',run/'calibration.log'),('retrieval',run/'retriever.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-18000));out['logs'][n]=f.read().decode(errors='replace')
print(json.dumps(out))
'''
inner='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(code)
mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 -o HostName=10.33.0.16 -o HostKeyAlias=100.108.173.4 -o StrictHostKeyChecking=yes h100-16 '+shlex.quote(mid)]
a=subprocess.run(route,capture_output=True,text=True,timeout=60)
p=pathlib.Path('outputs/gigpo-e1-instruct200/JOB3323-ACTUAL-STATUS-'+sys.argv[1]+'.json')
with p.open('x') as f:f.write(a.stdout if a.returncode==0 else json.dumps({'returncode':a.returncode,'stderr':a.stderr}))
if a.returncode:print(p.read_text());raise SystemExit(a.returncode)
o=json.loads(a.stdout)
print(json.dumps({k:o[k] for k in ['utc','hostname','job_id','files','metadata']},ensure_ascii=True))
print('queue '+o['queue']['stdout'])
for k,v in o['logs'].items():print(k+'\n'+v[-9000:])
