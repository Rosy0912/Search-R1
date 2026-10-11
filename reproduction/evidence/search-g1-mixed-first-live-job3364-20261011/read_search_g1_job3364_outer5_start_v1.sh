/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,subprocess,datetime\nbase=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011')\nr=json.loads((base/'logs/run-manifest.json').read_text())\no={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':r['run_id'],'actor_local_micro':r['actor_micro_batch_per_rank'],'global_micro':r['budget']['ppo_micro_batch_size'],'vllm_utilization':r['vllm_gpu_memory_utilization'],'gradient_accumulation':128//8//r['actor_micro_batch_per_rank'],'budget':{k:r['budget'][k] for k in ['world_size','train_batch_size','n_agent','trajectory_count','ppo_mini_batch_size','optimizer_opportunities_per_outer','compute_dtype']}}\nfor name in ['resolved-training-config.json','gpu-inventory-preflight.json','gpu-preflight.json']:\n p=base/'logs'/name\n if p.exists():\n  d=json.loads(p.read_text())\n  if name=='resolved-training-config.json':\n   c=d['resolved_config']['actor_rollout_ref'];d={'actor':{k:c['actor'].get(k) for k in ['ppo_micro_batch_size','ppo_mini_batch_size','ppo_epochs','compute_dtype']},'rollout_utilization':c['rollout']['gpu_memory_utilization']}\n  o[name]=d\np=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)\no['gpu_memory']={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}\no['pool_files']=[{'attempt':p.name,'files':{f.name:f.stat().st_size for f in p.iterdir() if f.is_file()},'post_rng_rank_files':len(list(p.glob('*after-generation-worker-rng/rank-*.pt')))} for p in (base/'artifacts/mixed-candidate-pools').glob('*') if p.is_dir()]\np=base/'logs/formal-inline-development-before.json'\nif p.exists():\n d=json.loads(p.read_text());o['baseline']={k:d[k] for k in ['step','weighted_em','weighted_f1','question_count','qa_count','scope','n_questions'] if k in d}\np=base/'logs'/('gpu-samples-'+r['run_id']+'.jsonl')\npeak={}\nif p.exists():\n for line in p.read_text().splitlines():\n  for g in json.loads(line)['gpus']:peak[str(g['index'])]=max(peak.get(str(g['index']),0),g['used_mib'])\no['sampled_gpu_peak_mib']=peak\nprint(json.dumps(o))\n"
worker_cmd='env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /usr/bin/python3 -E -B -S -c '+shlex.quote(observer)
mgmt=r'''import json,shlex,subprocess,datetime
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-8';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if ip!=gateway:args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
r=subprocess.run(args+[user+'@'+ip,WORKER_CMD],capture_output=True,text=True,timeout=65)
print(json.dumps({'route':'retained18 SSH-W h100-8','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''.replace('WORKER_CMD',repr(worker_cmd))
r=subprocess.run(['env','-u','LD_LIBRARY_PATH','/usr/bin/ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host']+['env -u LD_LIBRARY_PATH /usr/bin/python3 -c '+shlex.quote(mgmt)],capture_output=True,text=True,timeout=85)
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3364-ACTUAL-PROGRESS-MEMORY-20261011-outer5-start-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY
