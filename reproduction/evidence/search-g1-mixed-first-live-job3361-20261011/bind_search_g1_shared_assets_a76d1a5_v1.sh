/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer="import pathlib,json,datetime,os\nbase=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-a76d1a5-20261011')\nold=pathlib.Path('/mnt/data/shaoshitong/g1f-ef498ac3')\nassert (base/'repo/reproduction/common8/launch_native.sh').is_file()\nassert not (base/'logs/run-manifest.json').exists()\nlinks={'env':old/'env','bootstrap-tools':old/'bootstrap-tools','bootstrap-assets':old/'bootstrap-assets','assets':old/'assets','retriever-gpu-lib':old/'retriever-gpu-lib'}\nfor name,target in links.items():\n assert target.exists(), str(target)\n p=base/name\n assert not p.exists() and not p.is_symlink(), str(p)\n p.symlink_to(target.resolve(),target_is_directory=True)\nfor name in ['data','preparation','logs','tmp','cache','artifacts','checkpoints']:(base/name).mkdir(exist_ok=False)\nfor name in ['manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json']:\n target=old/'data'/name;assert target.is_file();(base/'data'/name).symlink_to(target.resolve())\ntarget=old/'preparation/official-model-manifest-original.json';assert target.is_file()\n(base/'preparation/official-model-manifest-original.json').write_bytes(target.read_bytes())\nproof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':str(base),'outputs_remote':str((base/'checkpoints').resolve()),'links':{name:str((base/name).resolve()) for name in links},'canonical_data_links':{p.name:str(p.resolve()) for p in (base/'data').iterdir()},'training_started':False,'shared_gpu_launch_settings':{'SEARCH_G1_ALLOW_EXISTING_GPU_PROCESSES':'1','SEARCH_G1_ROLLOUT_MEMORY_UTILIZATION':'0.2'},'source_sha_validation_required_before_launch':True}\nwith (base/'preparation/asset-binding.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())\nprint(json.dumps(proof))\n"
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
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-SHARED-ASSET-BINDING-20261011-v2.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':False},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY
