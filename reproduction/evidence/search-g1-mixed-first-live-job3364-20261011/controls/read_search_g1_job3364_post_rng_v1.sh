/usr/bin/python3 - <<'PY'
import subprocess,shlex,json,datetime
from pathlib import Path
observer='import subprocess,json\np=subprocess.run([\'/mnt/data/shaoshitong/g1f-ef498ac3/env/bin/python\',\'-B\',\'-c\',"import pathlib,json,hashlib,torch,datetime\\nbase=pathlib.Path(\'/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011\')\\nrun=json.loads((base/\'logs/run-manifest.json\').read_text())\\npool=base/\'artifacts/mixed-candidate-pools/9045ac8bdf72476c9e976cc466ad5aa8\'\\np=pool/\'0-after-generation.pt\';state=torch.load(p,map_location=\'cpu\',weights_only=False)\\nproofs=state[\'worker_files\'];assert len(proofs)==8\\nrows=[]\\nfor proof in proofs:\\n f=pathlib.Path(proof[\'path\']);assert f.resolve().is_relative_to(pool.resolve())\\n digest=hashlib.sha256(f.read_bytes()).hexdigest();assert digest==proof[\'sha256\']\\n v=torch.load(f,map_location=\'cpu\',weights_only=False)\\n assert v[\'run_id\']==run[\'run_id\'] and v[\'source_commit\']==run[\'source_commit\'] and v[\'world_size\']==8\\n assert v[\'actual_optimizer_writes\']==0\\n rng=v[\'rng\'];assert rng[\'vllm_version\']==\'0.6.3\'\\n for key in [\'torch_cpu\',\'torch_cuda\',\'actor_cuda_saved\',\'rollout_cuda_saved\',\'numpy\',\'python\',\'request_generators\']:assert key in rng\\n rows.append({\'rank\':v[\'rank\'],\'pid\':v[\'process_identity\'][\'pid\'],\'bytes\':f.stat().st_size,\'sha256\':digest,\'actual_optimizer_writes\':v[\'actual_optimizer_writes\'],\'rng_keys\':sorted(rng),\'boundary\':rng[\'boundary\']})\\nassert sorted(row[\'rank\'] for row in rows)==list(range(8))\\nassert state[\'stream\'][\'pending_source\'] is None and state[\'stream\'][\'pending_offset\']==0\\nprint(json.dumps({\'utc\':datetime.datetime.now(datetime.timezone.utc).isoformat(),\'run_id\':run[\'run_id\'],\'source_commit\':run[\'source_commit\'],\'round\':1,\'completed_generation_boundary\':512,\'driver_file_sha256\':hashlib.sha256(p.read_bytes()).hexdigest(),\'driver_rng_keys\':sorted(state[\'driver_rng\']),\'stream_cursor\':state[\'stream\'][\'candidate_cursor\'],\'consumed_source_rows\':state[\'stream\'][\'consumed_source_rows\'],\'ranks\':rows,\'scope\':\'actual saved post-generation states independently CPU read and SHA checked; no model reload or new generation\'}))\\n"],env={\'PATH\':\'/usr/bin:/bin\',\'CUDA_VISIBLE_DEVICES\':\'\',\'OMP_NUM_THREADS\':\'2\'},capture_output=True,text=True,timeout=40)\nprint(json.dumps({\'rc\':p.returncode,\'stdout\':p.stdout,\'stderr\':p.stderr}))\n'
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
p=Path('outputs/gigpo-e1-instruct200/SEARCH-G1-JOB3364-ACTUAL-POST-GENERATION-RNG-20261011-v1.json')
with p.open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'read_only':True},f,ensure_ascii=False,indent=2);f.write('\n')
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
m=json.loads(r.stdout)
if m['returncode']:print(json.dumps(m));raise SystemExit(m['returncode'])
print(m['stdout'])
PY
