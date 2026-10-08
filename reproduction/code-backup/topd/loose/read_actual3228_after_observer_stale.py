import pathlib,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import subprocess,json,time,pathlib,hashlib
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-grpo100-common-v5-r1');o={'epoch':time.time(),'scope':'READ_ONLY_SAME_ACTUAL3228_AFTER_STALE_OBSERVER'}
for name,args in [('scontrol',['scontrol','-o','show','job','3228']),('queue',['squeue','-h','-j','3228','-o','%i|%T|%M'])]:
 r=subprocess.run(args,capture_output=True,text=True,timeout=15);o[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
p=R/'actual-outer.json'
if p.exists():
 raw=p.read_bytes();o['actual_outer']={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)}
o['fault_result_files']=[str(p) for p in (R/'evidence').glob('*') if ('fault' in p.name or 'result' in p.name)]
o['checkpoint5_exists']=(R/'checkpoints/actor/global_step_5').exists();print(json.dumps(o))
'''
r=send(source,attempts=3,retry_safe=True,timeout=90);pathlib.Path('outputs/standard3228-after-stale-readonly.json').write_bytes(r.stdout+r.stderr);
if r.returncode==0:
 import json
 d=json.loads(r.stdout);pathlib.Path('outputs/remote-evidence/standard3228-after-stale-'+str(int(d['epoch']*1000000))+'-original-response.json').write_bytes(r.stdout);print(json.dumps({'epoch':d['epoch'],'queue':d['queue'],'actual_completed':d.get('actual_outer',{}).get('data',{}).get('actual_completed_outer_steps'),'fault_result_files':d.get('fault_result_files'),'checkpoint5_exists':d.get('checkpoint5_exists')}))
else:print(r.stderr.decode())
sys.exit(r.returncode)
