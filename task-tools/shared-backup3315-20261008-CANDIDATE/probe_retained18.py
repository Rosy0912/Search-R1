"""Actual read of own shared probe on retained18; existing identity and strict SSH."""
import argparse
import datetime
import json
import shlex
import subprocess
from pathlib import Path

REMOTE = '''import datetime,hashlib,json,socket,subprocess,sys,os,stat
from pathlib import Path
if sys.flags.optimize: raise RuntimeError('Optimized Python forbidden for identity checks')
base=Path('/mnt/shared/bapo-artifacts/formal3315-60f566b3-20261008')
assert base.resolve()==base and not any(p.is_symlink() for p in [base,*base.parents])
mount=json.loads(subprocess.check_output(['findmnt','-J','-T',str(base)],text=True))['filesystems']
assert len(mount)==1 and mount[0]['target']=='/mnt/shared' and mount[0]['source']=='10.32.0.8:/mnt/data/sihe-share5' and mount[0]['fstype']=='nfs4'
def read_leaf(name):
 p=base/name
 assert stat.S_ISREG(p.lstat().st_mode), 'Regular actual NFS leaf required'
 fd=os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  assert stat.S_ISREG(os.fstat(f.fileno()).st_mode) and os.fstat(f.fileno()).st_dev==base.stat().st_dev
  return f.read()
raw=read_leaf('probe.json');probe=json.loads(raw)
prepared=read_leaf('prepared.json');inventory=read_leaf('inventory.json')
assert socket.gethostname()=='h100-10' and type(probe['job_id'])is int and probe['job_id']==3315 and probe['source']=='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2' and probe['nonce']=='bab8981d9fd14afc98923eb4ebb89e33'
print(json.dumps({'target':'h100-10','hostname':socket.gethostname(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe':probe,'probe_sha256':hashlib.sha256(raw).hexdigest(),'mount':mount[0],'prepared_sha256':hashlib.sha256(prepared).hexdigest(),'inventory_sha256':hashlib.sha256(inventory).hexdigest()}))
'''


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    command = shlex.join(['env', '-u', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'python3', '-B', '-c', REMOTE])
    code = '''import json,shlex,subprocess,datetime
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target='h100-10';ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
args=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
assert ip==gateway, 'retained18 expected gateway'
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
''' + 'command=' + repr(command) + '''
r=subprocess.run(args+[user+'@'+ip,command],capture_output=True,text=True,timeout=30)
print(json.dumps({'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'management_observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=15', '-o', 'ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p', 'mgmt-host', 'env -u LD_LIBRARY_PATH python3 -']
    r = subprocess.run(ssh, input=code, capture_output=True, text=True, timeout=55)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + '.transport-record.json').write_text(json.dumps({'exit': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
    if r.returncode:
        raise RuntimeError('management SSH failed; original error retained')
    d = json.loads(r.stdout)
    Path(str(args.output) + '.management-record.json').write_text(json.dumps(d, indent=2))
    if d['exit_code'] != 0:
        raise RuntimeError('retained18 read failed; original error retained')
    proof = json.loads(d['stdout'])
    if proof['target'] != 'h100-10' or proof['hostname'] != 'h100-10':
        raise RuntimeError('target mismatch')
    with args.output.open('x') as f:
        json.dump(proof, f, indent=2); f.write('\n')
    print(json.dumps({'output': str(args.output), 'actual_target': proof['hostname'], 'observed_utc': proof['observed_utc']}))


if __name__ == '__main__':
    main()
