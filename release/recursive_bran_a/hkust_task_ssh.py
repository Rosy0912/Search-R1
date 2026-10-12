"""Task-only HKUST SSH connections, no global VPN/route/auth edits."""
import argparse,concurrent.futures,json,os,re,shlex,subprocess,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);a=p.parse_args()
receipt=Path(a.receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
if receipt.exists():root=Path(json.loads(receipt.read_text())['private_dir'])
else:root=Path(tempfile.mkdtemp(prefix='rba-hkust-',dir='/tmp'))
assert root.stat().st_uid==os.getuid() and root.stat().st_mode&0o777==0o700
hosts=['hkust1','hkust2','hkust3'];expected={'hkust1':'xielab1','hkust2':'xielab2','hkust3':'xielab3'}
common=['-oBatchMode=yes','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=3']
def run(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=25)
 return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
def connect(host):
 sock=root/host;check=run(['ssh','-S',str(sock),'-O','check',host]);start=None
 if check['returncode']:
  options=['-oProxyJump=none']
  if host=='hkust3':
   proxy=shlex.join(['ssh','-S',str(root/'hkust1'),'-oControlMaster=no','-oBatchMode=yes','-W','%h:%p','hkust1'])
   # ProxyCommand must be the first proxy option: earlier ProxyJump=none
   # would suppress it in OpenSSH's first-value-wins parsing.
   options=['-oProxyCommand='+proxy]
  else:options+=['-oProxyCommand=none','-oBindInterface=en0']
  start=run(['ssh','-fN','-M','-S',str(sock),'-oControlPersist=3600',*common,*options,host])
  if start['returncode']:return dict(host=host,accepted=False,start=start)
  check=run(['ssh','-S',str(sock),'-O','check',host])
 probes=[run(['ssh','-S',str(sock),'-oControlMaster=no',*common,host,'hostname']) for _ in range(2)]
 accepted=check['returncode']==0 and all(x['returncode']==0 and x['stdout']==expected[host] for x in probes)
 return dict(host=host,accepted=accepted,socket=str(sock),check=check,start=start,probes=probes,
 route='direct en0/no ProxyCommand' if host!='hkust3' else 'existing HKUST1 direct master to campus HKUST3; no Boost proxy')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:first=list(pool.map(connect,hosts[:2]))
third=connect('hkust3') if first[0]['accepted'] else dict(host='hkust3',accepted=False,reason='HKUST1 campus transport unavailable')
proof=dict(time=time.time(),private_dir=str(root),hosts=first+[third],global_network_or_auth_modified=False,
 limits='Actual current connectivity only, not a guarantee against future campus/Wi-Fi loss; reconnect task socket after check failure')
receipt.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
raise SystemExit(0 if all(x['accepted'] for x in proof['hosts']) else 1)
