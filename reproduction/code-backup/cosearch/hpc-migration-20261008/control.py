#!/usr/bin/env python3
"""Local SSH controller. No dependencies; credentials never written to reports."""
import argparse, concurrent.futures, csv, datetime, html, io, json, os, shlex, subprocess, sys, tempfile, time
from pathlib import Path
BASE=Path(__file__).resolve().parent
try:
 NETWORK=json.loads((BASE/'ssh-network-routes.json').read_text())
except (OSError,ValueError):
 NETWORK={}
HOSTS=['jump_h20-1','hkust1','hkust2','hkust3','hkust4','hkust5','mgmt-host','baai-a800']
H100=['h100-4','h100-5','h100-7','h100-8','h100-9','h100-10','h100-16','h100-11']
try:
 CLUSTERS=json.loads((BASE/'cluster-access.json').read_text())
except (OSError,ValueError):
 CLUSTERS={}
try:
 H100_ACCESS=json.loads((BASE/'wuhan-access-route.json').read_text())
except (OSError,ValueError):
 H100_ACCESS={}
PROBE=r'''
import subprocess,json,socket,datetime,csv,io,os

def run(args):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=12)
  return r.stdout.strip() if r.returncode==0 else None
 except (OSError,subprocess.TimeoutExpired): return None
raw=run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'])
proc=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
uuids=run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'])
apps={}
for row in csv.reader(io.StringIO(proc or '')):
 if len(row)>=3:apps.setdefault(row[0].strip(),[]).append({'pid':row[1].strip(),'memory':row[2].strip()})
umap={r[0].strip():r[1].strip() for r in csv.reader(io.StringIO(uuids or '')) if len(r)>=2}
gpus=[]
def number(s):
 try:return int(s.strip())
 except ValueError:return None
for row in csv.reader(io.StringIO(raw or '')):
 if len(row)!=5:continue
 idx,name,util,used,total=[s.strip() for s in row]; util,used,total=map(number,[util,used,total])
 processes=apps.get(umap.get(idx),[])
 state='占用' if processes or (used is not None and used>2048) or (util is not None and util>0) else ('低占用候选' if used is not None and util is not None and proc is not None else '未知')
 gpus.append(dict(index=idx,name=name,util=util,used=used,total=total,state=state,processes=processes))
p=os.path.expanduser('~/.local/bin/codex')
version=run([p,'--version']) if os.path.exists(p) else None
print('CONTROL_JSON:'+json.dumps(dict(hostname=socket.gethostname(),gpus=gpus,gpu_query_ok=raw is not None,codex=version,disk=run(['df','-h','/']),sampled_at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''
def ssh(host):
 if host in CLUSTERS:
  return ['ssh','-T','-F',str(BASE/CLUSTERS[host]['ssh_config']),'-o','BatchMode=yes',host]
 args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout='+('30' if host=='jump_h20-1' else '10'),'-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2']
 # Respect the HKUST SOCKS route already defined in ~/.ssh/config.
 # Public SSH uses BoostNet explicitly: EasyConnect currently captures
 # its direct routes as well as the campus addresses.
 proxy='nc -X 5 -x 127.0.0.1:7892 %h %p'
 if host=='mgmt-host':
  args+=['-o','ProxyCommand=none','-o','ProxyJump=none']
 elif host=='baai-a800':
  # Public DNS verified 2026-10-03; retain the configured hostname for host-key checking.
  args+=['-o','ProxyCommand=nc -b '+shlex.quote(NETWORK.get('physical_interface','en0'))+' -w 10 '+shlex.quote(NETWORK.get('a800_ip','110.43.160.159'))+' %p']
 elif host=='jump_h20-1':
  # Literal gateway avoids outer SSH expanding nested %h/%p into the GPU target.
  gateway='nc -b '+shlex.quote(NETWORK.get('physical_interface','en0'))+' -w 30 120.92.211.161 2224'
  jump="ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o StrictHostKeyChecking=no -o "+shlex.quote('ProxyCommand='+gateway)+" -p 2224 'jzshi#root#115.191.22.237'@120.92.211.161 'ncat localhost 22'"
  args+=['-o','ProxyCommand='+jump]
 return args+[host]
def route(host,command):
 if host in HOSTS or host in CLUSTERS:return ssh(host)+[command]
 if host not in H100:raise ValueError('Unknown SSH route: '+host)
 if H100_ACCESS:
  gateway=H100_ACCESS['gateway_ip'];user=H100_ACCESS['user']
  ip=H100_ACCESS['targets'][host]
  prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',H100_ACCESS['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
  client=prefix.copy()
  if ip!=gateway:
   proxy=shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])
   client+=['-o','ProxyCommand='+proxy]
  public_keys=H100_ACCESS.get('trusted_target_host_keys',{}).get(host)
  if public_keys:
   client+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(public_keys)])]
  return ssh('mgmt-host')+[shlex.join(client+[user+'@'+ip,command])]
 # Only read-only monitoring uses the internal H100 routes.
 inner=command if host=='h100-16' else 'env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=8 '+shlex.quote(host)+' '+shlex.quote(command)
 head='env -u LD_LIBRARY_PATH ssh -T -o BatchMode=yes -o ConnectTimeout=8 h100-16 '+shlex.quote(inner)
 return ssh('mgmt-host')+[head]
def sample(host):
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  r=subprocess.run(route(host,'python3 -c '+shlex.quote(PROBE)),capture_output=True,text=True,timeout=55)
  lines=[x[len('CONTROL_JSON:'):] for x in r.stdout.splitlines() if x.startswith('CONTROL_JSON:')]
  if r.returncode or not lines:return dict(host=host,ok=False,checked_at=started,error='SSH/采集失败 (exit %s)'%r.returncode)
  d=json.loads(lines[-1]);d.update(host=host,ok=True,checked_at=started);return d
 except subprocess.TimeoutExpired:return dict(host=host,ok=False,checked_at=started,error='SSH/采集超时')
 except Exception as e:return dict(host=host,ok=False,checked_at=started,error=type(e).__name__)
def slurm():
 try:
  r=subprocess.run(route(H100_ACCESS.get('gateway_alias','h100-16'),'scontrol show nodes -o'),capture_output=True,text=True,timeout=35)
  data={}
  for line in r.stdout.splitlines():
   fields=dict(x.split('=',1) for x in line.split() if '=' in x)
   if 'NodeName' in fields:data[fields['NodeName']]=fields.get('State','未知')
  return data
 except subprocess.TimeoutExpired:return {}
def collect():
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(sample,h) for h in HOSTS+H100];sf=pool.submit(slurm)
  cf={c:pool.submit(cluster_sample,c) for c in CLUSTERS}
  rows=[f.result() for f in futures];states=sf.result()
  clusters={c:f.result() for c,f in cf.items()}
 for row in rows:
  if row['host'] in H100:
   row['slurm']=states.get(row['host'],'未知')
   row['management_ip']=H100_ACCESS.get('targets',{}).get(row['host'])
   row['release_date']=H100_ACCESS.get('retirement_schedule_from_user',{}).get(row['management_ip'])
   row['retained']=row['host']==H100_ACCESS.get('gateway_alias')
 data=dict(cycle_started_at=start,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=rows,slurm_clusters=clusters,h100_gateway=dict(host=H100_ACCESS.get('gateway_alias'),ip=H100_ACCESS.get('gateway_ip')))
 runtime=BASE/'runtime';runtime.mkdir(exist_ok=True)
 tmp=runtime/'status.json.tmp';tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(runtime/'status.json')
 def local_time(value):
  return datetime.datetime.fromisoformat(value).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')+' 北京时间'
 out=['# GPU 状态','', '更新时间：'+local_time(data['updated_at']),'', 'H100 路线：mgmt-host → '+str(data['h100_gateway']['ip'])+' → 目标节点。','低占用候选不是调度空闲证明；H100 请结合 Slurm 状态。失败节点没有沿用旧数据。','', '| 主机／管理IP／释放计划 | GPU | 型号 | 利用率% | 显存 MiB | 状态 | Slurm | 采样时间（北京） |','|---|---|---|---|---|---|---|---|']
 for h in rows:
  host_label=h['host']+(' / '+h['management_ip'] if h.get('management_ip') else '')+(' / '+h['release_date']+'释放' if h.get('release_date') else ' / 保留' if h.get('retained') else '')
  sampled=local_time(h.get('sampled_at',h['checked_at']))
  if not h['ok']:out.append('| %s | — | — | — | — | 未核实：%s | %s | %s（连接检查） |'%(host_label,h['error'],h.get('slurm','—'),sampled));continue
  if not h['gpus']:out.append('| %s | — | — | — | — | %s | %s | %s |'%(host_label,'无 GPU' if h['host']=='mgmt-host' else 'GPU 查询失败或无 GPU',h.get('slurm','—'),sampled))
  for g in h['gpus']:out.append('| %s | %s | %s | %s | %s / %s | %s | %s | %s |'%(host_label,g['index'],g['name'],g['util'] if g['util'] is not None else '未知',g['used'],g['total'],g['state'],h.get('slurm','—'),sampled))
 for record in clusters.values():
  import slurm_inventory
  out+=['',slurm_inventory.markdown(record)]
 tmp=runtime/'status.md.tmp';tmp.write_text('\n'.join(out)+'\n');tmp.replace(runtime/'status.md')
 for c,record in clusters.items():save_cluster(record)
 print('Updated '+data['updated_at'],flush=True)
 return data
def cluster_sample(cluster):
 import slurm_inventory
 return slurm_inventory.sample(cluster,ssh(cluster))
def save_cluster(record):
 import slurm_inventory
 runtime=BASE/'runtime';runtime.mkdir(exist_ok=True)
 for suffix,body in [('json',json.dumps(record,ensure_ascii=False,indent=2)),('md',slurm_inventory.markdown(record))]:
  dest=runtime/('cluster-'+record['cluster']+'.'+suffix)
  with tempfile.NamedTemporaryFile(mode='w',dir=runtime,prefix=dest.name+'.',suffix='.tmp',delete=False) as f:
   f.write(body);tmp=Path(f.name)
  tmp.replace(dest)
def sync_auth(host):
 source=Path.home()/'.codex/auth.json'
 remote="""import sys,os,datetime
from pathlib import Path
p=Path.home()/'.codex';p.mkdir(mode=0o700,exist_ok=True);p.chmod(0o700)
d=sys.stdin.buffer.read();a=p/'auth.json'
if a.exists():
 b=p/('auth.json.backup-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));b.write_bytes(a.read_bytes());b.chmod(0o600)
t=p/'auth.json.control-tmp'
f=os.open(str(t),os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(f,'wb') as out:out.write(d)
t.chmod(0o600);t.replace(a)
assert a.read_bytes()==d
print('AUTH_COPIED_AND_VERIFIED')
"""
 with source.open('rb') as f:return subprocess.call(ssh(host)+['python3 -c '+shlex.quote(remote)],stdin=f)
def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
 r=s.add_parser('run');r.add_argument('host',choices=HOSTS+list(CLUSTERS));r.add_argument('command',nargs=argparse.REMAINDER)
 e=s.add_parser('exec');e.add_argument('host',choices=HOSTS);e.add_argument('--cwd',required=True);e.add_argument('--prompt',required=True);e.add_argument('--write',action='store_true');e.add_argument('--timeout',type=int,default=180)
 a=s.add_parser('sync-auth');a.add_argument('host',choices=HOSTS)
 m=s.add_parser('monitor');m.add_argument('--watch',action='store_true');m.add_argument('--interval',type=int,default=15)
 c=s.add_parser('cluster');c.add_argument('cluster',choices=list(CLUSTERS))
 args=p.parse_args()
 if args.action=='cluster':
  record=cluster_sample(args.cluster);save_cluster(record)
  print(json.dumps(record,ensure_ascii=False,indent=2));return 0 if record['ok'] else 1
 if args.action=='sync-auth':return sync_auth(args.host)
 if args.action=='run':
  cmd=args.command[1:] if args.command[:1]==['--'] else args.command
  if not cmd:p.error('run requires a remote command')
  return subprocess.call(ssh(args.host)+[shlex.join(cmd)])
 if args.action=='exec':
  command=['timeout',str(args.timeout)+'s','codex','exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','--sandbox','workspace-write' if args.write else 'read-only','-C',args.cwd,'-m','gpt-6.1-sol',args.prompt]
  remote='export PATH="$HOME/.local/bin:$PATH"; '+shlex.join(command)
  return subprocess.call(ssh(args.host)+['bash -lc '+shlex.quote(remote)],stdin=subprocess.DEVNULL)
 if args.action=='monitor':
  while True:
   collect()
   if not args.watch:break
   time.sleep(max(5,args.interval))
 return 0
if __name__=='__main__':sys.exit(main())
