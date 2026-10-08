"""Verify a faster same-HKUST TLS route before replacing this task's memory-pipe stream."""
import hashlib,json,os,secrets,shlex,signal,subprocess,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE='/home/shaoshitong/search-e1-archive100-eval-20261008'
SOURCE='/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/assets/archive100'
SRC_TASK='/data/shaoshitong/search-e1-archive100-transfer-20261008-'+uuid.uuid4().hex[:8]
SRC_IP='10.120.17.101';PORT=18443

def ssh(host,cmd):return ['ssh','-T','-o','ProxyCommand=none','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4',host,cmd]
def remote(host,code,payload,timeout=45):
 r=subprocess.run(ssh(host,'python3 -c '+shlex.quote(code)),input=json.dumps(payload),capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(f'{host} remote operation failed exit {r.returncode}: '+r.stderr[-500:])
 return json.loads(r.stdout.strip().splitlines()[-1])
def write(name,value):
 p=HERE/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def stop_source(launch):
 code='''import json,os,signal,sys\nfrom pathlib import Path\nv=json.load(sys.stdin);p=Path('/proc')/str(v['pid'])\nif p.exists():\n c=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode();assert v['script'] in c\n os.kill(v['pid'],signal.SIGTERM)\nprint(json.dumps({'stopped':True}))\n'''
 return remote('hkust1',code,launch)

TOKEN=secrets.token_urlsafe(48)
receipt=json.loads((HERE.parent/'search-e1-archive100-hf-verified-receipt-20261007.json').read_text())
source_bootstrap='''import json,os,sys,subprocess,time,hashlib\nfrom pathlib import Path\nv=json.load(sys.stdin);os.umask(0o077);b=Path(v['base']);assert not b.exists();b.mkdir(mode=0o700)\n(b/'direct_https_source.py').write_text(v['script']);(b/'private-config.json').write_text(json.dumps(v['config']));(b/'manifest.json').write_text(json.dumps(v['manifest']))\nr=subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(b/'source.key'),'-out',str(b/'source.crt'),'-days','1','-subj','/CN=HKUST-task-private-transfer','-addext','subjectAltName=IP:'+v['config']['bind_ip']],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);assert r.returncode==0,'Temporary certificate creation failed'\nlog=(b/'source.log').open('ab');p=subprocess.Popen(['python3','-B',str(b/'direct_https_source.py')],cwd=str(b),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\ntime.sleep(1);assert p.poll() is None,'Task service failed to bind'\nlaunch={'pid':p.pid,'script':str(b/'direct_https_source.py'),'base':str(b),'port':v['config']['port'],'started_epoch':time.time(),'cert_sha256':hashlib.sha256((b/'source.crt').read_bytes()).hexdigest()}\n(b/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps({'launch':launch,'certificate':(b/'source.crt').read_text()}))\n'''
v=remote('hkust1',source_bootstrap,{'base':SRC_TASK,'script':(HERE/'direct_https_source.py').read_text(),'manifest':receipt,'config':{'token':TOKEN,'source':SOURCE,'bind_ip':SRC_IP,'port':PORT,'lifetime_seconds':7200}})
source_launch=v['launch'];certificate=v['certificate']
write('direct-source-launch.json',source_launch)
(HERE/'direct-source-public.crt').write_text(certificate)
print(json.dumps({'stage':'source_TLS_ready','pid':source_launch['pid'],'port':PORT}),flush=True)
config={'token':TOKEN,'url':f'https://{SRC_IP}:{PORT}'}
test='''import json,sys,ssl,urllib.request,time,hashlib\nv=json.load(sys.stdin);ctx=ssl.create_default_context(cadata=v['certificate']);h={'Authorization':'Bearer '+v['config']['token']};url=v['config']['url']\ndef req(path,headers=None):return urllib.request.urlopen(urllib.request.Request(url+path,headers={**h,**(headers or {})}),context=ctx,timeout=25)\nwith req('/health') as r:assert json.load(r)=={'ready':True,'files':12}\nstart=time.time();size=8*1024*1024;sha=hashlib.sha256();n=0\nwith req('/file/model-00001-of-00003.safetensors',{'Range':'bytes=0-'+str(size-1)}) as r:\n assert r.status==206\n while True:\n  chunk=r.read(1024*1024)\n  if not chunk:break\n  sha.update(chunk);n+=len(chunk)\nelapsed=time.time()-start\nwith req('/prefix/model-00001-of-00003.safetensors?length='+str(size)) as r:prefix=json.load(r)\nassert n==size and sha.hexdigest()==prefix['sha256']\nprint(json.dumps({'tls_verified':True,'prefix_sha_verified':True,'bytes':n,'seconds':elapsed,'MiB_per_s':n/elapsed/1024**2,'finished_epoch':time.time()}))\n'''
try:
 proof=remote('hkust5',test,{'certificate':certificate,'config':config},timeout=40)
 write('direct-route-speedtest.json',proof);print(json.dumps({'stage':'direct_TLS_route_verified',**proof}),flush=True)
 assert proof['MiB_per_s']>=8,'Direct route is not demonstrably faster; preserve old stream'
except Exception:
 stop_source(source_launch);raise

# Read and verify process identity before signaling only this task's original stream.
old=json.loads((HERE/'controller-launch.json').read_text());oldpid=int(old['pid'])
ps=subprocess.run(['ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True)
rows={}
for line in ps.stdout.splitlines():
 x=line.strip().split(None,2)
 if len(x)==3:rows[int(x[0])]={'ppid':int(x[1]),'command':x[2]}
assert oldpid in rows and str(HERE/'launch_controller.py') in rows[oldpid]['command'],'Original controller identity changed; preserve all streams'
children=[pid for pid,row in rows.items() if row['ppid']==oldpid]
for pid in children:
 command=rows[pid]['command'];assert command.startswith('ssh ') and ('hkust1' in command or 'hkust5' in command) and ('tar' in command) and (SOURCE in command or BASE in command),'Unexpected original controller child; preserve all processes'
stopproof={'checked_epoch':time.time(),'controller_pid':oldpid,'controller_command':rows[oldpid]['command'],'ssh_children':[{ 'pid':pid,'command':rows[pid]['command']} for pid in children],'reason':'Authenticated pinned-TLS same-cluster route verified faster; preserve partial and SHA-prefix resume'}
write('direct-switch-process-proof.json',stopproof)
os.kill(oldpid,signal.SIGTERM)
for pid in children:
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(1)
for pid in [oldpid,*children]:
 r=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
 assert not r.stdout.strip(), 'Original stream did not stop; do not create duplicate receiver'
print(json.dumps({'stage':'old_owned_memory_stream_stopped','controller':oldpid,'ssh_children':children}),flush=True)

# Confirm the remote tar writer and every task-partial write FD have exited before a new receiver.
writer_probe=(HERE/'inspect_target_writers.py').read_text()
for attempt in range(20):
 writer_proof=remote('hkust5',writer_probe,{})
 if not writer_proof['tar_processes'] and not writer_proof['write_fds'] and not any(p['task_related'] for p in writer_proof['unreadable_owned_processes']):break
 time.sleep(.5)
else:raise RuntimeError('Target task still has an old writer or uninspectable owned process; do not create a second receiver')
write('direct-target-writer-gate.json',writer_proof)

receive_bootstrap='''import json,sys,os,subprocess,time\nfrom pathlib import Path\nv=json.load(sys.stdin);b=Path(v['base']);os.umask(0o077);assert b.is_dir();assert not (b/'worker-launch.json').exists();assert not (b/'direct-transfer-launch.json').exists()\n(b/'direct_https_receive.py').write_text(v['script']);(b/'direct-private-config.json').write_text(json.dumps(v['config']));(b/'direct-source.crt').write_text(v['certificate'])\nlog=(b/'direct-transfer.log').open('ab');p=subprocess.Popen(['python3','-B',str(b/'direct_https_receive.py')],cwd=str(b),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nlaunch={'pid':p.pid,'script':str(b/'direct_https_receive.py'),'launched_epoch':time.time(),'route':'HKUST1 -> authenticated pinned-TLS -> HKUST5','concurrency':3}\n(b/'direct-transfer-launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch))\n'''
launch=remote('hkust5',receive_bootstrap,{'base':BASE,'script':(HERE/'direct_https_receive.py').read_text(),'config':config,'certificate':certificate})
write('direct-target-launch.json',launch)
write('controller-status.json',{'phase':'direct_same_cluster_transfer','observed_epoch':time.time(),'remote_receiver':launch,'source_server':source_launch,'weight_bytes_written_on_mac':0,'speedtest':proof})
print(json.dumps({'stage':'direct_receiver_launched','launch':launch,'weight_bytes_written_on_mac':0}),flush=True)
