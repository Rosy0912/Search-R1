"""Launch isolated remote reprobes after production adopts the first fast exit."""
import base64,json,pathlib,shlex,subprocess,sys
source=pathlib.Path('work/topd_remote_speed_optimizer.py').read_text()
source=source.replace("ROOT=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009')", "ROOT=pathlib.Path('/data/shaoshitong/topd-hf-route-reprobe-20261009')").replace('19897','19898').replace('17897','17898').replace('/dev/shm/topd-hkust2-fresh-speed.yaml','/dev/shm/topd-hkust2-reprobe-speed.yaml')
encoded=base64.b64encode(source.encode()).decode()
remote=r'''
import base64,fcntl,json,os,pathlib,socket,subprocess,sys,time,yaml
token=sys.stdin.readline().strip();assert token
root=pathlib.Path('/data/shaoshitong/topd-hf-route-reprobe-20261009');root.mkdir(exist_ok=True)
lock=(root/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name in ['speed-optimizer-controller.json','fresh-exit-controller.json']:
 p=root/name
 if p.exists():
  c=json.loads(p.read_text());proc=pathlib.Path('/proc/'+str(c['pid'])+'/cmdline')
  if proc.exists():raise RuntimeError('Existing reprobe component; inspect before launch')
for port in [17898,19898]:
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
config=yaml.safe_load(pathlib.Path('/dev/shm/topd-hkust2-fresh-speed.yaml').read_text());config.update({'port':17898,'external-controller':'127.0.0.1:19898','bind-address':'127.0.0.1','allow-lan':False,'mixed-port':0,'socks-port':0,'secret':''})
ram=pathlib.Path('/dev/shm/topd-hkust2-reprobe-speed.yaml');ram.write_text(yaml.safe_dump(config,allow_unicode=True));ram.chmod(0o600)
runtime=pathlib.Path('/dev/shm/topd-hkust2-reprobe-runtime');runtime.mkdir(exist_ok=True);runtime.chmod(0o700)
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
exitproc=subprocess.Popen(['/home/shaoshitong/project/clash-for-linux/runtime/bin/mihomo','-d',str(runtime),'-f',str(ram)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,start_new_session=True)
(root/'fresh-exit-controller.json').write_text(json.dumps({'pid':exitproc.pid,'epoch':time.time(),'mac_used':False,'port':17898,'controller_port':19898}))
for _ in range(30):
 with socket.socket() as sock:ready=sock.connect_ex(('127.0.0.1',19898))==0
 if ready:break
 time.sleep(.2)
else:raise RuntimeError('Isolated controller not ready')
script=root/'topd_remote_speed_optimizer.py';script.write_bytes(base64.b64decode(DATA))
p=subprocess.Popen(['python3',str(script)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env,start_new_session=True);p.stdin.write((token+'\n').encode());p.stdin.close();time.sleep(.5);assert p.poll() is None
print(json.dumps({'epoch':time.time(),'optimizer_pid':p.pid,'exit_pid':exitproc.pid,'root':str(root),'port':17898,'production_17897_untouched':True,'mac_used':False}))
'''.replace('DATA',repr(encoded))
token=sys.stdin.readline().strip();assert token
p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(remote)],input=token+'\n',capture_output=True,text=True,timeout=20)
if p.returncode:print(json.dumps({'error':'remote_reprobe_launch_failed','exit':p.returncode}));sys.exit(1)
d=json.loads(p.stdout);pathlib.Path('outputs/hpc3-migration/speed-reprobe-launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
