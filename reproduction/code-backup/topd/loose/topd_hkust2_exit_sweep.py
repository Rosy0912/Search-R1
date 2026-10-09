"""Probe isolated HKUST2 exits from the user's existing config, RAM only."""
import json,pathlib,shlex,subprocess,sys
code=r'''
import concurrent.futures,json,os,pathlib,subprocess,sys,time,yaml,socket
token=sys.stdin.readline().strip();source=yaml.safe_load(pathlib.Path('/dev/shm/topd-hkust2-network.yaml').read_text())
nodes=source['proxies'];wanted=[n['name'] for n in nodes if '公告' not in n['name']]
def run(case):
 i,name=case;port=18116+i
 for p in (port,port+1000):
  s=socket.socket();s.bind(('127.0.0.1',p));s.close()
 node=next(x for x in nodes if x['name']==name)
 config={'port':port,'allow-lan':False,'bind-address':'127.0.0.1','mode':'rule','log-level':'silent','external-controller':'127.0.0.1:'+str(port+1000),'proxies':[node],'proxy-groups':[],'rules':['MATCH,'+name]}
 ram=pathlib.Path('/dev/shm/topd-hkust2-sweep-'+str(port)+'.yaml');fd=os.open(str(ram),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:yaml.safe_dump(config,f,allow_unicode=True)
 taskdir=pathlib.Path('/dev/shm/topd-hkust2-sweep-'+str(port));taskdir.mkdir(mode=0o700)
 # MATCH-only rules do not require GeoIP databases.
 p=subprocess.Popen(['/home/shaoshitong/project/clash-for-linux/runtime/bin/mihomo','-f',str(ram),'-d',str(taskdir)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 try:
  time.sleep(1);t=time.time()
  cfg='url = "https://hf-mirror.com/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst"\nheader = "Authorization: Bearer '+token+'"\nrange = "0-1048575"\n'
  env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
  r=subprocess.run(['curl','-4','--proxy','http://127.0.0.1:'+str(port),'-L','-s','-o','/dev/null','--connect-timeout','5','--max-time','10','-w','%{http_code} %{size_download} %{speed_download} %{time_starttransfer}','--config','-'],input=cfg,text=True,capture_output=True,timeout=14,env=env)
  return {'node':name,'exit':r.returncode,'metrics':r.stdout,'seconds':time.time()-t,'mac_used':False}
 finally:
  if p.poll() is None:
   p.terminate()
   try:p.wait(timeout=4)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  ram.unlink(missing_ok=True)
  import shutil
  shutil.rmtree(taskdir)
with concurrent.futures.ThreadPoolExecutor(6) as pool:out=list(pool.map(run,enumerate(wanted)))
root=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009');root.joinpath('own-exit-probe.json').write_text(json.dumps(out,ensure_ascii=False));print(json.dumps(out,ensure_ascii=False))
'''
token=sys.stdin.readline().strip()
p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=token+'\n',capture_output=True,text=True,timeout=100)
out=json.loads(p.stdout) if p.returncode==0 else {'error':'remote_failed','exit':p.returncode}
pathlib.Path('outputs/hpc3-migration/hkust2-sweep-probe.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps(out,ensure_ascii=False))
