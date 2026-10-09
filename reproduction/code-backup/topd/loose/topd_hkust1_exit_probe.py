"""Probe isolated HKUST1 exits from the user's existing config, RAM only."""
import json,pathlib,shlex,subprocess,sys
code=r'''
import concurrent.futures,json,os,pathlib,subprocess,sys,time,yaml,socket
token=sys.stdin.readline().strip();source=yaml.safe_load(pathlib.Path('/home/shaoshitong/project/clash-for-linux/conf/config.yaml').read_text())
nodes=source['proxies'];wanted=['美國洛杉矶10GB','美国加州[CUB]','香港PCCW01','日本[CM]','香港精品','美國西雅圖']
def run(case):
 i,name=case;port=17916+i
 for p in (port,port+1000):
  s=socket.socket();s.bind(('127.0.0.1',p));s.close()
 node=next(x for x in nodes if x['name']==name)
 config={'port':port,'allow-lan':False,'bind-address':'127.0.0.1','mode':'rule','log-level':'silent','external-controller':'127.0.0.1:'+str(port+1000),'proxies':[node],'proxy-groups':[],'rules':['MATCH,'+name]}
 ram=pathlib.Path('/dev/shm/topd-hkust1-speed-'+str(port)+'.yaml');fd=os.open(str(ram),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:yaml.safe_dump(config,f,allow_unicode=True)
 taskdir=pathlib.Path('/dev/shm/topd-hkust1-speed-'+str(port));taskdir.mkdir(mode=0o700)
 (taskdir/'Country.mmdb').symlink_to('/home/shaoshitong/project/clash-for-linux/conf/Country.mmdb')
 p=subprocess.Popen(['/home/shaoshitong/project/clash-for-linux/bin/clash-linux-amd64','-f',str(ram),'-d',str(taskdir)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
 try:
  time.sleep(1);t=time.time()
  cfg='url = "https://huggingface.co/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst"\nheader = "Authorization: Bearer '+token+'"\nrange = "0-8388607"\n'
  env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
  r=subprocess.run(['curl','-4','--proxy','http://127.0.0.1:'+str(port),'-L','-s','-o','/dev/null','--connect-timeout','5','--max-time','18','-w','%{http_code} %{size_download} %{speed_download} %{time_starttransfer}','--config','-'],input=cfg,text=True,capture_output=True,timeout=22,env=env)
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
p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust1','python3 -c '+shlex.quote(code)],input=token+'\n',capture_output=True,text=True,timeout=40)
out=json.loads(p.stdout) if p.returncode==0 else {'error':'remote_failed','exit':p.returncode}
pathlib.Path('outputs/hpc3-migration/hkust1-exit-probe.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps(out,ensure_ascii=False))
