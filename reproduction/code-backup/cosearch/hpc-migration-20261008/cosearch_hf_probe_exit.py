"""Probe additional isolated remote exits; token stdin/RAM, no shared changes."""
import pathlib,subprocess,shlex
with open('/dev/tty') as f:token=f.readline().strip()
code=r'''import pathlib,yaml,json,socket,subprocess,time,requests,os,shutil,sys,hashlib
source=yaml.safe_load(pathlib.Path('/home/shaoshitong/project/clash-for-linux/runtime/config.yaml').read_text());token=sys.stdin.readline().strip();nodes=[x for x in source.get('proxies',[]) if x.get('server')];results=[]
for i in range(2,min(8,len(nodes))):
 node=nodes[i];port=18400+i;s=socket.socket()
 try:s.bind(('127.0.0.1',port))
 except OSError:continue
 finally:s.close()
 root=pathlib.Path('/dev/shm/cosearch-exit-probe-'+str(port));root.mkdir(mode=0o700,exist_ok=False);ram=root/'config.yaml';cfg={'port':port,'allow-lan':False,'bind-address':'127.0.0.1','mode':'rule','log-level':'silent','proxies':[node],'proxy-groups':[],'rules':['MATCH,'+node['name']]};fd=os.open(str(ram),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:yaml.safe_dump(cfg,f)
 p=subprocess.Popen(['/home/shaoshitong/project/clash-for-linux/runtime/bin/mihomo','-f',str(ram),'-d',str(root)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);ok=False
 try:
  time.sleep(.5);s=requests.Session();s.trust_env=False;t=time.time()
  try:
   response=s.get('https://huggingface.co/Sutoonq/TMP_MODEL/resolve/220f71043c64b4e5663769be6f2b96a89075a036/cosearch-migration-20261009/archives/retriever-supplement.tar.zst',headers={'Authorization':'Bearer '+token,'Range':'bytes=0-1048575'},proxies={'https':'http://127.0.0.1:'+str(port)},timeout=(5,10));ok=response.status_code==206 and len(response.content)==1048576 and hashlib.sha256(response.content).hexdigest()=='1e909124b54f672d6260664b0905708c8b3280fe0c3c96e0911b985d970f8ad8';results.append({'node_index':i,'http':response.status_code,'valid_sample':ok,'seconds':time.time()-t})
  except Exception as e:results.append({'node_index':i,'error_type':type(e).__name__})
  print(json.dumps(results[-1]),flush=True)
  if ok:
   task=pathlib.Path('/data/shaoshitong/cosearch-hf-migration-20261009');(task/'own-exit.json').write_text(json.dumps({'pid':p.pid,'port':port,'ram_config':str(ram),'node_index':i,'epoch':time.time(),'mac_used':False}));(task/'additional-exit-probe.json').write_text(json.dumps(results));break
 finally:
  if not ok:
   p.terminate()
   try:p.wait(timeout=2)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   shutil.rmtree(root)
'''
args=['ssh','-T','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust2']
r=subprocess.run(args+['python3 -c '+shlex.quote(code)],input=token+chr(10),capture_output=True,text=True,timeout=115)
print(r.stdout);print('exit',r.returncode)
pathlib.Path('outputs/evidence/native-x-hpc-hf-additional-exit-probe-20261009.json').write_text(r.stdout)
