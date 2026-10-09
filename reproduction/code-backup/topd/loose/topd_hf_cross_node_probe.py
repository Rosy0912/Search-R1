"""Same-range HF benchmark on campus nodes; all model bytes stay in remote RAM."""
import concurrent.futures,json,pathlib,shlex,subprocess,sys
secret=sys.stdin.readline().strip()
code=r'''import concurrent.futures,requests,sys,time,json,hashlib,pathlib,socket
secret=sys.stdin.readline().strip();node=NODE;size=16777216
def test(mode):
 s=requests.Session();s.trust_env=False
 if mode=='official_remote_exit':s.proxies={'https':'http://127.0.0.1:17894'}
 host='https://hf-mirror.com' if mode=='mirror_direct' else 'https://huggingface.co'
 t=time.time();n=0;status=None;err=None;ttfb=None;h=hashlib.sha256()
 try:
  with s.get(host+'/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst',headers={'Authorization':'Bearer '+secret,'Range':'bytes=0-'+str(size-1)},stream=True,timeout=(6,8)) as r:
   status=r.status_code
   if status==206:
    for b in r.iter_content(65536):
     if ttfb is None:ttfb=time.time()-t
     n+=len(b);h.update(b)
     if n>=size or time.time()-t>22:break
 except Exception as e:err=type(e).__name__
 return {'epoch':time.time(),'node':node,'mode':mode,'status':status,'bytes':n,'seconds':time.time()-t,'first_byte_seconds':ttfb,'sha256':h.hexdigest() if n else None,'error':err,'mac_data_used':False,'mac_HF_proxy_used':False}
modes=['official_direct','mirror_direct']+(['official_remote_exit'] if node=='hkust2' else [])
with concurrent.futures.ThreadPoolExecutor(3) as e:out=list(e.map(test,modes))
p=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009');p.mkdir(exist_ok=True);p.joinpath('latest.json').write_text(json.dumps(out));print(json.dumps(out))
'''
def test(node):
 remote=code.replace('NODE',repr(node));p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes',node,'python3 -c '+shlex.quote(remote)],input=secret+chr(10),capture_output=True,text=True,timeout=40)
 if p.returncode:return {'node':node,'error':'ssh_failed','exit':p.returncode}
 try:return json.loads(p.stdout)
 except:return {'node':node,'error':'invalid_probe_response'}
with concurrent.futures.ThreadPoolExecutor(2) as e:
 out=list(e.map(test,['hkust1','hkust2']))
pathlib.Path('outputs/hpc3-migration/cross-node-probe-latest.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
