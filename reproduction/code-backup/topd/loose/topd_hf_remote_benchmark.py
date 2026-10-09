import sys,subprocess,shlex
secret=input().strip()
code=r'''import pathlib,sys,json,requests,time,ssl,concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
token=sys.stdin.readline().strip()
class TLS12(HTTPAdapter):
 def init_poolmanager(self,*a,**kw):
  c=ssl.create_default_context();c.maximum_version=ssl.TLSVersion.TLSv1_2;kw['ssl_context']=c;return super().init_poolmanager(*a,**kw)
 def proxy_manager_for(self,proxy,**kw):
  c=ssl.create_default_context();c.maximum_version=ssl.TLSVersion.TLSv1_2;kw['ssl_context']=c;return super().proxy_manager_for(proxy,**kw)
def test(mode):
 s=requests.Session();s.trust_env=False
 if mode!='mirror_direct':s.proxies={'https':'http://127.0.0.1:17894'}
 if mode=='official_tls12':s.mount('https://',TLS12())
 endpoint='https://hf-mirror.com' if mode=='mirror_direct' else 'https://huggingface.co'
 t=time.time();n=0;status=None;error=None
 try:
  with s.get(endpoint+'/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/wiki_index.tar.zst',headers={'Authorization':'Bearer '+token,'Range':'bytes=200000000-208388607'},stream=True,timeout=(8,10)) as r:
   status=r.status_code
   if status==206:
    for b in r.iter_content(65536):
     n+=len(b)
     if n>=8388608 or time.time()-t>22:break
 except Exception as e:error=type(e).__name__
 return {'mode':mode,'status':status,'bytes':n,'seconds':time.time()-t,'error':error,'mac_used':False}
with concurrent.futures.ThreadPoolExecutor(3) as e:out=list(e.map(test,['official','official_tls12']))
print(json.dumps(out));pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009/tls-benchmark.json').write_text(json.dumps(out))
'''
r=subprocess.run(['ssh','-T','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=secret+'\n',capture_output=True,text=True,timeout=40);print(r.stdout);print('exit',r.returncode)
