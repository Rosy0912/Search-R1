"""Bounded private-archive probe of this task's own HKUST2 exit; token RAM only."""
import subprocess,sys,shlex
secret=sys.stdin.readline().strip()
code=r'''import sys,pathlib,yaml,requests,json,time
from urllib.parse import quote
token=sys.stdin.readline().strip();root=pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009');c=yaml.safe_load(pathlib.Path('/dev/shm/topd-hkust2-network.yaml').read_text());ctl=requests.Session();ctl.trust_env=False;ctl.headers['Authorization']='Bearer '+str(c.get('secret',''));base='http://127.0.0.1:19894';j=ctl.get(base+'/proxies',timeout=5).json()['proxies']['Ghelper'];previous=j['now'];names=[next(n for n in j['all'] if n==name) for name in ['美国圣何塞','🇯🇵 日本','🇸🇬 新加坡','🇭🇰香港精品4']];out=[]
for name in names:
 ctl.put(base+'/proxies/'+quote('Ghelper'),json={'name':name},timeout=5).raise_for_status()
 s=requests.Session();s.trust_env=False;s.proxies={'https':'http://127.0.0.1:17894'};t=time.time();n=0;status=None;err=None
 try:
  with s.get('https://huggingface.co/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst',headers={'Authorization':'Bearer '+token,'Range':'bytes=1225785344-1234173951'},stream=True,timeout=(5,6)) as r:
   status=r.status_code
   if status==206:
    for b in r.iter_content(65536):
     n+=len(b)
     if n>=8388608 or time.time()-t>15:break
 except Exception as e:err=type(e).__name__
 record={'epoch':time.time(),'node':name,'bytes':n,'seconds':time.time()-t,'HTTP':status,'error_type':err,'mac_used':False};out.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
 if n==8388608:break
best=max(out,key=lambda x:x['bytes']/max(x['seconds'],.001));chosen=best['node'] if best['bytes']>=1048576 else previous;ctl.put(base+'/proxies/'+quote('Ghelper'),json={'name':chosen},timeout=5).raise_for_status();result={'epoch':time.time(),'samples':out,'selected':chosen,'previous':previous,'mac_used':False};(root/'node-probe-latest.json').write_text(json.dumps(result,ensure_ascii=False));print(json.dumps({'selected':chosen,'valid_sample':best['bytes']>=1048576},ensure_ascii=False))
'''
p=subprocess.Popen(['ssh','hkust2','python3 -c '+shlex.quote(code)],stdin=subprocess.PIPE);p.stdin.write((secret+chr(10)).encode());p.stdin.close();raise SystemExit(p.wait(timeout=90))
