"""Sanitized remote redirect and Range diagnostics; no secret URLs are logged."""
import concurrent.futures,json,pathlib,shlex,subprocess,sys
code=r'''
import requests,sys,time,json,urllib.parse
token=sys.stdin.readline().strip();s=requests.Session();s.trust_env=False
url='https://hf-mirror.com/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst'+SUFFIX
out=[]
for hop in range(5):
 t=time.time();r=None
 try:
  r=s.get(url,headers={'Authorization':'Bearer '+token,'Range':'bytes=0-1048575'},stream=True,allow_redirects=False,timeout=(4,8))
  row={'host':urllib.parse.urlsplit(url).hostname,'status':r.status_code,'header_seconds':time.time()-t,'content_length':r.headers.get('Content-Length'),'content_range':r.headers.get('Content-Range')}
  if r.is_redirect:
   url=urllib.parse.urljoin(url,r.headers['Location']);row['next_host']=urllib.parse.urlsplit(url).hostname;out.append(row);r.close();continue
  n=0
  if r.status_code in (200,206):
   for b in r.iter_content(65536):
    n+=len(b)
    if n>=1048576 or time.time()-t>15:break
  row.update(bytes=n,seconds=time.time()-t);out.append(row);r.close();break
 except Exception as e:
  out.append({'host':urllib.parse.urlsplit(url).hostname,'error':type(e).__name__,'seconds':time.time()-t});break
print(json.dumps(out))
'''
token=sys.stdin.readline().strip()
def run(case):
 node,suffix=case
 p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes',node,'python3 -c '+shlex.quote(code.replace('SUFFIX',repr(suffix)))],input=token+'\n',capture_output=True,text=True,timeout=50)
 return {'node':node,'variant':suffix,'result':json.loads(p.stdout) if p.returncode==0 else {'error':'remote_failed','exit':p.returncode}}
with concurrent.futures.ThreadPoolExecutor(4) as pool:out=list(pool.map(run,[('hkust1',''),('hkust1','?download=true'),('hkust2',''),('hkust2','?download=true')]))
pathlib.Path('outputs/hpc3-migration/redirect-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
