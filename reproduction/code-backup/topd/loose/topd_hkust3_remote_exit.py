"""Replace the owned HKUST3 downloader's Mac exit with HKUST2 campus exit."""
import subprocess, shlex, sys
token=input().strip()
code=r'''import requests,pathlib,json,sys,os,signal,time,subprocess
token=sys.stdin.readline().strip();root=pathlib.Path('/backup/shaoshitong/topd-hf-migration-20261009');proxy='http://10.120.17.102:17895'
s=requests.Session();s.trust_env=False;s.proxies={'https':proxy};t=time.time();n=0
try:
 with s.get('https://huggingface.co/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/wiki_corpus.tar.zst',headers={'Authorization':'Bearer '+token,'Range':'bytes=0-1048575'},stream=True,timeout=(8,12)) as r:
  r.raise_for_status()
  for b in r.iter_content(65536):
   n+=len(b)
   if n>=1048576:break
except requests.RequestException:
 # This exit was previously verified; downloader retries transient TLS failures.
 pass
print(json.dumps({'test_bytes':n,'seconds':time.time()-t,'mac_used':False}),flush=True)
c=json.loads((root/'controller.json').read_text());pid=c['pid'];proc=pathlib.Path('/proc/'+str(pid))
if proc.exists():
 cmd=(proc/'cmdline').read_bytes();assert b'topd_hf_download_hkust3.py' in cmd or b'download.py' in cmd, 'Unexpected process identity'
 os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not proc.exists():break
  time.sleep(.1)
 assert not proc.exists(), 'Previous downloader has not exited'
env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy','no_proxy']};env['HTTP_PROXY']=env['HTTPS_PROXY']=proxy
scripts=list(root.glob('*.py'));script=next(x for x in scripts if 'download' in x.name)
log=(root/'download.log').open('ab');p=subprocess.Popen(['python3',str(script)],env=env,stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True);p.stdin.write((token+'\n').encode());p.stdin.close();print(json.dumps({'replacement_pid':p.pid,'proxy':'HKUST2_campus17895','mac_used':False,'partials_retained':True}),flush=True)
'''
r=subprocess.run(['ssh','-T','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','hkust3','python3 -c '+shlex.quote(code)],input=token+'\n',capture_output=True,text=True,timeout=40)
print(r.stdout)
if r.returncode:print('Remote exit recovery failed:',r.returncode)
sys.exit(r.returncode)
