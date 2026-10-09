"""Recover this task's hfd through HKUST2's own network; token only stdin/RAM."""
import json, pathlib, shlex, subprocess, sys
token = input().strip()
script = pathlib.Path(__file__).with_name('topd_hfd_ram_token.sh').read_text()
code = r'''import pathlib,sys,json,requests,time,subprocess,os
token=sys.stdin.readline().strip();script=sys.stdin.read()
root=pathlib.Path('/data/shaoshitong/topd-hfd-20261009')
assert not pathlib.Path('/proc/3661373').exists(), 'Old hfd still alive'
s=requests.Session();s.trust_env=False;s.proxies={'https':'http://127.0.0.1:17894'}
t=time.time();n=0
with s.get('https://huggingface.co/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/wiki_corpus.tar.zst',headers={'Authorization':'Bearer '+token,'Range':'bytes=0-8388607'},stream=True,timeout=(8,15)) as r:
 r.raise_for_status()
 for b in r.iter_content(65536):
  n+=len(b)
  if n>=8388608:break
test={'epoch':time.time(),'bytes':n,'seconds':time.time()-t,'proxy':'HKUST2_local17894','mac_used':False}
(root/'remote-exit-success.json').write_text(json.dumps(test));print(json.dumps(test),flush=True)
(root/'hfd.sh').write_text(script)
env={k:v for k,v in os.environ.items() if k.lower() not in ['https_proxy','http_proxy','all_proxy','no_proxy']}
env.update(HF_TOKEN=token,HF_ENDPOINT='https://huggingface.co',HTTP_PROXY='http://127.0.0.1:17894',HTTPS_PROXY='http://127.0.0.1:17894',TOPD_HF_PROXY='http://127.0.0.1:17894')
log=(root/'hfd.log').open('ab');p=subprocess.Popen(['bash',str(root/'hfd.sh'),'Sutoonq/TMP_MODEL','--hf_username','Sutoonq','--include','topd-migration-20261009/archives/wiki_corpus.tar.zst','-x','8','-j','1','--local-dir',str(root/'official')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'pid':p.pid,'epoch':time.time(),'root':str(root),'group':'wiki_corpus','proxy':'HKUST2_local17894','mac_used':False,'resume':True,'token_persisted':False};(root/'controller.json').write_text(json.dumps(receipt));print(json.dumps(receipt),flush=True)
'''
r=subprocess.run(['ssh','-T','-B','en0','-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=token+'\n'+script,capture_output=True,text=True,timeout=45)
print(r.stdout)
if r.returncode: print('Recovery failed:',r.returncode)  # Avoid credential-bearing request traceback.
sys.exit(r.returncode)
