"""Hand off only the owned stalled downloader; keep forwarding and partials."""
import base64,json,pathlib,shlex,subprocess,sys
source=pathlib.Path('work/topd_hf_download_hkust3.py').read_bytes()
remote=r'''
import base64,fcntl,json,pathlib,os,signal,sys,subprocess,time,hashlib
token=sys.stdin.readline().strip();assert token
root=pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009')
c=json.loads((root/'controller.json').read_text());pid=c['pid'];proc=pathlib.Path('/proc/'+str(pid));cmd=(proc/'cmdline').read_bytes()
assert b'topd_hf_download_hkust3.py' in cmd and str(root).encode() in cmd,'Owned downloader identity required'
before={p.name:p.stat().st_size for p in root.glob('*.part')}
os.kill(pid,signal.SIGTERM)
lock=(root/'downloader.lock').open('a')
for i in range(100):
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(.1)
else:raise RuntimeError('Old writer still holds lock')
script=root/'topd_hf_download_hkust3.py';old=script.read_bytes();(root/'download-before-adaptive.py').write_bytes(old)
data=base64.b64decode(SOURCE);script.write_bytes(data)
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
env.update(HTTP_PROXY='http://127.0.0.1:17894',HTTPS_PROXY='http://127.0.0.1:17894',TOPD_REUSE_VERIFIED_INVENTORY='1')
with (root/'download.log').open('ab') as log:
 p=subprocess.Popen(['python3',str(script)],env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.DEVNULL,start_new_session=True)
 p.stdin.write((token+'\n').encode());p.stdin.close()
out={'epoch':time.time(),'old_pid':pid,'new_pid':p.pid,'partials_before':before,'source_sha256':hashlib.sha256(data).hexdigest(),'read_timeout_seconds':20,'max_retry_sleep_seconds':20,'mac_used':False,'forwarder_untouched':True,'hf_inventory':'Reuses prior official SHA inventory; not a fresh HF query'}
(root/'adaptive-handoff.json').write_text(json.dumps(out));print(json.dumps(out))
'''.replace('SOURCE',repr(base64.b64encode(source).decode()))
token=sys.stdin.readline().strip()
p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(remote)],input=token+'\n',capture_output=True,text=True,timeout=25)
out=json.loads(p.stdout) if p.returncode==0 else {'error':'handoff_failed','exit':p.returncode}
pathlib.Path('outputs/hpc3-migration/adaptive-handoff.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
