"""Remove teacher29 head-of-line blocking while preserving all owned partials."""
import base64,json,pathlib,shlex,subprocess,sys
source=base64.b64encode(pathlib.Path('work/topd_hf_download_hkust3.py').read_bytes()).decode()
code=r'''
import base64,fcntl,json,pathlib,os,signal,subprocess,sys,time
token=sys.stdin.readline().strip();assert token;r=pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009');sub=r/'index-download'
assert not (r/'split-download-controller.json').exists(),'Do not repeat split handoff'
plan=json.loads((r/'plan.json').read_text());records=plan['records'];idx=next(x for x in records if x['group']=='wiki_index');teacher=next(x for x in records if x['group']=='teacher_model-00029-of-00029')
c=json.loads((r/'controller.json').read_text());pid=c['pid'];cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes();assert b'topd_hf_download_hkust3.py' in cmd and str(r).encode() in cmd
os.kill(pid,signal.SIGTERM);lock=(r/'downloader.lock').open('a')
for _ in range(100):
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(.1)
else:raise RuntimeError('Prior writer lock retained')
(r/'plan-before-split.json').write_text(json.dumps(plan));sub.mkdir(exist_ok=False)
part='wiki_index.'+idx['archive_sha256']+'.part';assert (r/part).exists();os.replace(r/part,sub/part)
(sub/'plan.json').write_text(json.dumps(dict(plan,records=[idx])))
(sub/'hf-source-sha-inventory.json').write_bytes((r/'hf-source-sha-inventory.json').read_bytes())
(sub/'topd_hf_download_hkust3.py').write_bytes(base64.b64decode(SOURCE))
(r/'plan.json').write_text(json.dumps(dict(plan,records=[teacher])))
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')};env.update(HTTP_PROXY='http://127.0.0.1:17894',HTTPS_PROXY='http://127.0.0.1:17894',TOPD_REUSE_VERIFIED_INVENTORY='1')
def start(root):
 with (root/'download.log').open('ab') as log:
  p=subprocess.Popen(['python3',str(root/'topd_hf_download_hkust3.py')],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.DEVNULL,start_new_session=True,env=env);p.stdin.write((token+'\n').encode());p.stdin.close();return p.pid
tp=start(r);ip=start(sub)
# Publish index into the existing forwarder's directory only after full SHA receipt.
publish=r/'index_publish.py';publish.write_text("import pathlib,json,time,os,hashlib\nr=pathlib.Path("+repr(str(r))+");s=r/'index-download';f=s/'wiki_index.tar.zst';v=s/'wiki_index-download-verified.json'\nwhile not (f.exists() and v.exists()):time.sleep(5)\na=json.loads(v.read_text());assert a['sha256']=="+repr(idx['archive_sha256'])+" and a['bytes']=="+str(idx['archive_bytes'])+"\nh=hashlib.sha256()\nwith f.open('rb') as x:\n for b in iter(lambda:x.read(4194304),b''):h.update(b)\nassert h.hexdigest()==a['sha256'];assert not (r/f.name).exists();os.link(f,r/f.name);t=r/'wiki_index-download-verified.tmp';t.write_text(json.dumps(a));os.replace(t,r/v.name);(r/'index-publish-complete.json').write_text(json.dumps({'epoch':time.time(),'status':'INDEX_SHA_VERIFIED_PUBLISHED','mac_used':False}))\n")
pp=subprocess.Popen(['python3',str(publish)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
out={'epoch':time.time(),'retired_pid':pid,'teacher29_pid':tp,'index_pid':ip,'index_publish_pid':pp.pid,'index_root':str(sub),'index_partial_retained_bytes':(sub/part).stat().st_size,'forwarder_untouched':True,'mac_used':False,'reason':'Independent missing groups avoid head-of-line blocking'}
(r/'split-download-controller.json').write_text(json.dumps(out));print(json.dumps(out))
'''.replace('SOURCE',repr(source))
token=sys.stdin.readline().strip();p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=token+'\n',capture_output=True,text=True,timeout=25)
out=json.loads(p.stdout) if not p.returncode else {'error':'split_failed','exit':p.returncode};pathlib.Path('outputs/hpc3-migration/split-download-controller.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
