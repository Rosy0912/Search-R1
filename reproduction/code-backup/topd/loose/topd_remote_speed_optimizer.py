"""Keep testing isolated remote exits; hand off only after two fast real samples."""
import concurrent.futures,fcntl,json,os,pathlib,signal,subprocess,sys,time,urllib.request
ROOT=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009')
MIG=pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009')
lock=(ROOT/'speed-optimizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
TOKEN=sys.stdin.readline().strip();assert TOKEN
def save(name,data):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False));os.replace(t,p)
def api(path,method='GET',body=None):
 req=urllib.request.Request('http://127.0.0.1:19897'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'},method=method)
 with urllib.request.urlopen(req,timeout=4) as r:return json.loads(r.read() or '{}')
owned=json.loads((ROOT/'fresh-exit-controller.json').read_text());cmd=pathlib.Path('/proc/'+str(owned['pid'])+'/cmdline').read_bytes();assert b'/dev/shm/topd-hkust2-fresh-speed.yaml' in cmd
save('speed-optimizer-controller.json',{'pid':os.getpid(),'epoch':time.time(),'mac_used':False,'threshold_MB_s':5,'phase':'TESTING_ISOLATED_REMOTE_EXIT'})
api('/configs','PATCH',{'mode':'global'})
nodes=[n for n in api('/proxies')['proxies']['GLOBAL']['all'] if n not in ('DIRECT','REJECT','GLOBAL') and '智能' not in n and n!='Ghelper']
nodes=sorted(nodes,key=lambda n:0 if '硅谷' in n else 1)
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
def sample():
 start=time.time()
 def piece(i):
  cfg='url = "https://huggingface.co/Sutoonq/TMP_MODEL/resolve/main/topd-migration-20261009/archives/teacher_model-00028-of-00029.tar.zst"\nheader = "Authorization: Bearer '+TOKEN+'"\nrange = "'+str(i*4194304)+'-'+str((i+1)*4194304-1)+'"\n'
  p=subprocess.run(['curl','-4','--proxy','http://127.0.0.1:17897','-L','-s','-o','/dev/null','--connect-timeout','8','--max-time','16','-w','%{http_code} %{size_download}','--config','-'],input=cfg,text=True,capture_output=True,timeout=20,env=env)
  fields=p.stdout.split();return p.returncode==0 and fields==['206','4194304']
 with concurrent.futures.ThreadPoolExecutor(8) as pool:valid=all(list(pool.map(piece,range(8))))
 dt=time.time()-start;return {'epoch':time.time(),'valid_all_8_ranges':valid,'bytes':33554432 if valid else None,'seconds':dt,'MB_s':33.554432/dt if valid else None}
for cycle in range(100):
 for node in nodes:
  if (MIG/'teacher_model-00029-of-00029-download-verified.json').exists() and (MIG/'wiki_index-download-verified.json').exists():
   save('speed-optimizer-result.json',{'epoch':time.time(),'phase':'DOWNLOAD_FINISHED_NO_HANDOFF','mac_used':False});sys.exit(0)
  try:
   api('/proxies/GLOBAL','PUT',{'name':node});a=sample();a.update(node=node,cycle=cycle,mac_used=False);save('speed-optimizer-latest.json',a)
   with (ROOT/'speed-optimizer-history.jsonl').open('a') as f:f.write(json.dumps(a,ensure_ascii=False)+'\n')
   if not a['valid_all_8_ranges'] or a['MB_s']<5:continue
   b=sample();save('speed-optimizer-confirm.json',b)
   if not b['valid_all_8_ranges'] or b['MB_s']<5:continue
   # Do not interrupt an existing writer that is already faster.
   def total():return sum(p.stat().st_size for p in MIG.rglob('*.part'))+sum(p.stat().st_size for p in MIG.rglob('*.tar.zst'))
   before=total();t=time.time();time.sleep(20);baseline=(total()-before)/(time.time()-t)/1e6
   if baseline>=min(a['MB_s'],b['MB_s']):
    save('speed-optimizer-result.json',{'epoch':time.time(),'phase':'EXISTING_WRITER_FASTER','baseline_MB_s':baseline,'candidate_MB_s':b['MB_s'],'mac_used':False});sys.exit(0)
   active=MIG if not (MIG/'download-complete.json').exists() else MIG/'index-download'
   c=json.loads((active/'controller.json').read_text());pid=c['pid'];cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes();assert str(active/'topd_hf_download_hkust3.py').encode() in cmd
   os.kill(pid,signal.SIGTERM);writer_lock=(active/'downloader.lock').open('a')
   for i in range(100):
    try:fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:time.sleep(.1)
   else:raise RuntimeError('Prior writer lock retained')
   fcntl.flock(writer_lock,fcntl.LOCK_UN);writer_lock.close()
   newenv=dict(env,HTTP_PROXY='http://127.0.0.1:17897',HTTPS_PROXY='http://127.0.0.1:17897',TOPD_REUSE_VERIFIED_INVENTORY='1')
   with (active/'download.log').open('ab') as log:
    p=subprocess.Popen(['python3',str(active/'topd_hf_download_hkust3.py')],env=newenv,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.DEVNULL,start_new_session=True);p.stdin.write((TOKEN+'\n').encode());p.stdin.close()
   index_new_pid=None;idx=MIG/'index-download'
   if active!=idx and idx.exists() and not (idx/'download-complete.json').exists():
    ic=json.loads((idx/'controller.json').read_text());old_index=ic['pid'];icmd=pathlib.Path('/proc/'+str(old_index)+'/cmdline').read_bytes();assert str(idx/'topd_hf_download_hkust3.py').encode() in icmd
    os.kill(old_index,signal.SIGTERM);il=(idx/'downloader.lock').open('a')
    for _ in range(100):
     try:fcntl.flock(il,fcntl.LOCK_EX|fcntl.LOCK_NB);break
     except BlockingIOError:time.sleep(.1)
    else:raise RuntimeError('Index writer lock retained')
    fcntl.flock(il,fcntl.LOCK_UN);il.close()
    with (idx/'download.log').open('ab') as log:
     ip=subprocess.Popen(['python3',str(idx/'topd_hf_download_hkust3.py')],env=newenv,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.DEVNULL,start_new_session=True);ip.stdin.write((TOKEN+'\n').encode());ip.stdin.close();index_new_pid=ip.pid
   save('speed-optimizer-result.json',{'epoch':time.time(),'phase':'FAST_EXIT_HANDOFF','old_pid':pid,'new_pid':p.pid,'index_new_pid':index_new_pid,'node':node,'samples':[a,b],'baseline_MB_s':baseline,'partials_retained':True,'mac_used':False});sys.exit(0)
  except Exception as e:save('speed-optimizer-failure.json',{'epoch':time.time(),'node':node,'cycle':cycle,'error_type':type(e).__name__})
 time.sleep(30)
save('speed-optimizer-result.json',{'epoch':time.time(),'phase':'SEARCH_LIMIT_REACHED_NO_FAST_EXIT','mac_used':False})
