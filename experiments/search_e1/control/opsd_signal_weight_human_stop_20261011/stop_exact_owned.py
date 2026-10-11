import pathlib,json,os,time,signal,ctypes,platform
T=pathlib.Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
roots={3550871:643848202,176419:649251801}
assert platform.system()=='Linux' and platform.machine()=='x86_64'
lib=ctypes.CDLL(None,use_errno=True);lib.syscall.restype=ctypes.c_long

def call(n,*a):
 v=lib.syscall(ctypes.c_long(n),*a)
 if v<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
 return int(v)
def opened(pid):return call(434,ctypes.c_int(pid),ctypes.c_uint(0))
def send(fd,sig):return call(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(None),ctypes.c_uint(0))
def proc(pid):
 p=pathlib.Path('/proc',str(pid));t=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,ppid=int(t[1]),tick=int(t[19]),state=t[0],uid=p.stat().st_uid,argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0')[:-1]])
def tree():
 d={}
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:d[int(p.name)]=proc(int(p.name))
   except (OSError,ValueError):pass
 own=set(roots)
 while True:
  new={i for i,a in d.items() if a['ppid'] in own}-own
  if not new:break
  own|=new
 return {i:d[i] for i in own if i in d}
def write(p,v):
 tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
started=time.time();receipt=T/'receipts'/('human-request-stop-'+str(int(started))+'.json')
v=dict(started_unix=started,human_stop_text='停掉现在的实验吧，没意义了',human_cancel_automation_text='取消现在的自动发送任务',human_original_time='UNKNOWN; latest live human turn after 20:24 continuation',scope='Exact current driver/controller descendants only; preserve all artifacts; do not resume',roots=roots,signals=[],retained_artifacts=True)
fds={};initial={}
for pid,tick in roots.items():
 p=proc(pid);assert p['uid']==1001 and p['tick']==tick
 assert any('hkust5-device-isolation-v2' in x for x in p['argv'])
 if pid==176419:assert 'ecf9c8afeabf4d90994726c7b73219bb' in p['argv']
 fd=opened(pid);assert proc(pid)['tick']==tick;fds[pid]=fd;initial[pid]=p
v['initial_roots']=initial;write(receipt,v)
for pid in roots:
 send(fds[pid],signal.SIGSTOP);v['signals'].append(dict(pid=pid,tick=roots[pid],signal='SIGSTOP'))
time.sleep(.2);owned=tree();v['owned_snapshot']=owned
for pid,p in owned.items():
 assert p['uid']==1001
 if pid not in fds and p['state']!='Z':
  try:
   fd=opened(pid);a=proc(pid);assert a['tick']==p['tick'] and a['uid']==1001;fds[pid]=fd
  except ProcessLookupError:continue
write(receipt,v)
# Children first; driver/controller stay frozen until their pending termination.
for pid in [i for i in fds if i not in roots]+[176419,3550871]:
 try:
  send(fds[pid],signal.SIGTERM);v['signals'].append(dict(pid=pid,tick=owned.get(pid,initial.get(pid))['tick'],signal='SIGTERM'))
  if pid in roots:send(fds[pid],signal.SIGCONT);v['signals'].append(dict(pid=pid,tick=roots[pid],signal='SIGCONT after pending SIGTERM'))
 except ProcessLookupError:v['signals'].append(dict(pid=pid,result='already exited'))
for fd in fds.values():os.close(fd)
v['signals_complete_unix']=time.time();v['current']={}
for pid,p in owned.items():
 try:a=proc(pid);v['current'][pid]=a if a['tick']==p['tick'] else dict(identity_absent=True,reused_pid=True)
 except OSError:v['current'][pid]=dict(absent=True)
write(receipt,v);print(json.dumps(dict(receipt_path=str(receipt),signals=v['signals'],current=v['current'],retained_artifacts=True)))
