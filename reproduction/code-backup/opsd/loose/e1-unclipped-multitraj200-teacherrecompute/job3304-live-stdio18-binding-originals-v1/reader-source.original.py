import pathlib,os,stat,json,datetime,subprocess,re
P=pathlib.Path('/proc');expected={2548319:'1834784786',2548974:'1834785562',2549227:'1834786108',2549228:'1834786110',2549229:'1834786111',2549231:'1834786113',2549232:'1834786114',2549230:'1834786112',2549233:'1834786115'}
def proc(pid):
 p=P/str(pid);a=p.joinpath('stat').read_text();s=p.joinpath('status').read_text();c=p.joinpath('cgroup').read_text();assert a.rsplit(')',1)[1].split()[19]==expected[pid] and re.search(r'^Uid:\s+0\s+0\s+0\s+0$',s,re.M) and re.search(r'^State:\s+[RSDI]',s,re.M) and '/job_3304/step_0\n' in c;return {'stat':a,'status':s,'cgroup':c,'start_ticks':expected[pid]}
def st(s):return {'device':s.st_dev,'inode':s.st_ino,'uid':s.st_uid,'mode':s.st_mode,'link_count':s.st_nlink,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def cmd(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=25);return {'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
assert os.getuid()==0;slurm=cmd(['scontrol','show','job','-o','3304']);assert slurm['returncode']==0
for v in ['JobId=3304 ','JobState=RUNNING','Restarts=0','NodeList=h100-10','gres:gpu:h100:8']:assert v in slurm['stdout']
rows={};identities=set()
for pid in expected:
 a=proc(pid);f={}
 for n,suffix in [(1,'.out'),(2,'.err')]:
  q=P/str(pid)/'fd'/str(n);target=pathlib.Path(os.readlink(q));assert target.is_absolute() and target.resolve(strict=True)==target and str(target).startswith('/tmp/o4e-3304/ray/session_2026-10-06_19-44-37_985734_2537562/logs/worker-') and target.name.endswith('-'+str(pid)+suffix)
  before=target.stat();via=q.stat();assert stat.S_ISREG(before.st_mode) and before.st_uid==0 and before.st_nlink==1 and (before.st_dev,before.st_ino)==(via.st_dev,via.st_ino);info=(P/str(pid)/'fdinfo'/str(n)).read_text();flags=int(re.search(r'^flags:\s+(\d+)$',info,re.M)[1],8);assert (flags & os.O_ACCMODE) in (os.O_WRONLY,os.O_RDWR)
  assert (before.st_dev,before.st_ino) not in identities;identities.add((before.st_dev,before.st_ino));f[str(n)]={'path':str(target),'fdinfo':info,'before':st(before),'fd_stat':st(via)}
 b=proc(pid)
 for n,v in f.items():
  q=P/str(pid)/'fd'/n;assert os.readlink(q)==v['path'];after=pathlib.Path(v['path']).stat();via=q.stat();assert (after.st_dev,after.st_ino,after.st_uid,after.st_nlink)==(v['before']['device'],v['before']['inode'],0,1)==(via.st_dev,via.st_ino,via.st_uid,via.st_nlink) and after.st_size>=v['before']['bytes'];v['after']=st(after)
 rows[str(pid)]={'pid':pid,'before_proc':a,'after_proc':b,'stdio':f}
repo='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/repo-fcecbea5d939';head=cmd(['git','--no-optional-locks','-C',repo,'rev-parse','HEAD']);clean=cmd(['git','--no-optional-locks','-C',repo,'status','--porcelain']);assert head['returncode']==clean['returncode']==0 and head['stdout'].strip()=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6' and not clean['stdout'].strip()
print(json.dumps({'version':1,'kind':'ACTUAL_LIVE3304_STDIO18_FD_SOURCE_BINDING_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'3304','stdio_count':18,'rows':rows,'slurm_original':slurm,'source_head':head,'source_status':clean,'scope':'Actual live FD1/2 path+inode/PIDstart origin only; file bodies not read, no closed log/terminal/release/200 claim'},sort_keys=True))
