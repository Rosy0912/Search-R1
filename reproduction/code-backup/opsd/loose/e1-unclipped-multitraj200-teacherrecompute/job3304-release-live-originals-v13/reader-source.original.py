PID=2548319
RUN='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
REPO='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/repo-fcecbea5d939'
WORKERS=[2548974, 2549227, 2549228, 2549229, 2549231, 2549232, 2549230, 2549233]
import pathlib,os,json,hashlib,base64,subprocess,datetime,stat
p=pathlib.Path('/proc')/str(PID);status=p.joinpath('status').read_text();cgroup=p.joinpath('cgroup').read_text();ticks=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19];assert ticks=='1834784786' and '/job_3304/' in cgroup
fd=p/'fd/1';target=pathlib.Path(os.readlink(fd));assert target.is_absolute() and target.resolve(strict=True)==target and str(target).startswith('/tmp/o4e-3304/ray/') and target.name.endswith('-2548319.out');before=target.stat();assert stat.S_ISREG(before.st_mode) and before.st_uid==0 and before.st_nlink==1 and before.st_size<16*1024*1024
files={}
for name,f in [('driver-current-prefix.out',target),('train.log',pathlib.Path(RUN)/'train.log'),('acceptance-5-receipt.json',pathlib.Path(RUN)/'acceptance-5-receipt.json')]:
 st=f.stat();n=st.st_size;assert n<16*1024*1024 and f.resolve(strict=True)==f and st.st_uid==0
 with f.open('rb') as x:b=x.read(n);opened=os.fstat(x.fileno());x.seek(0);assert x.read(n)==b
 after=f.stat();assert (st.st_dev,st.st_ino,st.st_uid)==(opened.st_dev,opened.st_ino,opened.st_uid)==(after.st_dev,after.st_ino,after.st_uid) and len(b)==n and after.st_size>=n
 files[name]={'physical':str(f),'bytes':n,'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode(),'scope':'Observed original full initial prefix; not terminal writer-closure proof'}
def command(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=25);return {'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
workers={}
for q in WORKERS:
 x=pathlib.Path('/proc')/str(q);workers[str(q)]={'status':x.joinpath('status').read_text(),'cgroup':x.joinpath('cgroup').read_text(),'stat':x.joinpath('stat').read_text()}
base=pathlib.Path(RUN)/'trajectories/outer_000006';arc={f.name:f.stat().st_size for f in base.iterdir()} if base.exists() else None
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_pid':PID,'driver_start':ticks,'driver_status':status,'driver_cgroup':cgroup,'driver_stat':p.joinpath('stat').read_text(),'driver_wchan':p.joinpath('wchan').read_text(),'workers':workers,'files':files,'outer6_members_observed':arc,'source_head':command(['git','--no-optional-locks','-C',REPO,'rev-parse','HEAD']),'source_status':command(['git','--no-optional-locks','-C',REPO,'status','--porcelain']),'slurm':command(['scontrol','show','job','-o','3304']),'gpu':command(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']),'gpu_apps':command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])},sort_keys=True))
