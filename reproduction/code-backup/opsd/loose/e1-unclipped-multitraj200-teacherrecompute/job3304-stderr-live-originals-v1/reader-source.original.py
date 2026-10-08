import pathlib,os,stat,datetime,json,hashlib,base64
starts={2548319:'1834784786',2548974:'1834785562',2549227:'1834786108',2549228:'1834786110',2549229:'1834786111',2549231:'1834786113',2549232:'1834786114',2549230:'1834786112',2549233:'1834786115'}
files={}
for pid,start in starts.items():
 p=pathlib.Path('/proc')/str(pid);beforeproc=p.joinpath('stat').read_text();assert beforeproc.rsplit(')',1)[1].split()[19]==start;cg=p.joinpath('cgroup').read_text();assert '/job_3304/step_0\n' in cg
 f=pathlib.Path(os.readlink(p/'fd/2'));assert f.is_absolute() and f.resolve(strict=True)==f and str(f).startswith('/tmp/o4e-3304/ray/session_') and f.name.endswith('-'+str(pid)+'.err')
 s=f.stat();assert stat.S_ISREG(s.st_mode) and s.st_uid==0 and s.st_nlink==1 and s.st_size<16*1024**2
 with f.open('rb') as z:
  raw=z.read(s.st_size);opened=os.fstat(z.fileno());z.seek(0);assert z.read(s.st_size)==raw
 a=f.stat();assert (s.st_dev,s.st_ino,s.st_uid)==(opened.st_dev,opened.st_ino,opened.st_uid)==(a.st_dev,a.st_ino,a.st_uid) and a.st_size>=s.st_size and len(raw)==s.st_size
 afterproc=p.joinpath('stat').read_text();assert afterproc.rsplit(')',1)[1].split()[19]==start
 files[str(pid)]={'physical':str(f),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode(),'pid_start_ticks':start,'proc_stat_before':beforeproc,'proc_stat_after':afterproc,'cgroup':cg,'scope':'In-flight original initial prefix, not writer-closed terminal stderr'}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'3304','files':files},sort_keys=True))
