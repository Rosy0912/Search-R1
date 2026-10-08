python3 - <<'PY'
import pathlib,json,datetime,subprocess,hashlib,os,fcntl,errno
R=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004');out=R/'logs/ACTION-PREFIX-NATIVE-FAILURE3297-RELEASE-v1.json'
def cmd(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return {'args':a,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def path_state(p):
 try:s=os.stat(p);return {'path':str(p),'stat':'PRESENT','dev':s.st_dev,'ino':s.st_ino}
 except OSError as e:return {'path':str(p),'stat':'ERROR','errno':e.errno,'error':str(e)}
gpu=cmd(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']);assert gpu['returncode']==0;uuids=gpu['stdout'].splitlines();assert len(uuids)==len(set(uuids))==8
rows=[]
for uid in uuids:
 p=pathlib.Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(uid+'.lock');d={'uuid':uid,'path':str(p),'scope':'One temporary nonblocking open-file-description flock probe; released immediately; no file created or modified'};fd=None
 try:
  fd=os.open(p,os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);d['probe']='ACQUIRED_AND_RELEASED';fcntl.flock(fd,fcntl.LOCK_UN)
 except OSError as e:d.update(probe='FAILED',errno=e.errno,error=str(e))
 finally:
  if fd is not None:os.close(fd)
 rows.append(d)
cg=pathlib.Path('/sys/fs/cgroup/freezer/slurm/uid_0/job_3297');v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3297','node':cmd(['hostname']),'gpu_uuids':gpu,'all_uuid_existing_lease_probes':rows,'all_probes_free':all(d['probe']=='ACQUIRED_AND_RELEASED' for d in rows),'gpu_compute':cmd(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader']),'queue':cmd(['squeue','-w','h100-8','-h','-o','%i %T %j %N']),'standard_slurm_v1_job_cgroup_stat':path_state(cg),'standard_slurm_v1_step_cgroup_stat':path_state(cg/'step_0'),'management_actual_cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'scope':'Startup failed before own native cgroup header/result creation. Standard v1 job/step path real errno is recorded; not an invented helper cleanup {} or header-bound path.'}
b=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'payload':v},ensure_ascii=False))
PY
