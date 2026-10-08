import pathlib,sys,json,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
src=r'''
import pathlib,json,hashlib,base64,subprocess,time
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');p=R/'logs/r1-32b-grpo-3213.log';a=p.stat();b=p.read_bytes();z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
out={'scope':'OLD3213_FAILURE_ONLY_NEW3214_NOT_MODIFIED','epoch':time.time(),'file':{'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size':len(b),'base64':base64.b64encode(b).decode()},'old_job_cgroup_members':[],'applications':{},'gpu_identities':[],'old_cleanup_receipt_exists':(R/'logs/r1-standard32b-grpo100-direct-v2/evidence/owned-cleanup.json').exists()}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:cg=(p/'cgroup').read_text()
 except FileNotFoundError:continue
 if '/job_3213/' in cg or cg.rstrip().endswith('/job_3213'):out['old_job_cgroup_members'].append({'pid':int(p.name),'cgroup':cg})
a=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10);out['applications']={'returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr}
for row in a.stdout.splitlines():
 parts=[x.strip() for x in row.split(',')]
 if len(parts)!=3 or not parts[1].isdecimal():continue
 p=pathlib.Path('/proc')/parts[1]
 try:st=(p/'stat').read_text();tail=st.rsplit(')',1)[1].split();out['gpu_identities'].append({'pid':int(parts[1]),'starttime_ticks':int(tail[19]),'cgroup':(p/'cgroup').read_text(),'uid':p.stat().st_uid})
 except FileNotFoundError:out['gpu_identities'].append({'pid':int(parts[1]),'error':'PROC_ABSENT_NOT_ZERO'})
print(json.dumps(out))
'''
r=send(src,attempts=3,retry_safe=True,timeout=60);r.check_returncode();d=json.loads(r.stdout);f=d['file'];b=base64.b64decode(f.pop('base64'));assert hashlib.sha256(b).hexdigest()==f['sha256'];p=pathlib.Path('outputs/remote-evidence/direct3213');p.mkdir(exist_ok=True);(p/'r1-32b-grpo-3213.log').write_bytes(b);(p/'readonly-failure-physical.json').write_text(json.dumps(d,indent=2)+'\n');print(b.decode()[-3000:]);print('logSHA',f['sha256'],'oldcg',d['old_job_cgroup_members'],'cleanup_recorded',d['old_cleanup_receipt_exists'])
