import pathlib,json,subprocess,datetime,hashlib,os
P=pathlib.Path;base=P('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v1');source=P('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')
assert (base/'preparation/stop3381-intent.json').exists() and not (base/'preparation/stop3381-receipt.json').exists()
job=subprocess.check_output(['scontrol','show','job','-o','3381'],text=True);assert 'JobState=FAILED' in job and 'ExitCode=1:0' in job
assert not subprocess.check_output(['squeue','-h','-j','3381','-o','%T'],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not apps.strip()
pids=[207996,220732,235993,236904,237335,237337,237338,237339,237340,237341,237342];assert all(not P(f'/proc/{p}').exists() for p in pids)
cgroup=P('/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3381');tasks=[]
if cgroup.exists():
 for p in cgroup.rglob('tasks'):tasks.extend(p.read_text().split())
assert not tasks
logs=[]
for p in [source/'logs/native-3381.log',source/'logs/search-g1-common8-training.log']:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 s=p.stat();logs.append({'path':str(p),'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'full_sha256':h.hexdigest()})
v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job3381_stopped':True,'manual_stop_scope':'final evaluation only; explicit user request; not spontaneous training failure','job_terminal':job,'legacy_cleanup_receipt_present':(source/'logs/cleanup-status.json').exists(),'legacy_cleanup_status':json.loads((source/'logs/pipeline-status.json').read_text()),'independent_resource_release_proved':True,'original_pids_gone':pids,'cgroup_tasks':tasks,'gpu_compute_apps':apps,'terminal_original_logs':logs,'completed200_preserved':True,'new_submission_performed':False}
with (base/'preparation/stop3381-receipt.json').open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps({'old_job':3381,'independent_resources_released':True,'legacy_cleanup_failed_preserved':True,'terminal_full_sha_log_count':len(logs)}))
