import pathlib,json,time,subprocess,fcntl,os,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent));import upload
ROOT=pathlib.Path(__file__).parent;lock=(ROOT/'cleanup-controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)
def remote(script):
 p=subprocess.run(upload.source_args('env -u LD_LIBRARY_PATH python3 '+script),capture_output=True,text=True,timeout=1900)
 if p.returncode:raise RuntimeError('Remote controller step exited '+str(p.returncode))
 return json.loads(p.stdout)
write('cleanup-controller.json',{'pid':os.getpid(),'epoch':time.time(),'status':'WAITING_BEST_INFERENCE_HF_SHA'})
while not (ROOT/'best-inference-complete.json').exists():time.sleep(20)
gate=json.loads((ROOT/'best-inference-complete.json').read_text());assert gate['status']=='BEST_INFERENCE_ALL_COMPONENTS_HF_SHA_VERIFIED';assert gate['files']>=13
pre=remote('/mnt/data/shaoshitong/cleanup-backups-20261012-gigpo/progress-check.py');write('cleanup-train-before.json',pre)
write('cleanup-controller.json',{'pid':os.getpid(),'epoch':time.time(),'status':'BEST_VERIFIED_FRESH_SOURCE_CHECK_AND_SAFE_DELETE'})
result=remote('/mnt/data/shaoshitong/cleanup-backups-20261012-gigpo/verified-best-then-clean.py');write('cleanup-remaining-result.json',result)
first=remote('/mnt/data/shaoshitong/cleanup-backups-20261012-gigpo/progress-check.py');write('cleanup-train-after1.json',first)
start=time.time()
while True:
 time.sleep(30);second=remote('/mnt/data/shaoshitong/cleanup-backups-20261012-gigpo/progress-check.py');write('cleanup-train-after2.json',second)
 if second.get('step',0)>pre.get('step',0) and len(second.get('ranks',[]))==8 and all(x['finite'] for x in second['ranks']):break
 if second.get('failure_files'):raise RuntimeError('Training failure after cleanup; source preserved and no train operation performed')
 if time.time()-start>3600:raise RuntimeError('Training completed-step increase not verified within one hour')
write('cleanup-complete.json',{'status':'BEST_HF_INFERENCE_SHA_VERIFIED_SOURCE_FRESH_CHECKED_OLD_OUTPUTS_CLEANED_TRAINING_ADVANCED','epoch':time.time(),'deleted_bytes':result['deleted_allocated_bytes'],'deleted_paths':len(result['deleted']),'blocked':result['blocked'],'after_df':result['after'],'before_step':pre['step'],'after_step':second['step'],'extra_state_upload_status':'separate uploader continues until upload-complete.json'})
