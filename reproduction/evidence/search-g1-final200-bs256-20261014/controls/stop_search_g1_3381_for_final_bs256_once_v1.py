import pathlib,json,os,datetime,subprocess,time,hashlib
P=pathlib.Path;base=P('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v1');source=P('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')
def once(p,v):
 with P(p).open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
assert (base/'preparation/deploy-receipt.json').is_file() and (base/'preparation/asset-receipt.json').is_file()
assert not (base/'preparation/stop3381-intent.json').exists(),'Inspect existing intent/receipt instead of replay'
assert not (source/'logs/final-identity.json').exists(),'Original final evaluation completed; no stop/replay'
detail=subprocess.check_output(['scontrol','show','job','-o','3381'],text=True);assert 'JobState=RUNNING' in detail and 'NodeList=h100-8' in detail and str(source/'preparation/launch-mixed-native.sh') in detail
updates=[json.loads(l) for l in (source/'logs/actual-updates.jsonl').read_text().splitlines()];assert updates[-1]['step']==200 and updates[-1]['optimizer_writes_after']==4000
ledger=json.loads((source/'logs/process-ledger.json').read_text())
known_pids=[235993,236904,237335,237337,237338,237339,237340,237341,237342]
births={pid:int(P(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]) for pid in known_pids if P(f'/proc/{pid}').exists()}
intent={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'能否停掉resume，通过增大bs去加速推理？？','scope':'Only current final evaluation job3381; preserve completed200, original inputs/checkpoints/logs and other tasks; restart final-only fromCP200','job_before':detail,'old_process_ledger':ledger,'source_completed_outer':200,'source_actual_adam_per_rank':4000,'new_base':str(base)}
once(base/'preparation/stop3381-intent.json',intent)
r=subprocess.run(['scancel','--signal=TERM','--batch','3381'],capture_output=True,text=True);once(base/'preparation/stop3381-signal.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0
end=time.monotonic()+90
while time.monotonic()<end:
 q=subprocess.run(['squeue','-h','-j','3381','-o','%T'],capture_output=True,text=True).stdout.strip()
 cleanup=source/'logs/cleanup-status.json'
 if not q and cleanup.exists() and json.loads(cleanup.read_text()).get('cleaned'):break
 time.sleep(2)
q=subprocess.run(['squeue','-h','-j','3381','-o','%T'],capture_output=True,text=True).stdout.strip();assert not q,'Old job still active; inspect, do not submit final'
assert (source/'logs/cleanup-status.json').exists() and json.loads((source/'logs/cleanup-status.json').read_text())['cleaned'],'Owned cleanup not verified'
for pid,ticks in births.items():
 if P(f'/proc/{pid}').exists():assert int(P(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])!=ticks,'Original owned process still live'
assert not P('/proc/235993').exists(),'Original driver still live; no replacement submit'
logs=[]
for p in [source/'logs/native-3381.log',source/'logs/search-g1-common8-training.log']:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 st=p.stat();logs.append({'path':str(p),'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'full_sha256':h.hexdigest()})
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job3381_stopped':True,'manual_stop_scope':'final evaluation only','cleanup':json.loads((source/'logs/cleanup-status.json').read_text()),'terminal_original_logs':logs,'completed200_preserved':True,'new_submission_performed':False}
once(base/'preparation/stop3381-receipt.json',receipt);print(json.dumps(receipt))
