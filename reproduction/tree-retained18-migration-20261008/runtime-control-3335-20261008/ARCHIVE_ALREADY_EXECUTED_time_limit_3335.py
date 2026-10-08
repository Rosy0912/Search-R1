from pathlib import Path
import os,json,hashlib,re,subprocess,time,socket
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-current200-continue-3335'
expected_source='6efb60466a233f70bf8f50de26df32012f49b4bc';expected_config='5e530fc597e26e113c42418ca54faa9bdd0a42ad40bc1998e22c9599be9668fb';expected_manifest='1a2a7ea8dea6a599d7d170671fb0ff6f2a6076b2dc5c8b61653e0381642ebb6b'
package=root/'instruct300/candidates/current200-retained18-v11-6efb604';intent=root/'instruct300/SUBMISSION_ONCE-retained18-current200-v11-6efb604.json'
assert socket.gethostname().split('.')[0]=='h100-10' and os.getuid()==0
assert root.resolve()==Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
package_sha=hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest();assert package_sha==expected_manifest
journal=json.loads(intent.read_text());assert journal['status']=='SUBMITTED' and journal['job_id']==3335 and journal['manifest_sha256']==expected_manifest
latest=json.loads((run/'checkpoints/training_state/rolling-completed/latest.json').read_text());manifest=Path(latest['manifest_path']);assert manifest.resolve().is_relative_to((run/'checkpoints/training_state/rolling-completed').resolve())
raw=manifest.read_bytes();assert hashlib.sha256(raw).hexdigest()==latest['manifest_sha256'];identity=json.loads(raw)['identity']
assert identity['job_id']=='3335' and identity['code_commit']==expected_source and identity['config_sha256']==expected_config and identity['run_id']==run.name and 5<=identity['completed_step']<=200
before=subprocess.run(['scontrol','--local','show','job','3335'],text=True,capture_output=True,timeout=25);assert before.returncode==0 and before.stderr==''
fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_\-/]*)=([^\s]+)',before.stdout))
expected={'JobId':'3335','JobState':'RUNNING','NodeList':'h100-10','BatchHost':'h100-10','UserId':'root(0)','StartTime':'2026-10-08T12:31:23','TimeLimit':'7-00:00:00','Restarts':'0','Requeue':'0','StdOut':str(root/'instruct300/node-move-logs/3335.out'),'TresPerNode':'gres:gpu:8'}
assert all(fields.get(k)==value for k,value in expected.items()),'Exact current own job/time lineage differs; no mutation'
folder=run/'output/runtime-control';folder.mkdir(exist_ok=True);record_path=folder/'time-limit-168h-to-240h-20261008.json'
record={'status':'INTENT_SAVED','job_id':3335,'node':'h100-10','created_epoch':time.time(),'source_commit':expected_source,'config_sha256':expected_config,'original_package_manifest_sha256':package_sha,'original_startup_wall_hours':168,'requested_live_wall_hours':240,'reason':'Authorized fixed200/full51713 goal requires finite extra scheduler time at actually observed pace; no extra training iterations or change of algorithm/split/batch/length/optimizer/evaluation/source. Original startup manifest and identity preserved.','precheck':'RETAINED18_FORMAL3335_DEADLINE_READONLY_PRECHECK.json','before':{'returncode':before.returncode,'stdout':before.stdout,'stderr':before.stderr},'command':['scontrol','--local','update','JobId=3335','TimeLimit=10-00:00:00'],'model_data_pt_or_gpu_calls':False,'training_paused_restarted_or_requeued':False}
with record_path.open('x') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
updated=subprocess.run(record['command'],text=True,capture_output=True,timeout=25)
record['update_result']={'returncode':updated.returncode,'stdout':updated.stdout,'stderr':updated.stderr}
after=subprocess.run(['scontrol','--local','show','job','3335'],text=True,capture_output=True,timeout=25)
record['after']={'returncode':after.returncode,'stdout':after.stdout,'stderr':after.stderr};new=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_\-/]*)=([^\s]+)',after.stdout))
record['confirmed_live_wall_hours']=240 if updated.returncode==0 and after.returncode==0 and new.get('TimeLimit')=='10-00:00:00' else None
record['confirmed_end_time']=new.get('EndTime');record['after_state']=new.get('JobState')
record['same_job_and_start']=all(new.get(k)==expected[k] for k in ['JobId','NodeList','BatchHost','UserId','StartTime','Restarts','Requeue','StdOut','TresPerNode'])
record['package_unchanged']=hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest()==package_sha
record['status']='LIVE_TIME_LIMIT_240H_CONFIRMED' if record['confirmed_live_wall_hours']==240 and record['same_job_and_start'] and record['package_unchanged'] else 'UPDATE_OUTCOME_REQUIRES_INSPECTION'
record['finished_epoch']=time.time();pending=record_path.with_suffix('.pending')
with pending.open('x') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
os.replace(pending,record_path);print(json.dumps(record))
