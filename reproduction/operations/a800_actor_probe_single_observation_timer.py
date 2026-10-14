
"""Wait locally; make one observation of the exact existing actor-probe handle."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--handle',type=Path,required=True);p.add_argument('--not-before',type=float,required=True)
a=p.parse_args();root=a.root.resolve();handle=a.handle.resolve();handle.relative_to(root/'preparation')
intent=json.loads(handle.read_text());assert intent['reward_schedule']=='g1_through100_then_binary' and intent['source_commit']=='c4d156de124de7a94652d75858bcbabc8c56083c'
print(json.dumps(dict(role='single_local_actual_actor_probe_observer_waiter',pid=os.getpid(),created_unix=time.time(),not_before_unix=a.not_before,deployment_id=intent['deployment_id'],remote_query_performed=False)),flush=True)
while True:
 remaining=a.not_before-time.time()
 if remaining<=0:break
 time.sleep(min(30,remaining))
assert json.loads(handle.read_text())==intent,'Original deployment handle changed during wait'
try:
 result=subprocess.run([sys.executable,'-m','reproduction.a800_common.deploy_actor_probes','--handle',str(handle),'--observe'],cwd=root/'actor-probes-ten',capture_output=True,text=True,timeout=120)
 assert len(result.stdout.encode())<2*1024**2,'Unexpected non-metadata observation'
 row=json.loads(result.stdout)
 print(json.dumps(dict(role='single_existing_actor_probe_observation_completed',exit_code=result.returncode,result=row,no_mutation_or_auto_retry=True,checked_unix=time.time())),flush=True)
except (subprocess.TimeoutExpired,OSError,ValueError,AssertionError) as error:
 row=dict(role='single_actor_probe_observation_failed',failure_type=type(error).__name__,observed_remote_state='unknown',checked_unix=time.time(),no_mutation_or_auto_retry=True)
 target=root/'progress-checks'/('actor-probes-ten-observation-unknown-'+str(time.time_ns())+'.json')
 with target.open('x') as f:json.dump(row,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(**row,record=str(target),record_sha256=hashlib.sha256(target.read_bytes()).hexdigest())),flush=True)
