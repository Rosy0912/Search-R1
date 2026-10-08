import subprocess,json,time
job='3109'
r=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True,check=True);print(r.stdout,flush=True)
assert 'JobId=3109 JobName=bapo-rfix-formal500-numeric' in r.stdout
assert 'WorkDir=/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric' in r.stdout
assert 'Command=/dev/shm/bapo-rewardfix-numeric-scheduler-20261004/formal500.sbatch' in r.stdout
if 'JobState=RUNNING' in r.stdout or 'JobState=PENDING' in r.stdout:
 subprocess.run(['scancel',job],check=True);print('CANCEL_REQUESTED_ONLY_OWN_JOB_3109',flush=True)
for i in range(20):
 r=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True);print(r.stdout+r.stderr,flush=True)
 if any('JobState='+s in r.stdout for s in ['CANCELLED','COMPLETED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY']):break
 time.sleep(5)
