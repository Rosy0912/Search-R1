import subprocess,time,sys
job=sys.argv[1]
assert job.isascii() and job.isdigit()
for i in range(10000):
 p=subprocess.run(['scontrol','show','job',job],text=True,capture_output=True)
 print('CHECK',time.strftime('%Y-%m-%dT%H:%M:%S%z'),flush=True)
 print(p.stdout+p.stderr,flush=True)
 if any('JobState='+s in p.stdout for s in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY']):break
 if p.returncode:break
 time.sleep(30)
