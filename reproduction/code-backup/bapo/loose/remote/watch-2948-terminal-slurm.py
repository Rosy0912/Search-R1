import subprocess,time
for i in range(60):
 p=subprocess.run(['scontrol','show','job','2948'],capture_output=True,text=True)
 if p.returncode or 'JobState=RUNNING' not in p.stdout:
  print('OBSERVED_EPOCH='+str(time.time()),flush=True);print(p.stdout,flush=True);print(p.stderr,flush=True);break
 time.sleep(10)
else:print('TERMINAL_NOT_OBSERVED_WITHIN_600_SECONDS',flush=True)
