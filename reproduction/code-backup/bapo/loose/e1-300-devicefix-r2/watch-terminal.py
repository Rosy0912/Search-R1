"""Head-node read-only Slurm polling; preserve actual terminal bytes, never submit."""
import pathlib,re,subprocess,sys,time
D=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-devicefix-r2-20261004/scheduler-head')
phase,job=sys.argv[1:]
assert phase in ('hardware','formal300') and job.isascii() and job.isdigit()
assert (D/(phase+'-job-id.txt')).read_text().strip()==job
terminal=D/(phase+'-'+job+'-terminal.txt')
last=None
while not terminal.exists():
 p=subprocess.run(['scontrol','show','job','-dd',job],text=True,capture_output=True)
 if p.returncode:
  print('OBSERVATION_FAILED_NOT_JOB_TERMINAL',time.time(),p.returncode,p.stderr.strip(),flush=True)
 else:
  assert re.search(r'JobId='+job+r'\b',p.stdout)
  assert re.search(r'JobName=bapo-instruct300-'+phase+r'\s',p.stdout)
  state=re.search(r'JobState=(\S+)',p.stdout).group(1)
  if state!=last:
   print('ACTUAL_JOB_STATE',time.time(),job,state,flush=True);last=state
  if state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'):
   with terminal.open('x')as f:f.write(p.stdout)
   print(p.stdout,flush=True)
   break
 time.sleep(30)
assert terminal.is_file() and not terminal.is_symlink()
print('TERMINAL_BYTES_SAVED',str(terminal),flush=True)
