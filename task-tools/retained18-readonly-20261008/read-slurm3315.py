"""Read only the fixed BAPO job; unknown query state is never terminal."""
import subprocess,json,datetime
p=subprocess.run(['scontrol','show','job','3315','-o'],capture_output=True,text=True,timeout=15)
print(json.dumps({'job_id':3315,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}));raise SystemExit(p.returncode)
