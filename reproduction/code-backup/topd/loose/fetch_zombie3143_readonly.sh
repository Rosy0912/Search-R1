python3 - <<'PY'
import pathlib,subprocess,json,hashlib,base64,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs');job=3143
ctl=subprocess.run(['scontrol','show','job','-o',str(job)],capture_output=True,text=True)
files=list((r/f'g1-stage-zombie-native-stress-{job}').glob('*'))+[r/f'g1-stage-zombie-native-stress-{job}.log',r/'g1-stage-zombie-native-stress-v3-submission.json']
rows=[]
for p in files:
 if p.is_file():
  b=p.read_bytes();rows.append({'path':str(p),'name':p.name,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
print(json.dumps({'job_id':job,'captured_epoch':time.time(),'scontrol':{'returncode':ctl.returncode,'stdout':ctl.stdout,'stderr':ctl.stderr},'files':rows}))
PY
