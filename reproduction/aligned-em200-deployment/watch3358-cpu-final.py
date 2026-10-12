import json,subprocess,time
from pathlib import Path
b=Path(__file__).resolve().parent;cwd=b.parents[3]
for n in range(30):
 out=cwd/'work/evidence'/f'node11-3358-cpu-final-watch-{n:02}.actual.json'
 r=subprocess.run(['python3',str(b/'inventory_transport.py'),'--target','h100-11','--python-file',str(b/'requests/read3358-cpu-final.py'),'--output',str(out),'--timeout','80'],cwd=cwd,capture_output=True,text=True,timeout=100)
 if r.returncode==0:
  d=json.loads(out.read_text())
  if not d.get('pending',False):print(json.dumps({'evidence':str(out),'status':d.get('status'),'issues':d.get('issues'),'cost':d.get('cost')}),flush=True);break
 else:print(json.dumps({'observation_exit':r.returncode,'not_training_exit':True}),flush=True)
 time.sleep(45)
