import sys,subprocess,time,json
from pathlib import Path
sys.path.insert(0,'work');from remote import route
out=Path('work/tree-instruct300-3154-review');code=(out/'read_five_ready.py').read_text();start=time.monotonic();i=0
while time.monotonic()-start<1800:
 i+=1;p=subprocess.run(route('python3 -'),input=code,text=True,capture_output=True,timeout=180)
 (out/f'FIVE_READY_WATCH_{i:02d}.json').write_text(p.stdout);(out/f'FIVE_READY_WATCH_{i:02d}.stderr').write_text(p.stderr)
 if p.returncode==0:
  x=json.loads(p.stdout);names=[Path(k).name for k in x['files']]
  if all(k in names for k in ['ready.json','hf5-full-reload.json','state5-files-cpu.json']) or x['result']:
   print(json.dumps({'status':'FIVE_CPU_REPORTS_OR_TERMINAL_READY','snapshot':i,'files':names,'result':x['result']}),flush=True);break
 time.sleep(90)
else:print(json.dumps({'status':'BOUNDED_FIVE_WAIT_EXPIRED'}),flush=True)
