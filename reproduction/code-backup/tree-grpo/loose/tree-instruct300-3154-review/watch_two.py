import sys,subprocess,time,json
from pathlib import Path
sys.path.insert(0,'work');from remote import route
out=Path('work/tree-instruct300-3154-review');start=time.monotonic();i=0
code='''from pathlib import Path\nimport json,subprocess,time\nr=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154');reports=[]\nfor p in (r/'output/validation-cost').glob('*.json'):\n try: reports.append(dict(path=str(p),value=json.loads(p.read_text())))\n except json.JSONDecodeError:pass\ns=subprocess.run(['scontrol','show','job','3154'],capture_output=True,text=True)\nprint(json.dumps(dict(epoch=time.time(),slurm=s.stdout,error=s.stderr,reports=reports,rank2_count=len(list((r/'output/tree-rank-audit').glob('step-000002-rank-*.json'))),result=json.loads((r/'run-result.json').read_text()) if (r/'run-result.json').exists() else None)))\n'''
while time.monotonic()-start<1500:
 i+=1;p=subprocess.run(route('python3 -'),input=code,text=True,capture_output=True,timeout=180)
 (out/f'TWO_WATCH_{i:02d}.json').write_text(p.stdout);(out/f'TWO_WATCH_{i:02d}.stderr').write_text(p.stderr)
 if p.returncode==0:
  x=json.loads(p.stdout)
  if x['rank2_count']==8 or x['result']:
   print(json.dumps({'status':'REPORT_OR_TERMINAL_READY','snapshot':i,'rank2_count':x['rank2_count'],'result':x['result']}),flush=True);break
 time.sleep(90)
else:print(json.dumps({'status':'BOUNDED_TWO_WATCH_EXPIRED'}),flush=True)
