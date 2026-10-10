"""Local readonly first five + sixth real-progress follower; never signals."""
import subprocess,json,time,pathlib,sys
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
script='''import pathlib,json,time
b=pathlib.Path('/data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010');s=json.loads((b/'runs/status.json').read_text());closed=[]
for i in range(5):
 p=b/'runs/outers'/f'outer-{i:05d}'/'training/outer-complete.json'
 if p.exists():
  d=json.loads(p.read_text());closed.append({'outer':i,'accepted':d['accepted'],'updates':d['optimizer_updates'],'checkpoint':d['checkpoint'],'K':d['K']})
p=b/'runs/outers/outer-00005';rows=sum(sum(1 for _ in f.open()) for f in p.glob('sampling-attempt-*/rollouts-[0-7].jsonl'));m=json.loads((b/'receipts/memory-live.json').read_text())
print(json.dumps({'time':time.time(),'phase':s['phase'],'outer_index':s.get('outer_index'),'owner':s['owner'],'error':s.get('error'),'closed':closed,'sixth_rows':rows,'memory_age':time.time()-m['updated_unix'],'below':m['samples_below_limit']}))
'''
while True:
 r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=8','hkust1','python3','-'],input=script,text=True,capture_output=True)
 if r.returncode:
  print('READ_UNKNOWN',r.stderr[-200:],flush=True);time.sleep(30);continue
 d=json.loads(r.stdout);(out/'latest.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ['owner','closed']}|{'completed':len(d['closed']),'updates':[x['updates'] for x in d['closed']]}),flush=True)
 if 'FAILED' in d['phase']:
  (out/'failure.json').write_text(json.dumps(d,indent=2));sys.exit(2)
 if len(d['closed'])==5 and all(x['accepted'] for x in d['closed']) and d['sixth_rows']>0:
  (out/'five-and-sixth-accepted.json').write_text(json.dumps(d,indent=2));sys.exit(0)
 time.sleep(30)
