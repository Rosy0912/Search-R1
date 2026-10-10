"""Read-only follow-up: do not return success until eval10, five restored outers and outer16 activity."""
import subprocess,json,time,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
remote="""import json,time
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010');s=json.load(open(B/'runs/status.json'));D=B/'periodic512/eval-010';summary=json.load(open(D/'summary.json')) if (D/'summary.json').exists() else None;closed=[]
for i in range(10,15):
 p=B/'runs/outers'/f'outer-{i:05d}'/'training/outer-complete.json'
 if p.exists():
  v=json.load(open(p));closed.append({k:v.get(k) for k in ['accepted','outer_index','optimizer_updates','optimizer_updates_delta','P','K','checkpoint']})
counts=[len(p.read_bytes().splitlines()) for p in sorted(D.glob('results-*.jsonl'))]
print(json.dumps(dict(time=time.time(),phase=s['phase'],outer_index=s.get('outer_index'),error=s.get('error'),owner=s['owner'],summary=summary,counts=counts,closed=closed)))
"""
while True:
 r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=10','hkust1','python3 -B -'],input=remote,text=True,capture_output=True)
 if r.returncode:
  print(json.dumps(dict(ssh_rc=r.returncode,error=r.stderr[-500:])),flush=True);time.sleep(30);continue
 d=json.loads(r.stdout);(output/'latest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 summary=d['summary'];compact=dict(time=d['time'],phase=d['phase'],outer=d['outer_index'],eval_count=sum(d['counts']),eval_hits=summary.get('hits') if summary else None,restored_complete=len(d['closed']),error=d['error']);print(json.dumps(compact),flush=True)
 if d['phase'].startswith('FAILED'):raise SystemExit('Actual failure; inspect and repair before proceeding')
 if summary and summary['count']==512 and len(d['closed'])==5 and all(v['accepted'] for v in d['closed']) and d['outer_index'] is not None and d['outer_index']>=15 and d['phase'] in ('SAMPLING','TRAINING','OUTER_COMPLETE'):
  (output/'five-restored-outers-and-next-running.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print('ACCEPTED eval10 + restored11to15 + outer16 continuing',flush=True);break
 time.sleep(30)
