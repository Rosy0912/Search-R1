"""Read-only acceptance of the uninterrupted new experiment's real updates."""
import argparse,json,os,re,time,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from verify_checkpoint import verify

p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
b=Path(a.base);rdir=b/'receipts';stage=b/'runs/r1/round-1-grpo'
assert b.name=='search-e1-config-study-r1-aligned-8gpu-20261005'
budget=json.loads((rdir/'aligned-config-approved.json').read_text())['grpo_updates_per_stage']
assert budget in [200,300]
expected_run=None;checks={}
def save(name,record):
 target=rdir/name;temp=target.with_suffix('.tmp')
 temp.write_text(json.dumps(record,indent=2));temp.replace(target)
while True:
 f=rdir/'r1-round-1-grpo.json'
 if not f.exists():time.sleep(10);continue
 try:r=json.loads(f.read_text())
 except (OSError,json.JSONDecodeError):time.sleep(2);continue
 if expected_run is None:expected_run=r['run_id']
 assert r['run_id']==expected_run,'Observer must not silently follow a different run'
 live=False
 try:
  proc=Path('/proc')/str(r['pid'])
  live=proc.stat().st_uid==os.getuid() and ('E1_RUN_ID='+expected_run).encode() in proc.joinpath('environ').read_bytes().split(b'\0')
 except FileNotFoundError:pass
 text=(stage/'train.log').read_text(errors='replace') if (stage/'train.log').exists() else ''
 text=re.sub(r'\x1b\[[0-9;]*m','',text)
 rows={}
 for line in text.splitlines():
  match=re.search(r'\bstep:(\d+) -',line)
  if match and 'actor/grad_norm:' in line:rows[int(match[1])]=line
 for step in [2,5]:
  checkpoint=stage/'actor'/('global_step_'+str(step))
  if step in rows and step not in checks and (checkpoint/'config.json').exists():
   checks[step]=verify(b/'assets/model',checkpoint)
   save('aligned-step'+str(step)+'-weights.json',checks[step])
 totals={x[0]:int(x[2]) for x in r['admission']['devices']}
 reserves={idx:totals[idx]-peak for idx,peak in r['peaks_mib'].items()}
 invalid=any(not math.isfinite(float(value)) for line in rows.values() for value in re.findall(r'actor/(?:pg_loss|grad_norm|kl_loss):(-?(?:\d+(?:\.\d+)?|nan|inf))',line,re.I))
 accepted=all(step in rows for step in range(1,6)) and all(checks.get(s,{}).get('accepted') for s in [2,5]) and not invalid and len(reserves)==8 and min(reserves.values())>=6144
 report=dict(run_id=expected_run,node=os.uname().nodename,base=str(b),driver_group_pid=r['pid'],live_owned_group=live,status=r['status'],completed_iteration=max(rows,default=0),grpo_updates_per_stage=budget,minimum_reserve_mib=min(reserves.values(),default=0),reserves_mib=reserves,nonfinite_actor_metric=invalid,checkpoint_checks=checks,checkpoints=[str(x) for x in (stage/'actor').glob('global_step_*') if x.is_dir()],latest_metric=rows[max(rows)] if rows else None,five_real_updates_accepted=accepted,updated=time.time())
 save('aligned-live-acceptance.json',report)
 if accepted and not (rdir/'aligned-five-iterations.json').exists():save('aligned-five-iterations.json',report)
 if r['status'] not in ['RUNNING','STARTING'] and not live:break
 time.sleep(30)
