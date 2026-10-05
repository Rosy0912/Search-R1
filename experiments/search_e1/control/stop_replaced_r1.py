"""Stop one freshly verified private old R1 process group, preserving evidence."""
import os,json,re,signal,time
from pathlib import Path
old=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004')
new=Path('/data/shaoshitong/search-e1-config-study-r1-aligned-8gpu-20261005')
run='5dfc903d248842b4bfc4e1ebcf5d94e8';group=3546009
r=json.loads((old/'receipts/r1-round-1-grpo.json').read_text())
assert r['status']=='RUNNING' and r['run_id']==run and r['pid']==group
full=json.loads((old/'preserved'/('replacement-state-'+run)/'full-complete.json').read_text())
assert full['run_id']==run and full['world_size']==8
assert sorted(x['rank'] for x in full['ranks'])==list(range(8))
assert all(Path(x['path']).stat().st_size==x['bytes'] for x in full['ranks'])
owned=[];marker=('E1_RUN_ID='+run).encode()
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  pid=int(p.name)
  if os.getpgid(pid)!=group:continue
  assert p.stat().st_uid==os.getuid()
  if marker not in p.joinpath('environ').read_bytes().split(b'\0'):
   cmd=p.joinpath('cmdline').read_bytes()
   assert b'ray-dashboard-' in cmd and b'--multiprocessing-fork' in cmd and str(old/'env/bin/python').encode() in cmd
   parent=int(re.search(r'^PPid:\s+(\d+)',p.joinpath('status').read_text(),re.M).group(1));pp=Path('/proc')/str(parent)
   assert marker in pp.joinpath('environ').read_bytes().split(b'\0') and b'/ray/dashboard/dashboard.py' in pp.joinpath('cmdline').read_bytes()
   assert os.getsid(pid)==group and os.getpgid(parent)==group and pp.stat().st_uid==os.getuid()
  owned.append(pid)
 except (FileNotFoundError,ProcessLookupError):continue
assert group in owned and 3548832 in owned
t=(old/'runs/r1/round-1-grpo/train.log').read_text()
steps=[int(v) for v in re.findall(r'\bstep:(\d+)\s+-',t)]
proof=dict(reason='Human replacement goal; preserve old R1 and run aligned fresh8 experiment',run_id=run,private_group=group,owned_pids=owned,completed_metric_steps=steps,preserved_full_state_step=full['global_step'],old_stage_receipt=r,time=time.time())
for b in [old,new]:(b/'receipts/old-r1-user-replacement-stop.json').write_text(json.dumps(proof,indent=2))
os.killpg(group,signal.SIGTERM)
print(json.dumps(proof),flush=True)
