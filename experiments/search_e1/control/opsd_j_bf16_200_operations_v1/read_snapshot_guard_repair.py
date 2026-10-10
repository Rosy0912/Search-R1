"""Small read-only CPU snapshot of the new BF16 five-outer run; no raw outputs leave node."""
import json,time,subprocess,hashlib
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010')
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def proc(pid):
 try:
  p=Path('/proc')/str(pid);raw=(p/'stat').read_text();s=raw[raw.rfind(')')+2:].split();uid=int((p/'status').read_text().split('Uid:')[1].split()[0]);return dict(pid=pid,tick=int(s[19]),uid=uid,state=s[0])
 except (FileNotFoundError,ProcessLookupError):return None
cfg=read(B/'receipts/training-config.json');status=read(B/'runs/status.json');out={'remote_unix':time.time(),'config_sha256':hashlib.sha256((B/'receipts/training-config.json').read_bytes()).hexdigest(),'run_id':cfg['run_id'],'precision':cfg['precision'],'sampling_dtype':cfg['sampling']['dtype']}
if status:
 out['status']={k:status.get(k) for k in ['phase','outer_index','actual_optimizer_updates','optimizer_updates_before','error','observed_returncodes','owner']};out['actual_controller']=proc(status['owner']['pid'])
closed=[];live=[];sampling=[]
for outer in sorted((B/'runs/outers').glob('outer-*')):
 v=read(outer/'training/outer-complete.json')
 if v:closed.append({k:v.get(k) for k in ['accepted','outer_index','optimizer_updates_delta','optimizer_updates','P','K','cumulative_pair_exposure','global_valid_tokens','checkpoint']})
 v=read(outer/'training/live-update.json')
 if v:live.append(v)
 for attempt in sorted(outer.glob('sampling-attempt-*')):
  shards=[]
  for p in sorted(attempt.glob('rollouts-*.jsonl')):
   size=p.stat().st_size
   with p.open('rb') as f:lines=sum(block.count(b'\n') for block in iter(lambda:f.read(1<<20),b''))
   shards.append({'file':p.name,'bytes':size,'complete_lines':lines})
  sampling.append({'outer':outer.name,'attempt':attempt.name,'shards':shards,'complete_lines':sum(p['complete_lines'] for p in shards)})
out.update(completed_outers=len(closed),closed=closed,live_updates=live,sampling=sampling)
m=read(B/'receipts/memory-live.json')
if m:out['memory']={'age_s':time.time()-m['current']['sample_unix'],'samples_below_limit':m['samples_below_limit'],'minimum_actual_free_mib':m['minimum_actual_free_mib'],'current':m['current']}
r=subprocess.run(['tmux','list-panes','-t','e1-j-bf16-guard200-recovery2-20261010','-F','#{session_id} #{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],text=True,capture_output=True);out['tmux']=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
log=B/'logs/controller-v2.log'
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));out['controller_log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
