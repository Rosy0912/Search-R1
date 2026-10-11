"""Retain rolling2 plus requested protected snapshots, only within this run."""
import json,os,shutil,time
from pathlib import Path
PROTECTED={5,50,100,150,200}
KEYS=('job_id','code_commit','config_sha256','data_identity','model_identity','actual_update_limit')

def prune(base,completed):
 base=Path(base).resolve();state=base/'training_state'
 current=json.loads((state/f'global_step_{completed}'/'manifest.json').read_text())
 if current['status']!='COMPLETE_SAME_WORLD_STATE_SAVED' or current['identity']['completed_step']!=completed:
  raise RuntimeError('No complete current rolling boundary')
 receipt_dir=base/'rolling-pruned-receipts';receipt_dir.mkdir(exist_ok=True)
 pruned=[]
 for directory in sorted(state.glob('global_step_*')):
  suffix=directory.name.removeprefix('global_step_')
  if not suffix.isdecimal():continue
  step=int(suffix)
  if step in PROTECTED or step>=completed-1:continue
  manifest_path=directory/'manifest.json'
  if not manifest_path.exists():continue # Partial/unknown evidence is never cleaned up.
  old=json.loads(manifest_path.read_text())
  if old.get('status')!='COMPLETE_SAME_WORLD_STATE_SAVED' or old.get('identity',{}).get('completed_step')!=step:continue
  if any(old['identity'].get(k)!=current['identity'].get(k) for k in KEYS):continue
  targets=[base/name/directory.name for name in ('training_state','actor','algorithm')]
  # Validate every deletion before deleting any of the known same-run directories.
  for target in targets:
   if not target.exists():continue
   if target.is_symlink() or not target.resolve().is_relative_to(base) or target.stat().st_uid!=os.getuid():
    raise RuntimeError('Rolling candidate path/owner changed')
   if any(p.is_symlink() for p in target.rglob('*')):raise RuntimeError('Rolling candidate contains a symlink')
  receipt=receipt_dir/(directory.name+'.json')
  with receipt.open('x') as f:
   json.dump(dict(time=time.time(),status='PRUNE_AUTHORIZED_ROLLING_ONLY',
     replaced_by=completed,original_manifest=old,targets=list(map(str,targets))),f)
   f.flush();os.fsync(f.fileno())
  for target in targets:
   if target.exists():shutil.rmtree(target)
  pruned.append(step)
 return pruned
