import json,time,subprocess,hashlib,re
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');output=root/'output';files={}
for name in ['released.json','approved.json','ready.json']:
 p=output/'acceptance-step105'/name
 if p.exists():
  raw=p.read_bytes();files[str(p)]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest())
log=(root/'stage.log').read_text();lines=[s for s in log.splitlines() if 'step:' in s and 'training/actual_outer_updates:' in s]
finished=[int(re.search(r'training/actual_outer_updates:([0-9]+)',s)[1]) for s in lines]
rounds={}
for step in range(106,111):
 ranks=list((output/'tree-rank-audit').glob(f'step-{step:06d}-rank-*.json'));profiles=list((output/'tree-rank-audit/memory-profile').glob(f'step-{step:06d}-rank-*.json'));ratios=list((output/'tree-rank-audit/ratio-tail').glob(f'step-{step:06d}-rank-*-optimizer-*.json'));tree=list((output/'tree-audit').glob(f'step_{step}_rollout_*.json'));trajectory=(output/f'trajectories/outer-{step:06d}/manifest.json').exists()
 rounds[str(step)]=dict(ranks=len(ranks),profiles=len(profiles),ratios=len(ratios),tree=len(tree),trajectory_manifest=trajectory,console_completed=step in finished)
print(json.dumps(dict(epoch=time.time(),slurm=subprocess.check_output(['scontrol','show','job','3234'],text=True),gate105=files,rounds=rounds,completed_lines=lines,validation_reports=[str(p) for p in (output/'validation-cost').glob('*.json')],checkpoint110_manifest=(output/'checkpoints/tree-recovery300/training_state/global_step_110/manifest.json').exists())),flush=True)
