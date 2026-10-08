import json,subprocess,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154')
gate=r/'output/acceptance-step5';cp=r/'output/checkpoints/tree-instruct300/training_state/global_step_5'
result=dict(epoch=time.time(),slurm=subprocess.run(['scontrol','show','job','3154'],capture_output=True,text=True).stdout,gate_files=[],state_files=[],stage_tail=(r/'stage.log').read_bytes()[-2500:].decode(errors='replace'))
for p in sorted(gate.glob('*')):result['gate_files'].append(dict(name=p.name,bytes=p.stat().st_size))
for p in sorted(cp.glob('*')):result['state_files'].append(dict(name=p.name,bytes=p.stat().st_size))
print(json.dumps(result))
