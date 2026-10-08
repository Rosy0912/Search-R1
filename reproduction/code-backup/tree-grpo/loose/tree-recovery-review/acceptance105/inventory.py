from pathlib import Path
import json,time,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');g=r/'output/acceptance-step105';files={}
for p in [g/'ready.json',g/'hf105-full-reload.json',g/'state105-files-cpu.json']:
 if p.exists():files[str(p)]=p.read_text()
print(json.dumps(dict(epoch=time.time(),slurm=subprocess.check_output(['scontrol','show','job','3234'],text=True),files=files,rank105_count=len(list((r/'output/tree-rank-audit').glob('step-000105-rank-*.json'))),guardpath=[str(p) for p in r.glob('*guard*')],resourcepath=[str(p) for p in r.glob('*resource*')]),ensure_ascii=False))
