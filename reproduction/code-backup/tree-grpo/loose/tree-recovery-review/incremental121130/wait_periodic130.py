import subprocess,time,json
from pathlib import Path
p=Path(__file__).parent
for n in range(1,21):
 dst=p/f'PERIODIC130_DISCOVERY_{n:02d}.json'
 with dst.open('xb') as out, (p/'read_periodic130.py').open('rb') as source:
  r=subprocess.run(['python3','work/remote.py','--stdin',str(p/'read_periodic130.py'),"CUDA_VISIBLE_DEVICES='' /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python -"],stdout=out,stderr=subprocess.PIPE,timeout=60)
 dst.with_suffix('.stderr').write_bytes(r.stderr)
 if r.returncode:raise RuntimeError(r.stderr.decode())
 x=json.loads(dst.read_text());ready=len(x['step130_console_lines'])>0 and any('/training_state/global_step_130/manifest.json' in k for k in x['files']) and any('/validation-cost/' in k and k.endswith('.json') for k in x['files'])
 print(json.dumps(dict(iteration=n,complete=ready,files=len(x['files']),console=len(x['step130_console_lines']))),flush=True)
 if ready:
  with (p/'PERIODIC130_RAW.json').open('xb') as out:out.write(dst.read_bytes())
  break
 time.sleep(60)
