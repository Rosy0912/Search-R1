from pathlib import Path
import base64
b=base64.b64encode(Path('work/evidence/rewardfix-smoke3069-terminal-watch.log').read_bytes()).decode()
s='''import pathlib,base64,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097');run=root/'runs/bapo-smoke2-3069';terminal=root/'logs/smoke2-3069-terminal-slurm.log'
b=base64.b64decode(TERMINAL)
if terminal.exists():assert terminal.read_bytes()==b
else:terminal.write_bytes(b)
assert (run/'TRAIN_RETURNED_SUCCESSFULLY').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
args=[str(root/'env/bin/python'),str(root/'tools/rewardfix-evidence/bapo-smoke2-evidence-gpu64-tied.py'),'--run-root',str(run),'--log',str(run/'train.log'),'--base-model',str(root/'assets-root/assets/qwen2.5-3b'),'--compare-weights','--hf-load','--hash-checkpoint','--job-id','3069','--slurm-record',str(terminal),'--output',str(root/'logs/smoke2-3069-gpu64-tied-extraction.json')]
with (root/'logs/smoke2-3069-gpu64-tied-extractor.log').open('w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('ACTUAL_SMOKE3069_CPU_HF_EXTRACTION_RETURNED')
'''.replace('TERMINAL',repr(b))
Path('work/rewardfix/extract-smoke3069.py').write_text(s)
