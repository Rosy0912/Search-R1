from pathlib import Path
import base64
b=base64.b64encode(Path('work/evidence/numericfix-accept5-3101-terminal-watch.log').read_bytes()).decode()
s='''import pathlib,base64,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric');run=root/'runs/bapo-accept5-3101';terminal=root/'logs/accept5-3101-terminal-slurm.log'
b=base64.b64decode(TERMINAL)
if terminal.exists():assert terminal.read_bytes()==b
else:terminal.write_bytes(b)
assert (run/'TRAIN_RETURNED_SUCCESSFULLY').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
args=[str(root/'env/bin/python'),str(root/'tools/rewardfix-evidence/bapo-accept5-evidence-gpu64-tied.py'),'--run-root',str(run),'--log',str(run/'train.log'),'--base-model',str(root/'assets-root/assets/qwen2.5-3b'),'--compare-weights','--hf-load','--hash-checkpoint','--job-id','3101','--slurm-record',str(terminal),'--output',str(root/'logs/accept5-3101-gpu64-tied-extraction.json')]
with (root/'logs/accept5-3101-gpu64-tied-extractor.log').open('w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('ACTUAL_ACCEPT5_3101_CPU_HF_EXTRACTION_RETURNED')
'''.replace('TERMINAL',repr(b))
Path('work/numericfix/extract-accept5-3101.py').write_text(s)
