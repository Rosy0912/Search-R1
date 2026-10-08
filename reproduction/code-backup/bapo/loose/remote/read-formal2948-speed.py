import json,subprocess,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo/runs/bapo-formal500-2948')
rows=[]
for line in (r/'train.log').read_text(errors='replace').splitlines():
 if 'BAPO_TRAIN_METRICS_JSON=' in line:
  x=json.loads(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]);m=x['metrics']
  rows.append({'step':x['step'],'metrics':{k:v for k,v in m.items() if k.startswith(('timing_s/','bapo/stage','bapo/rollout','rollout/','actor/optimizer','actor/lr','actor/grad','response_length/','prompt_length/','env/','bapo/early','bapo/stalled','bapo/resampl','bapo/loss_tokens','state_tokens/'))}})
watch=[json.loads(l) for l in (r/'resource-watch.jsonl').read_text().splitlines()]
query='index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate'
samples=[]
for i in range(3):
 samples.append({'time':time.time(),'csv':subprocess.check_output(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader'],text=True)})
 if i<2:time.sleep(2)
tail=(r/'retriever.log').read_text(errors='replace').splitlines()[-20:]
print(json.dumps({'time':time.time(),'run':str(r),'completed_outers':len(rows),'train_metrics':rows,'watch_events':len(watch),'watch_last':watch[-1],'live_gpu_query':query,'live_gpu_samples':samples,'retriever_tail':tail},ensure_ascii=False))
