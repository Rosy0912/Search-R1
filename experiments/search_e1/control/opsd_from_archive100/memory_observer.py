from pathlib import Path
import subprocess,json,time
b=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007');peaks={};samples=0
while True:
 try:
  rows=[]
  for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
   i,used,total,util=[int(x.strip()) for x in line.split(',')];peaks[str(i)]=max(peaks.get(str(i),0),used);rows.append(dict(gpu=i,used_mib=used,total_mib=total,utilization=util,reserve_mib=total-used))
  samples+=1
  stage=json.loads((b/'receipts/pipeline-stage.json').read_text())
  report=dict(time=time.time(),samples=samples,phase=stage,peaks_mib=peaks,current=rows,minimum_reserve_mib=min(r['total_mib']-peaks[str(r['gpu'])] for r in rows),scope='HKUST1 allocated8 cards; total sampled usage, keep stage identity; old-GRPO pre-release samples excluded')
  if stage['phase'] not in ['WAITING_COMPLETE_OLD_STATE_AND_ARCHIVE100_TRANSFER','STOPPING_ONLY_PRESERVED_OLD_ALIGNED8','WAITING_OLD_OWNED_GPU_RELEASE']:
   (b/'receipts/memory-continuous.json').write_text(json.dumps(report,indent=2))
  else:peaks={};samples=0
 except Exception as e:print(repr(e),flush=True)
 time.sleep(2)
