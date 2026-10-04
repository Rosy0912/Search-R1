import json,time,subprocess
from pathlib import Path
b=Path('/weka_3/shaoshitong/search-e1-retriever-stream-batched-20261004'); peak=0;svcpeak=0;n=0
while True:
 try:
  rows=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
  used,total=map(int,rows);peak=max(peak,used)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
  service=[int(x.split(',')[1]) for x in apps.splitlines() if x.split(',')[0].strip()=='1749337'];svcpeak=max(svcpeak,max(service,default=0));n+=1
  if n%5==0:(b/'receipts/shared-memory-monitor.json').write_text(json.dumps(dict(peak_mib=peak,service_peak_mib=svcpeak,current_mib=used,total_mib=total,reserve_mib=total-peak,samples=n,time=time.time(),service_pid=1749337),indent=2))
 except Exception as e:print(repr(e),flush=True)
 time.sleep(2)

