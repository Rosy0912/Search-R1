import time,json,pynvml as n
n.nvmlInit()
print(json.dumps({'driver':n.nvmlSystemGetDriverVersion(),'time_us':int(time.time()*1e6)}))
for i in range(n.nvmlDeviceGetCount()):
 h=n.nvmlDeviceGetHandleByIndex(i);r={'index':i,'uuid':n.nvmlDeviceGetUUID(h),'whole_gpu_util':n.nvmlDeviceGetUtilizationRates(h).gpu}
 try:
  samples=n.nvmlDeviceGetProcessUtilization(h,0)
  r['samples']=[{k:getattr(s,k) for k in ['pid','timeStamp','smUtil','memUtil']} for s in samples]
 except n.NVMLError as e:r['error']={'code':e.value,'text':str(e)}
 print(json.dumps(r),flush=True)
n.nvmlShutdown()
