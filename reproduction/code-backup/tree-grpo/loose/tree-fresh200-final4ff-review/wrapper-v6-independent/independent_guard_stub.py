import hashlib,json,os,pathlib,subprocess,sys,time,types
source_path=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);raw=source_path.read_bytes()
if hashlib.sha256(raw).hexdigest()!='284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88':raise RuntimeError('fixed guard bytes changed')
read=pathlib.Path.read_text;uuids=['GPU-INDEPENDENT-CPU-'+str(i) for i in range(8)]
def fake_read(p,*a,**kw):
 text=str(p)
 if text.endswith('/tools/expected-gpu-uuids.json'):return json.dumps(uuids)
 if text=='/proc/self/cgroup':return '0::/slurm/job_9001/step_batch\n'
 if text=='/proc/999999/cgroup':return '0::/foreign/job_9002\n'
 if text=='/proc/999999/stat':return '999999 (foreign stub) '+' '.join(['S']+['0']*18+['123'])
 if text.startswith('/proc/'):raise RuntimeError('Real proc read prohibited')
 return read(p,*a,**kw)
pathlib.Path.read_text=fake_read
nv=types.ModuleType('pynvml');nv.NVMLError_NotFound=type('NoSamples',(Exception,),{})
nv.nvmlInit=lambda:None;nv.nvmlDeviceGetCount=lambda:8;nv.nvmlDeviceGetHandleByIndex=lambda i:i;nv.nvmlDeviceGetUUID=lambda i:uuids[i]
nv.nvmlDeviceGetMemoryInfo=lambda h:types.SimpleNamespace(total=100,used=80);nv.nvmlDeviceGetUtilizationRates=lambda h:types.SimpleNamespace(gpu=90)
nv.nvmlDeviceGetComputeRunningProcesses=lambda h:[types.SimpleNamespace(pid=999999,usedGpuMemory=50)];nv.nvmlDeviceGetGraphicsRunningProcesses=lambda h:[]
nv.nvmlDeviceGetProcessUtilization=lambda h,t:[types.SimpleNamespace(pid=999999,smUtil=90,timeStamp=t+1)]
sys.modules['pynvml']=nv;cancels=[]
def fake_cancel(c,**kw):cancels.append(c);return types.SimpleNamespace(returncode=0)
subprocess.run=fake_cancel
os.pidfd_open=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('Real pidfd prohibited'))
def stop(seconds):raise RuntimeError('CPU_SENTINEL_AFTER_COMPLETED_SAMPLE')
time.sleep=stop;sys.argv=[str(source_path),'--output',str(out)]
try:exec(compile(raw,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path)})
except BaseException as e:terminal=repr(e)
rows=[json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
print(json.dumps(dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE=os.environ.get('PYTHONOPTIMIZE'),terminal=terminal,healthy_samples=sum('gpus' in r and not r.get('event') for r in rows),records=rows,scancel_calls_stubbed=cancels,actual_scancel_calls=0,actual_nvml_calls=0,actual_proc_reads=0)))
