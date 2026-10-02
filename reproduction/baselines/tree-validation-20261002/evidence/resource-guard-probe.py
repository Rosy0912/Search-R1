"""Run actual guard source against fake NVML/procfs; never GPU or scancel."""
import sys,os,json,tempfile,pathlib,types,time,subprocess,hashlib
from unittest.mock import patch
ROOT=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
script=(ROOT/'work/resource_guard.py').read_text()
class NoSample(Exception):pass
class Unsupported(Exception):pass
class EndPoll(Exception):pass
S=types.SimpleNamespace
uids=['GPU-toy-'+str(i) for i in range(8)]
results=[]
for case in ('baseline_notfound','foreign_memory','foreign_sm','unknown_memory','not_supported','same_pid_graphics_overwrite','foreign_cgroup_prefix','departed_known_own','departed_unknown_new_sample','compute_unknown_then_graphics_known'):
 with tempfile.TemporaryDirectory() as td:
  output=pathlib.Path(td)/'guard.jsonl';cancellations=[];phase=[0];clock=[99.9]
  def cg(pid):return '0::/slurm/job_1234/step_0\n' if pid in ('self','20') else '0::/slurm/job_999/step_0\n'
  original_read=pathlib.Path.read_text
  def read(p,*args,**kwargs):
   if str(p)=='/proc/self/cgroup':return '0::/slurm/job_12345/step_0\n' if case=='foreign_cgroup_prefix' else cg('self')
   if str(p).endswith('expected-gpu-uuids.json'):return json.dumps(uids)
   if str(p).startswith('/proc/') and str(p).endswith('/cgroup'):
    if p.parts[-2]=='20' and phase[0]>0 and case.startswith('departed_'):raise FileNotFoundError()
    return cg(p.parts[-2])
   return original_read(p,*args,**kwargs)
  def compute(h):
   if h!=0:return []
   if case.startswith('departed_'):return [S(pid=20,usedGpuMemory=1)] if phase[0]==0 else []
   amount=20 if case in ('foreign_memory','same_pid_graphics_overwrite') else (2**64-1 if case in ('unknown_memory','compute_unknown_then_graphics_known') else 1)
   return [S(pid=10,usedGpuMemory=amount)]
  def graphics(h):return [S(pid=10,usedGpuMemory=1)] if h==0 and case in ('same_pid_graphics_overwrite','compute_unknown_then_graphics_known') else []
  def samples(h,stamp):
   if case in ('not_supported','foreign_cgroup_prefix'):raise Unsupported('unsupported')
   if case.startswith('departed_') and phase[0]>0 and h==0:return [S(pid=20,timeStamp=100750000 if case=='departed_known_own' else 100850000,smUtil=80)]
   if case=='foreign_sm' and h==0:return [S(pid=10,timeStamp=100000001,smUtil=16)]
   raise NoSample()
  nv=S(nvmlInit=lambda:None,nvmlDeviceGetCount=lambda:8,nvmlDeviceGetHandleByIndex=lambda i:i,
   nvmlDeviceGetUUID=lambda h:uids[h],nvmlDeviceGetMemoryInfo=lambda h:S(total=100,used=1),
   nvmlDeviceGetUtilizationRates=lambda h:S(gpu=0),nvmlDeviceGetComputeRunningProcesses=compute,
   nvmlDeviceGetGraphicsRunningProcesses=graphics,nvmlDeviceGetProcessUtilization=samples,NVMLError_NotFound=NoSample)
  def cancel(cmd,**kw):cancellations.append(cmd);return S(returncode=0)
  error=None
  def now():
   clock[0]+=.1;return clock[0]
  def pause(seconds):
   if case.startswith('departed_') and phase[0]==0:phase[0]=1;return
   raise EndPoll('probe end')
  with patch.dict(os.environ,{'SLURM_JOB_ID':'1234','SLURM_JOB_NODELIST':'h100-9'}),patch.dict(sys.modules,{'pynvml':nv}),patch.object(sys,'argv',['guard','--output',str(output)]),patch.object(pathlib.Path,'read_text',read),patch.object(subprocess,'run',cancel),patch.object(time,'time',now),patch.object(time,'sleep',pause):
   try:exec(compile(script,'resource_guard.py','exec'),{})
   except BaseException as e:error=type(e).__name__+':'+str(e)
  rows=[json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
  results.append({'case':case,'completed_poll':any('gpus' in row for row in rows),'completed_poll_count':sum('gpus' in row for row in rows),'exception':error,'cancellations':cancellations})
print(json.dumps({'mock_only_no_real_gpu_or_scancel':True,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'sleep_ends_healthy_probe_by_injected_EndPoll':True,'cases':results},indent=2))
