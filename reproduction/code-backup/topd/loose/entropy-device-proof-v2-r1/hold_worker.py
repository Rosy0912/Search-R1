"""Keep actual own CUDA PID alive until parent finished original foreign gate."""
import os,sys,json,time,runpy,pathlib
folder=pathlib.Path(os.environ['R1_DEVICE_PROOF_FOLDER'])
def publish(name,value):
 p=folder/name;t=p.with_name(name+'.private')
 with t.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
 os.link(t,p);t.unlink()
import torch
x=torch.zeros((),device='cuda');torch.cuda.synchronize();del x;torch.cuda.empty_cache()
publish('child-context-ready.json',{'pid':os.getpid(),'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()})
while not (folder/'run-authorized').exists():time.sleep(.02)
runpy.run_path(str(pathlib.Path(__file__).parent/'worker.py'),run_name='__main__')
publish('child-work-complete.json',{'pid':os.getpid(),'epoch':time.time()})
while not (folder/'exit-authorized').exists():time.sleep(.02)
