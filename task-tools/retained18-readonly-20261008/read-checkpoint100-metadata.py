from pathlib import Path
import json,datetime,os,time
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
run=root/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3315,'run':str(run),'scope':'Small metadata and file presence only; no large hash, torch/HF/GPU load or acceptance','groups':{}}
for label,base in [('hf',run/'checkpoints/actor/global_step_100'),('native',run/'checkpoints/fresh200_state/outer_100_COMPLETE'),('algorithm',run/'checkpoints/algorithm/global_step_100')]:
 files=[];meta={}
 if base.is_dir():
  for p in sorted(base.rglob('*')):
   if p.is_file():
    st=p.stat();files.append({'relative':str(p.relative_to(base)),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'symlink':p.is_symlink()})
    if (p.name in ('status.json','driver-reload.json','state.json','model.safetensors.index.json') or (p.parent.name=='ranks' and p.suffix=='.json')) and st.st_size<2000000:
     try:meta[str(p.relative_to(base))]=json.loads(p.read_text())
     except ValueError:meta[str(p.relative_to(base))]={'status':'WRITE_IN_PROGRESS_OR_MALFORMED'}
 result['groups'][label]={'directory_exists':base.is_dir(),'files':files,'small_metadata':meta}
st=os.statvfs(run);result['disk_available_bytes']=st.f_bavail*st.f_frsize
print(json.dumps(result,allow_nan=False))
