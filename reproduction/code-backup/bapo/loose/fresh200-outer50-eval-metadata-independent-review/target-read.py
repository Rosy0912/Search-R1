import sys,hashlib
from pathlib import Path
import json,datetime,os,time
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
run=root/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
sys.path.insert(0,str(root/'first5-evidence-tools-3315-v4'))
from first5_helpers import bind_source
started=time.perf_counter();bind_source()
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3315,'run':str(run),'scope':'Small metadata and file presence only; no large hash, torch/HF/GPU load or acceptance','groups':{}}
for label,base in [('hf',run/'checkpoints/actor/global_step_50'),('native',run/'checkpoints/fresh200_state/outer_50_COMPLETE'),('algorithm',run/'checkpoints/algorithm/global_step_50')]:
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
actual=hashlib.sha256(json.dumps(result['groups'],sort_keys=True,allow_nan=False).encode()).hexdigest()
assert actual=='a49e4f59d260c12c1f491870935abaa559767d06f8d45efb67dc60a6944aef94'
print(json.dumps(dict(status='PASS_READONLY_CHECKPOINT50_METADATA_PRESENCE_UNCHANGED',groups_semantic_sha256=actual,groups_counts={k:len(v['files']) for k,v in result['groups'].items()},wall_seconds=time.perf_counter()-started,whole200_accepted=False,full_hash_checked=False,cpu_hf_checked=False),allow_nan=False))
