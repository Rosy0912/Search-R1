"""Read-only; no process signals, no credential reads, no submission."""
import pathlib, json, hashlib, subprocess, time, os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006')
run=root/'runs/bapo-instruct300-formal300-3292'
common=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
result={'observed_unix':time.time(),'host':os.uname().nodename,'old_job_id':3292,'own_remaining':[],'files':{}}
assert os.uname().nodename=='h100-5'
for name,path in [('model_manifest',common/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json'),('data_manifest',common/'common-data/manifest.json'),('processes_before',run/'processes-before.csv'),('gpus_exit',run/'gpus-exit.csv'),('cleanup',run/'own-process-cleanup.json')]:
 assert path.is_file(),str(path)
 blob=path.read_bytes();result['files'][name]={'path':str(path),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'content':blob.decode()}
assert result['files']['data_manifest']['sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert result['files']['model_manifest']['sha256']=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
marker=('BAPO_RUN_ID='+str(run)).encode()
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  env=(p/'environ').read_bytes().split(b'\0')
  if marker in env: result['own_remaining'].append({'pid':int(p.name),'stat':(p/'stat').read_text(),'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')})
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
result['current_gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True)
result['current_gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv'],text=True)
print(json.dumps(result,allow_nan=False))
