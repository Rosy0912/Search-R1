import os,json,hashlib,time
from pathlib import Path
from huggingface_hub import snapshot_download
repo='Qwen/Qwen2.5-3B-Instruct';revision='aa8e72537993ba99e69dfaafa59ed015b17504d1'
staging=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/tree-instruct300-official-aa8e725')
staging.mkdir(parents=True,exist_ok=True)
path=snapshot_download(repo_id=repo,revision=revision,local_dir=str(staging/'model'),token=False,allow_patterns=['*.json','*.safetensors','*.txt'],max_workers=3)
files={}
for p in sorted(Path(path).iterdir()):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  files[p.name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
record={'source_repo':repo,'revision':revision,'source':'Official Hugging Face public repository; no other compute-cluster cache','local_staging':path,'files':files,'download_completed_epoch':time.time(),'cleanup_after_all_authorized_H100_targets_verified':True}
(staging/'MODEL_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n');Path('outputs/tree-instruct-300/MODEL_DOWNLOAD_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n');print('OFFICIAL_DOWNLOAD_COMPLETE',sum(x['bytes'] for x in files.values()),flush=True)
