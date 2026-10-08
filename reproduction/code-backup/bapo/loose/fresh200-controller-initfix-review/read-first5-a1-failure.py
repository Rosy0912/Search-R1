from pathlib import Path
import hashlib, json, subprocess, datetime
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
source='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==source
receipt=json.loads((root/'scheduler-head/formal200-receipt.json').read_text())
assert receipt['source']==source and str(receipt['job_id'])=='3315' and receipt['nonce']=='bab8981d9fd14afc98923eb4ebb89e33'
run=root/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
mp=run/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json'
raw=mp.read_bytes(); assert hashlib.sha256(raw).hexdigest()=='a960351edd3bfb712ca3c89f2e278542ac67d5cfc9750256768853117d2d9f09'
manifest=json.loads(raw)
documents=[]
for item in manifest['files']:
    if Path(item['path']).name.startswith('native-initial-rank_') and item['path'].endswith('.json'):
        p=Path(item['path']); assert run in p.parents and p.resolve()==p and not p.is_symlink()
        content=p.read_bytes(); assert hashlib.sha256(content).hexdigest()==item['sha256']
        documents.append(dict(binding=item,document=json.loads(content)))
assert len(documents)==8
failed=run/'logs/first5-evidence-v3-20261007-a1/collection.failed.json'
out=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=source,job_id=3315,run_root=str(run),manifest=manifest,manifest_sha256=hashlib.sha256(raw).hexdigest(),failure=json.loads(failed.read_text()),native_initial_documents=documents,current_slurm_raw=subprocess.check_output(['scontrol','show','job','-o','3315'],text=True))
print(json.dumps(out,allow_nan=False))
