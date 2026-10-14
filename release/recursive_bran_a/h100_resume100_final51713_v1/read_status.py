"""Read exact rebound CPU watcher identity and final status without touching training."""
import hashlib,json,time,subprocess,shutil
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013');f=b/'final51713-supervisor-recovery-v1'
launch=json.loads((f/'launch.json').read_text());owner=launch['process'];p=Path('/proc')/str(owner['pid'])
proc=None
if p.exists():
 s=(p/'stat').read_text().rsplit(')',1)[1].split();proc=dict(pid=owner['pid'],uid=p.stat().st_uid,tick=int(s[19]),state=s[0]);assert all(proc[k]==owner[k] for k in ('pid','uid','tick'))
status=json.loads((f/'status.json').read_text())
manifest={name:hashlib.sha256((f/name).read_bytes()).hexdigest() for name in launch['payload_manifest']};assert manifest==launch['payload_manifest']
config=json.loads((b/'receipts/configuration.json').read_text());assert hashlib.sha256(Path(config['config']).read_bytes()).hexdigest()==config['config_sha256']
result=dict(time=time.time(),watcher_process=proc,status=status,status_age=time.time()-status['time'],config_sha256=config['config_sha256'],final_scoring_started=(f/'evaluation-launch.json').exists(),actual_final_exit_present=(f/'evaluation-actual-exit.json').exists(),watcher_log_tail=(f/'watcher.log').read_text(errors='replace')[-2000:],disk_free_bytes=shutil.disk_usage(b).free,payload_manifest_stable=True)
print(json.dumps(result))
