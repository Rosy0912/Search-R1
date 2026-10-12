"""One explicit zero-update recovery of the observed fresh H100 profile binding failure."""
import pathlib,json,time,requests
b=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert not pathlib.Path('/proc/3578892').exists()
assert 'ACTUAL_TRAIN_ENTRY_EXIT=1 LAUNCH='+str(b/'control/launch-1791788076.sh') in (b/'logs/slurm-3387.log').read_text()
t=(b/'logs/training.log').read_text();assert 'No-full-beta profile exceeds authorized zero-KL behavior-mu scope' in t and 'epoch 0, step 2' not in t
assert not list((b/'checkpoints').rglob('manifest.json'))
initial=json.loads((b/'output/fresh-initialization.json').read_text());assert initial['driver_completed_outer']==0 and initial['checkpoint_loaded'] is False
h=requests.get('http://127.0.0.1:18086/health',timeout=5).json();i=json.loads((b/'retriever/receipts/source-identity.json').read_text());assert h['pid']==i['pid'] and h['source_identity_sha256']==i['source_identity_sha256']
token=(b/'retriever/receipts/private-probe-token').read_text().strip();r=requests.post('http://127.0.0.1:18086/release-gpu',headers={'x-e1-probe-token':token},timeout=60);r.raise_for_status();release=r.json();assert release['released'] and release['gpu_worker_pid'] is None
preserved=b/'preserved/fresh-profile-zero-update-3578892';preserved.mkdir(parents=True,exist_ok=False)
(b/'output').rename(preserved/'output');(b/'logs/training.log').rename(preserved/'training.log')
script=(b/'control/launch-1791788076.sh').read_text().replace('source-b824482','source-b103cbc')
assert not (b/'control/launch-ready.sh').exists();(b/'control/launch-ready.sh').write_text(script)
proof={'time':time.time(),'previous_owner':3578892,'committed_updates':0,'initial_state':initial,'release':release,'preserved':str(preserved),'new_runtime_source':'b103cbc','config_unchanged':True}
(b/'receipts/fresh-profile-zero-update-recovery.json').write_text(json.dumps(proof));print(json.dumps(proof))
