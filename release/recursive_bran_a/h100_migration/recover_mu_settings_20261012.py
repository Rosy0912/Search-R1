"""One explicit zero-update recovery of the observed mu metadata interface failure."""
import pathlib,json,time,requests
b=pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert not pathlib.Path('/proc/3542960').exists()
assert 'ACTUAL_TRAIN_ENTRY_EXIT=1 LAUNCH='+str(b/'control/launch-1791786929.sh') in (b/'logs/slurm-3387.log').read_text()
t=(b/'logs/training.log').read_text();assert 'Complete recorded native settings required' in t and 'epoch 0, step 2' not in t
assert not list((b/'checkpoints').rglob('manifest.json'))
initial=json.loads((b/'output/fresh-initialization.json').read_text());assert initial['driver_completed_outer']==0 and initial['checkpoint_loaded'] is False
h=requests.get('http://127.0.0.1:18086/health',timeout=5).json();i=json.loads((b/'retriever/receipts/source-identity.json').read_text());assert h['pid']==i['pid'] and h['source_identity_sha256']==i['source_identity_sha256']
token=(b/'retriever/receipts/private-probe-token').read_text().strip();r=requests.post('http://127.0.0.1:18086/release-gpu',headers={'x-e1-probe-token':token},timeout=60);r.raise_for_status();release=r.json();assert release['released'] and release['gpu_worker_pid'] is None
preserved=b/'preserved/mu-settings-zero-update-3542960';preserved.mkdir(parents=True,exist_ok=False)
(b/'output').rename(preserved/'output');(b/'logs/training.log').rename(preserved/'training.log')
script=(b/'control/launch-1791786929.sh').read_text().replace('source-99e46ca','source-b824482')
assert not (b/'control/launch-ready.sh').exists();(b/'control/launch-ready.sh').write_text(script)
proof={'time':time.time(),'previous_owner':3542960,'committed_updates':0,'initial_state':initial,'release':release,'preserved':str(preserved),'new_runtime_source':'b824482','config_unchanged':True}
(b/'receipts/mu-settings-zero-update-recovery.json').write_text(json.dumps(proof));print(json.dumps(proof))
