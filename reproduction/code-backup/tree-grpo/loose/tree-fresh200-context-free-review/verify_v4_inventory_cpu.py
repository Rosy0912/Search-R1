"""Review frozen V4 builder only; never read mutable training source or run services."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
V4=HERE.parent/'tree-fresh200-mu-runtime-v4'
SOURCE=HERE.parent/'tree-behavior-mu-ppo'
COMMON=HERE.parent/'tree-fresh200-dispatch-preparation/fresh200_dispatch_common.py'
HEAD='1e06c6aa6f54eee5de8e0a29dd50efea76111c10'
def sha(raw):return hashlib.sha256(raw).hexdigest()
manifest_raw=(V4/'FILE_HASHES_V4.json').read_bytes()
assert sha(manifest_raw)=='13903f9befd47387dc22ca53f69597000a72e60f783cf3c7e49d6e986a440501'
owner_raw=(V4/'SOURCE_CPU_RECEIPT_V4.json').read_bytes()
assert sha(owner_raw)=='aa8bf369aaca362f8f5caa2398d8bc3ac73b8fec0af7a9b58b2962542d2ddc41'
manifest=json.loads(manifest_raw)
raws={name:(V4/name).read_bytes() for name in manifest['files']}
assert all(sha(raws[name])==digest for name,digest in manifest['files'].items())
common_raw=COMMON.read_bytes()
assert sha(common_raw)=='d45b2be2abdd91b922d243f4bfb5fa207de207cd6b07ab06e75c9b6290ae3af0'
original_audit=json.loads((HERE/'BYTE_AUDIT.json').read_bytes())
source_names=('verl/workers/fsdp_workers.py','search_r1/baselines/sampling_consistency.py',
              'verl/trainer/ppo/ray_trainer.py','verl/trainer/config/ppo_tree_fresh200.yaml')
source_bytes={}
for name in source_names:
    # Read the already audited immutable commit object, not working-tree bytes.
    raw=subprocess.check_output(['git','show',HEAD+':'+name],cwd=SOURCE)
    assert sha(raw)==original_audit['source_files'][name]['sha256']
    source_bytes[name]=raw

with tempfile.TemporaryDirectory(prefix='v4-independent-cpu-',dir=HERE) as tmp:
    root=Path(tmp)
    frozen=root/'runtime-v4';frozen.mkdir()
    for name,raw in raws.items():
        path=frozen/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (frozen/'FILE_HASHES_V4.json').write_bytes(manifest_raw)
    (frozen/'SOURCE_CPU_RECEIPT_V4.json').write_bytes(owner_raw)
    for name,raw in source_bytes.items():
        path=root/'tree-behavior-mu-ppo'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    path=root/'tree-fresh200-dispatch-preparation/fresh200_dispatch_common.py'
    path.parent.mkdir();path.write_bytes(common_raw)
    environment=dict(__import__('os').environ,PYTHONDONTWRITEBYTECODE='1')
    completed=subprocess.run([sys.executable,'-m','unittest','-v','test_package_inventory_cpu'],
                             cwd=frozen,env=environment,text=True,capture_output=True)
    log=completed.stdout+completed.stderr
    (HERE/'v4_inventory_cpu.log').write_text(log)
    print(log,end='')
    results={'wrapper_inventory_sha256':sha(manifest_raw),'owner_receipt_sha256':sha(owner_raw),
             'all_manifest_files_rehashed':len(raws),'fixed_wrapper_count':len(manifest['wrapper_files']),
             'all_frozen_bytes_match':True,'source_fixture_only_head':HEAD,
             'source_fixture_sha256':{name:sha(raw) for name,raw in source_bytes.items()},
             'mutable_training_source_read':False,'mutable_helpers_read':False,
             'old_frozen_review_common_sha256':sha(common_raw),
             'command':'python3 -m unittest -v test_package_inventory_cpu (isolated immutable fixtures)',
             'exit_code':completed.returncode,'scope':'actual V4 builder/all15/asset selection; source and review metadata fixture only',
             'execution':{'remote_calls':0,'real_scheduler_calls':0,'model_assets':0,'gpu_operations':0}}
    (HERE/'V4_CPU_RESULTS.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    assert completed.returncode==0
