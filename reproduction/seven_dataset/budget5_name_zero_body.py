"""Executed only after the frozen exact-name inspection definitions are prepended."""
from pathlib import Path
import importlib.util
import os
import time
from reproduction.seven_dataset.ablation_pj import pj_job_name

spec = importlib.util.spec_from_file_location('pt',
    '/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
client = module.get_cluster().client
names = [pj_job_name('budget-5', i, CID) for i in (1, 2, 3)] + [
    'zhangjunming-ttt-e-b5-150-r' + str(i) + '-' + CID[:8] + '-pj' for i in (1, 2, 3)]
proof = prove_absent(client, names)
base = Path(B)
paths = [Path(DEST), base / 'ablation-imports' / CID,
         base / 'ablation-seven-evaluations' / CID]
assert all(not path.exists() for path in paths)
assert not list((base / 'receipts').glob('ablation-pj-budget-5-150-r*.json'))
free = os.statvfs(base).f_bavail * os.statvfs(base).f_frsize
assert free >= 450_000_000_000
proof.update(at=time.time(), absent_paths=[str(path) for path in paths], free_bytes=free)
print('ZERO_PROOF=' + json.dumps(proof))
