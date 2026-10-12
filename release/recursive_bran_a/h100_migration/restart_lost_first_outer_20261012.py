"""Explicit fresh replacement after the first-outer accounting NameError.

Not a checkpoint resume: the old Adam state is unavailable. Run only after
human approval, using RBA_APPROVED_LOST_FIRST_OUTER_RESTART=1.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

b = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert os.environ.get('RBA_APPROVED_LOST_FIRST_OUTER_RESTART') == '1'
assert not Path('/proc/3628041').exists()
assert not (b/'control/launch-ready.sh').exists()
assert 'ACTUAL_TRAIN_ENTRY_EXIT=1 LAUNCH=' + str(b/'control/launch-1791790654.sh') in (b/'logs/slurm-3387.log').read_text()
assert 'NameError: name \'os\' is not defined' in (b/'logs/training.log').read_text()
ranks = [json.loads(p.read_text()) for p in (b/'output/fresh-state').glob('*.json')]
assert len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8))
assert all(r['actual_optimizer_calls'] == 20 and r['scheduler_step_after'] == 1 for r in ranks)
assert not list((b/'checkpoints').rglob('*.pt'))
source = b/'source-71d4e69'
assert hashlib.sha256((source/'search_r1/baselines/tree_guard_continue.py').read_bytes()).hexdigest() == '1798ab681336671cbdfb524f8b730c587d42f5ca9eeee7980da3a8ae5e3cc7ce'
receipt_path = b/'receipts/configuration.json'
original = json.loads(receipt_path.read_text())
assert original['run_id'] == '48528afa6b6443cfbb35328fd8b0ff05'
assert hashlib.sha256(Path(original['config']).read_bytes()).hexdigest() == original['config_sha256']
preserved = b/'preserved'/('accounting-nameerror-after20-' + str(int(time.time())))
preserved.mkdir(parents=True, exist_ok=False)
receipt_path.rename(preserved/'configuration.json')
(b/'output').rename(preserved/'output')
(b/'logs/training.log').rename(preserved/'training.log')
if (b/'receipts/train-start.json').exists():
    (b/'receipts/train-start.json').rename(preserved/'train-start.json')
replacement = dict(original, run_id=uuid.uuid4().hex)
receipt_path.write_text(json.dumps(replacement))
proof = dict(time=time.time(), previous_run=original['run_id'], new_run=replacement['run_id'],
    previous_adam_calls_per_rank=20, previous_scheduler_step=1,
    discarded_unrecoverable_optimizer_state=True, exact_resume=False,
    config_sha256=original['config_sha256'], restart_from='official aa8e725',
    preserved=str(preserved), runtime_commit='71d4e69d0b098ef37df00b1267ab4b9804e5024d')
(b/'receipts/accounting-first-outer-explicit-restart.json').write_text(json.dumps(proof))
script = (b/'control/launch-1791790654.sh').read_text().replace('source-78eecc1', 'source-71d4e69')
assert script != (b/'control/launch-1791790654.sh').read_text()
(b/'control/launch-ready.sh').write_text(script)
print(json.dumps(proof), flush=True)
