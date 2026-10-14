"""Permit retry only when a real failed entry never reached initialization."""
import json
from pathlib import Path


def check(base, entry, proc_root=Path('/proc')):
    base = Path(base).resolve()
    process = Path(proc_root) / str(entry['pid'])
    if process.exists():
        # A reused PID is not this original owner; unreadable state stays unknown.
        fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        if int(fields[19]) == entry['tick'] and process.stat().st_uid == entry['uid']:
            raise ValueError('Previous entry is still present; retry cannot duplicate it')
    wait_path = base / 'receipts' / 'actual-parent-wait.json'
    wait = json.loads(wait_path.read_text())
    if (wait.get('pid') != entry['pid'] or type(wait.get('rc')) is not int
            or wait['rc'] == 0 or wait.get('group') != entry['group']):
        raise ValueError('Retry requires this exact failed entry’s actual nonzero parent wait')
    output = Path(entry['output']).resolve()
    if not output.is_relative_to(base / 'output'):
        raise ValueError('Previous entry output is outside its group')
    if any((base / 'output').rglob('fresh-initialization.json')):
        raise ValueError('This group already reached initialization; use full native recovery')
    if any(path.is_file() for path in (base / 'checkpoints').rglob('*')):
        raise ValueError('Checkpoint payload exists; no fresh retry or Adam reset')
    if any((base / 'output').rglob('completed-outers.jsonl')):
        raise ValueError('Committed rounds exist; unstarted retry is forbidden')
    return dict(previous_pid=entry['pid'], previous_tick=entry['tick'], previous_uid=entry['uid'],
        actual_parent_wait=str(wait_path), actual_parent_wait_rc=wait['rc'],
        previous_owner_absent=True, no_fresh_initialization=True, no_checkpoint_payload=True,
        scope='Retry pre-initialization failure only; no lost Adam or rollout work')
