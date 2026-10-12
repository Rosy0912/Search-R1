"""Read native stage/rank ledgers; no model import, signal, or node write."""
import json
import pathlib
import time

base = pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
entry = json.loads((base / 'receipts/train-start.json').read_text())
proc = pathlib.Path('/proc') / str(entry['pid'])
state = None
if proc.exists():
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    env = (proc / 'environ').read_bytes().split(b'\0')
    # /proc/environ describes startup environment, not Python's later updates.
    assert entry['run_id'] == '1c806d15752d44a7ac99687f0737d7e3'
    assert entry['pid'] == 3714909 and proc.stat().st_uid == 0 and int(stat[19]) == 658086351
    state = dict(uid=proc.stat().st_uid, starttick=int(stat[19]), state=stat[0],
        run_in_startup_environment=('E1_RUN_ID=' + entry['run_id']).encode() in env)
stages = []
for line in (base / 'logs/training.log').read_text(errors='replace').splitlines():
    if 'TREE_DRIVER_STAGE=' in line:
        try:
            stages.append(json.JSONDecoder().raw_decode(line.split('TREE_DRIVER_STAGE=', 1)[1])[0])
        except ValueError:
            pass
ranks = {}
incomplete_reads = []
for path in sorted((base / 'output/rank-audit/policy-safety').glob('*.json')):
    try:
        record = json.loads(path.read_text())
    except (ValueError, FileNotFoundError):
        incomplete_reads.append(path.name)
        continue
    ranks[str(record['rank'])] = dict(
        file=path.name, outer=record['outer_step'], attempt=record.get('attempt_id'),
        phase=record['phase'], committed_this_outer=record['committed_optimizer_calls'],
        allowed=record['decision'].get('allowed'), reasons=record['decision'].get('reasons'),
        gradient_preclip=record.get('measured_gradient_preclip'),
        gradient_postclip=record.get('measured_gradient_postclip'))
print(json.dumps(dict(time=time.time(), entry=entry, process=state,
    stages=stages[-8:], native_latest_per_rank=ranks, concurrent_incomplete_reads=incomplete_reads,
    scope='Native live scalar ledgers, not complete-outer or independent Adam tensor acceptance')))
