"""Generic original-record utilities extracted unchanged; no old80/125 constants."""
import json
from pathlib import Path
from formal_protection_contract_v1 import require,sha

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def validate_update_sequence(records, start, stop):
    require([v['outer_step'] for v in records] == list(range(start, stop+1)),
            'Missing/duplicate/extra actual outer update')

def raw_driver_snapshot(run):
    driver=rows(run/'audit/driver.jsonl');pids={v['pid'] for v in driver}
    require(len(pids)==1, 'Driver PID changed')
    pid=next(iter(pids));found=[]
    for session in (run/'ray').iterdir():
        if session.is_dir() and not session.is_symlink():
            found.extend((session/'logs').glob(f'worker-*-{pid}.out'))
    require(len(found)==1, 'Missing/ambiguous actual raw driver log')
    path=found[0];stat=path.stat();raw=path.read_text()
    receipt=dict(path=str(path),sha256=sha(path),bytes=stat.st_size,inode=stat.st_ino,device=stat.st_dev,
                 forwarded_path=str(run/'training.log'),forwarded_sha256=sha(run/'training.log'))
    (run/'raw-log-terminal-identity.json').write_text(json.dumps(receipt,indent=2))
    return raw,receipt
