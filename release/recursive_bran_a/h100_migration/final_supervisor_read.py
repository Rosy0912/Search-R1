"""Read the task-owned CPU final watcher and current native training receipts."""
import hashlib
import json
from pathlib import Path
import time

base=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
control=base/'final51713-supervisor-0b3acde'
launch=json.loads((control/'watcher-launch.json').read_text())
proc=Path('/proc')/str(launch['pid'])
fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
assert proc.stat().st_uid==launch['uid'] and int(fields[19])==launch['starttick']
assert fields[0]!='Z'
publication=json.loads((control/'publication.json').read_text())
source=Path(publication['deployed'])
assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    for name,digest in publication['files'].items())
status=json.loads((control/'status.json').read_text())
assert status['pid']==launch['pid']
now=time.time()
assert now-status['time']<90
results=[]
for path in (base/'output/attempt-results').glob('*.json'):
    try:
        record=json.loads(path.read_text())
        results.append(record)
    except (FileNotFoundError,ValueError): pass
print(json.dumps(dict(time=now,watcher=launch,process_state=fields[0],
    current_status=status,status_age=now-status['time'],
    deployed_file_hashes_verified=True,
    watcher_log_tail=(control/'watcher.log').read_text(errors='replace')[-2000:],
    attempt_receipts=results[-3:],
    scope='CPU watcher and native small receipts; not final GPU or full200 acceptance')))
