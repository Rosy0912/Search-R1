"""Read final watcher terminal evidence without signalling or changing remote files."""
import hashlib
import json
from pathlib import Path
import time

base=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
control=base/'final51713-supervisor-0b3acde'
def proc(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,uid=path.stat().st_uid,state=fields[0],tick=int(fields[19]))
    except FileNotFoundError:
        return dict(pid=pid,absent=True)
def read(path):
    try: return json.loads(path.read_text())
    except FileNotFoundError: return None
launch=read(control/'watcher-launch.json')
publication=read(control/'publication.json')
source=Path(publication['deployed'])
assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==digest for name,digest in publication['files'].items())
log=control/'watcher.log'
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf8',errors='replace')
print(json.dumps(dict(time=time.time(),launch=launch,watcher=proc(launch['pid']),training_owner=proc(3714909),status=read(control/'status.json'),watcher_log_tail=tail,source_hashes_verified=True,terminal_receipts={name:read(control/name) for name in ['native-final-complete.json','complete200-boundary.json','evaluation-launch.json','evaluation-actual-exit.json']},scope='Read-only CPU watcher terminal evidence; no training mutation')))
