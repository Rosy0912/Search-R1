"""Durable per-rank scalar evidence, independent of Ray console forwarding."""
import json
import os
from pathlib import Path


def append(kind, payload):
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8'
    rank=payload['rank'];assert rank in range(8)
    row=dict(kind=kind,run_id=run['run_id'],**payload)
    path=base/'logs'/f'optimizer-rank-{rank}.jsonl'
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
    try:
        data=(json.dumps(row)+'\n').encode()
        written=os.write(fd,data)
        assert written==len(data)
        os.fsync(fd)
    finally:os.close(fd)


def read_console_form(base, run_id):
    lines=[]
    for rank in range(8):
        path=Path(base)/'logs'/f'optimizer-rank-{rank}.jsonl'
        for line in path.read_text().splitlines():
            row=json.loads(line)
            assert row.pop('run_id')==run_id and row['rank']==rank
            kind=row.pop('kind')
            assert kind in ('OPTIMIZER_WRITE','OUTER_BUDGET')
            lines.append('SEARCH_G1_'+kind+'='+json.dumps(row))
    return '\n'.join(lines)
