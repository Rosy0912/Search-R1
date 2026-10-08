"""Bind once to live F worker; thereafter capture only that historical inode."""
import hashlib,json,os,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
r=root/'runs/cosearch-local-gpu-five-20261003-k'
job=2961
provenance=r/'worker-log-provenance.json'
prior=json.loads(provenance.read_text()) if provenance.exists() else None
if prior:
    p=Path(prior['source']);session=Path(prior['session']);pid=prior['worker_pid']
else:
    session=(root/'session_latest').resolve()
    assert session.parent==root and session.name.startswith('session_')
    head_pid=int(session.name.rsplit('_',1)[1])
    assert f'/job_{job}/' in (Path('/proc')/str(head_pid)/'cgroup').read_text()
    candidates=[]
    for source in (session/'logs').glob('worker-*-*.out'):
        worker_pid=int(source.stem.rsplit('-',1)[1])
        proc=Path('/proc')/str(worker_pid)
        if not proc.is_dir():continue
        cgroup=(proc/'cgroup').read_text()
        if f'/job_{job}/' not in cgroup:continue
        with source.open('rb') as stream:prefix=stream.read(16384).decode(errors='replace')
        if ':task_name:main_task' in prefix or 'ray::main_task' in (proc/'cmdline').read_text(errors='replace'):
            candidates.append((source,worker_pid))
    assert len(candidates)==1, f'Expected one live own main worker, found {len(candidates)}'
    p,pid=candidates[0]
source_stat=p.stat();proc=Path('/proc')/str(pid);live=proc.is_dir()
if live:
    cgroup=(proc/'cgroup').read_text();assert f'/job_{job}/' in cgroup
    ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
    binding_at=prior['live_binding_at'] if prior else time.time()
    if prior:assert ticks==prior['process_start_ticks']
else:
    assert prior and prior['process_cgroup_verified'] and (r/'driver-exit.json').is_file()
    cgroup=prior['cgroup'];ticks=prior['process_start_ticks'];binding_at=prior['live_binding_at']
if prior:
    assert prior['job']==job and str(p)==prior['source']
    assert source_stat.st_ino==prior['source_inode'] and source_stat.st_dev==prior['source_device']
data=p.read_bytes();dest=r/'worker-training.log';temporary=dest.with_suffix('.pending')
temporary.write_bytes(data);os.replace(temporary,dest)
record=dict(captured_at=time.time(),job=job,session=str(session),source=str(p),worker_pid=pid,
 destination=str(dest),process_start_ticks=ticks,cgroup=cgroup,process_cgroup_verified=True,
 currently_live=live,live_binding_at=binding_at,source_inode=source_stat.st_ino,
 source_device=source_stat.st_dev,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
temporary=provenance.with_suffix('.pending');temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,provenance)
print(json.dumps(record))
print('\n'.join(line for line in data.decode(errors='replace').splitlines() if 'COSEARCH_' in line)[-6000:])
