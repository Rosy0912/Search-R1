"""Human-authorized stop of afb2d350 only, retaining complete3 and partial logs."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

ROOT=Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
RUN='afb2d3503bf64d3fa0a14467e2b3c04d'
SOURCE='a3b21f060a2d5bbdce19319902852dfbea55445b'
CONTROLLER=781626
TICK=656260418
LIBC=ctypes.CDLL(None,use_errno=True)

def read(pid):
    p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
    uid=int(next(l for l in (p/'status').read_text().splitlines() if l.startswith('Uid:')).split()[1])
    return dict(pid=pid,uid=uid,tick=int(s[19]),state=s[0],ppid=int(s[1]))

def send(r,sig):
    current=read(r['pid'])
    assert current['uid']==r['uid']==1001 and current['tick']==r['tick']
    fd=LIBC.syscall(434,ctypes.c_int(r['pid']),ctypes.c_uint(0))
    if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
    try:
        now=read(r['pid']);assert now['tick']==r['tick'] and now['uid']==1001
        if LIBC.syscall(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))<0:
            raise OSError(ctypes.get_errno(),'pidfd_send_signal')
    finally:os.close(fd)

def alive(r):
    try:
        now=read(r['pid'])
        return now['tick']==r['tick'] and not (now['state']=='Z' and not list((Path('/proc')/str(r['pid'])/'fd').glob('*')))
    except FileNotFoundError:return False

def save(path,record):
    raw=json.dumps(record,sort_keys=True).encode()
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()

def main():
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    controller=read(CONTROLLER);assert controller['tick']==TICK and controller['uid']==1001
    start=ROOT/('output-speed-resume3-'+RUN)/'start.json';s=json.loads(start.read_text())
    assert s['pid']==CONTROLLER and s['run_id']==RUN and s['source_commit']==SOURCE
    parent=ROOT/'checkpoints/training_state/global_step_3/manifest.json';m=json.loads(parent.read_text())
    assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and m['identity']['completed_step']==3
    assert all(w['actual_total_optimizer_calls']==60 and w['scheduler_last_epoch']==3 for w in m['worker_states'])
    for rank in range(4):assert (parent.parent/f'rank-{rank:02d}.pt').is_file()
    driver=read(792545);assert driver['uid']==1001
    send(driver,signal.SIGSTOP)
    try:
        time.sleep(.1);assert read(driver['pid'])['state'] in ('T','t')
        log=ROOT/'logs/recovery-a3b21f0-afb2d350.log'
        assert 'TREE_DRIVER_STAGE=' not in log.read_text(errors='replace'), 'Update preparation has started; capture newer state instead'
        assert not (parent.parent.parent/'global_step_4/manifest.json').exists()
        procs={}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:procs[int(p.name)]=read(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        owned={781621,CONTROLLER}
        while True:
            more={pid for pid,r in procs.items() if r['ppid'] in owned}
            if more<=owned:break
            owned|=more
        for pid,r in procs.items():
            if r['uid']!=1001:continue
            try:
                e=dict(x.split('=',1) for x in (Path('/proc')/str(pid)/'environ').read_bytes().decode().split('\0') if '=' in x)
                if e.get('SEARCH_R1_BASELINE_RUN_ID')==RUN and e.get('SEARCH_R1_BASELINE_COMMIT')==SOURCE:owned.add(pid)
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        assert {792545,793224,793673,793674,793675}<=owned
        records=[procs[p] for p in sorted(owned)];assert all(r['uid']==1001 for r in records)
        request=dict(protocol='recursive-b-controlled-cost80-stop-v1',run=RUN,source=SOURCE,time=time.time(),
            controller=controller,owned_processes=records,parent_manifest=str(parent),
            parent_manifest_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
            parent_completed_outer=3,partial_outer=4,no_update_entry_in_current_log=True,
            stop_authority='Latest human: stop current slow B and use GPUs 0-3 for repaired B versus old OPSD 16x5 complete-iteration cost analysis',
            preserved='Complete3 parameter/Adam/scheduler/RNG/cursor payloads and all partial4 logs remain remote',
            no_failure_time_tensor_snapshot_claimed=True)
        rp=ROOT/'receipts'/('cost80-stop-'+RUN+'-request.json');request_sha=save(rp,request)
        for r in sorted(records,key=lambda r:r['pid'],reverse=True):
            try:
                send(r,signal.SIGTERM)
                if r['pid']==driver['pid']:send(r,signal.SIGCONT)
            except (FileNotFoundError,ProcessLookupError):pass
    except BaseException:
        try:send(driver,signal.SIGCONT)
        except (FileNotFoundError,ProcessLookupError):pass
        raise
    deadline=time.monotonic()+25
    while time.monotonic()<deadline and any(alive(r) for r in records):time.sleep(.5)
    remaining=[r for r in records if alive(r)]
    for r in remaining:
        try:send(r,signal.SIGKILL)
        except (FileNotFoundError,ProcessLookupError):pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and any(alive(r) for r in records):time.sleep(.2)
    assert not any(alive(r) for r in records), 'Own kernel processes not fully released'
    terminal=dict(request,status='CONTROLLED_COST80_ALL_OWNED_KERNEL_EXITED',request_path=str(rp),
        request_sha256=request_sha,all_owned_kernel_exited=True,finished_time=time.time(),
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    tp=ROOT/'receipts'/('cost80-stop-'+RUN+'-terminal.json');sha=save(tp,terminal)
    print(json.dumps(dict(terminal=str(tp),sha256=sha,owned_count=len(records),parent_completed_outer=3)))

if __name__=='__main__':main()
