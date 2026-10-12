"""Project subprocess ownership with PID/start ticks/session/UID and durable ledger.

Only bound project sessions are eligible; unknown members after leader exit cause
safe refusal. Never enumerate or stop another project's Ray service by name.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    p=Path('/proc')/str(pid);text=(p/'stat').read_text();f=text[text.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(f[19]),sid=int(f[3]),uid=p.stat().st_uid,state=f[0],
                command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))


def same(record):
    try:
        current=identity(record['pid'])
        return all(current[k]==record[k] for k in ('start_ticks','sid','uid'))
    except (FileNotFoundError,ProcessLookupError):return False


def remaining_cuda_contexts(records, rows):
    """NVML reports PIDs only; a reused birth is another task's context."""
    births = {}
    for record in records:
        births.setdefault(record['pid'],set()).add(tuple(record[key] for key in ('start_ticks','sid','uid')))
    remaining = []; reused = []
    for row in rows:
        pid = int(row[1])
        if pid not in births: continue
        try: actual = identity(pid)
        except (FileNotFoundError,ProcessLookupError):
            # A recorded PID still reported by NVML without /proc may be an
            # unwinding owned context. Wait for release; never signal it.
            remaining.append(row); continue
        if tuple(actual[key] for key in ('start_ticks','sid','uid')) in births[pid]:
            remaining.append(row)
        else: reused.append(pid)
    return remaining, sorted(set(reused))


def owned_session_survivors(leaders):
    """Read-only private-session audit, including numeric SID/leader PID reuse.

    A disappeared leader can leave owned children. A proven different birth
    which is itself the leader of that numeric SID belongs to a new session.
    Shared caller sessions must not be supplied as private owned sessions.
    """
    sessions = {}
    for leader in leaders:
        assert leader['pid'] == leader['sid'] and leader['start_ticks'] > 0
        key = (leader['sid'],leader['uid'])
        sessions.setdefault(key,set()).add(leader['start_ticks'])
    survivors = []; reused = set()
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try: actual = identity(int(path.name))
        except (FileNotFoundError,ProcessLookupError): continue
        except PermissionError:
            try: assert path.stat().st_uid != os.getuid(), 'Cannot inspect a potentially owned process'
            except FileNotFoundError: pass
            continue
        key = (actual['sid'],actual['uid'])
        if actual['state'] in ('Z','X') or key not in sessions: continue
        recorded_births = sessions[key]
        try: current_leader = identity(actual['sid'])
        except (FileNotFoundError,ProcessLookupError): current_leader = None
        if current_leader is not None and current_leader['pid'] == current_leader['sid'] == actual['sid'] \
                and current_leader['start_ticks'] not in recorded_births:
            reused.add(actual['sid']); continue
        survivors.append(actual)
    return survivors, sorted(reused)


def signal_owned(record, sig):
    fd=os.pidfd_open(record['pid'])
    try:
        assert same(record), 'Owned identity changed before signal'
        signal.pidfd_send_signal(fd,sig)
    finally:os.close(fd)


def publish(path, data):
    tmp=path.with_suffix('.json.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class OwnedProcesses:
    def __init__(self,path,run_id):
        self.path=Path(path);self.run_id=run_id;self.jobs={}
        if self.path.exists():
            saved=json.loads(self.path.read_text());assert saved['run_id']==run_id
            self.jobs=saved['jobs'];self.recovery_only=True
        else:
            self.recovery_only=False;self.save()
    def save(self):publish(self.path,dict(run_id=self.run_id,jobs=self.jobs))
    def start(self,name,args,env,log,cwd):
        assert not self.recovery_only, 'Recovered ledger permits cleanup only, never duplicate launch'
        assert name not in self.jobs
        with Path(log).open('a') as output:
            process=subprocess.Popen(args,env=env,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        # Popen returns after exec; bind before poll/wait can reap a fast-exit leader.
        leader=identity(process.pid);assert leader['sid']==leader['pid'] and leader['uid']==os.getuid()
        self.jobs[name]=dict(leader=leader,ledger={str(leader['pid']):leader},log=str(log),state='STARTED',arguments=args)
        self.save();self.observe(name)
        return process
    def observe(self,name):
        job=self.jobs[name];leader=job['leader'];live=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:r=identity(int(p.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):continue
            if r['state']!='Z' and r['sid']==leader['sid'] and r['uid']==leader['uid']:live.append(r)
        valid_leader=same(leader)
        for r in live:
            old=job['ledger'].get(str(r['pid']))
            if old:
                assert all(old[k]==r[k] for k in ('start_ticks','sid','uid')), 'Recorded subprocess changed identity'
                if old['command']!=r['command']:
                    job.setdefault('command_history',[]).append(dict(pid=r['pid'],before=old['command'],after=r['command'],unix=time.time()))
                    job['ledger'][str(r['pid'])]=r
            else:
                assert valid_leader, 'Unbound process after session leader exit'
                job['ledger'][str(r['pid'])]=r
        self.save();return live
    def stop(self,name):
        job=self.jobs[name];leader=job['leader']
        if job['state']=='STOPPED':return
        try:
            self.observe(name);job['state']='CLEANUP_PENDING';self.save()
            if same(leader) and identity(leader['pid'])['state']!='Z':signal_owned(leader,signal.SIGINT)
            end=time.monotonic()+15
            while time.monotonic()<end and self.observe(name):time.sleep(.2)
            for sig in (signal.SIGTERM,signal.SIGKILL):
                for r in self.observe(name):
                    try:signal_owned(r,sig)
                    except ProcessLookupError:pass
                end=time.monotonic()+3
                while time.monotonic()<end and self.observe(name):time.sleep(.2)
                if not self.observe(name):break
            assert not self.observe(name), 'Owned component cleanup incomplete'
            job['state']='STOPPED';self.save()
        except Exception as e:
            job.update(state='CLEANUP_FAILED',cleanup_error=repr(e));self.save();raise
    def stop_all(self):
        errors=[]
        for name in reversed(self.jobs):
            try:self.stop(name)
            except Exception as e:errors.append(dict(name=name,error=repr(e)))
        if errors:raise RuntimeError(json.dumps(errors))
