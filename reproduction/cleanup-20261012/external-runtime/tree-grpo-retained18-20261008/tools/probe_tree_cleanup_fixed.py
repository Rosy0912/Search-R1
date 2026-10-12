"""Actual Linux own Slurm cgroup orphan cleanup; no GPU work or foreign signals."""
import json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from tree_shared_owned_runtime_fixed import ROOT, OwnedProcesses, cgroup_scope, cgroup_pids, gpu_sample, sha

def main():
    job=os.environ['SLURM_JOB_ID']; out=ROOT/'logs'/f'tree-cleanup-fixed-{job}'; out.mkdir()
    scope,relative=cgroup_scope(job)
    protected=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
    before_gpu=gpu_sample()
    owned=OwnedProcesses(scope,relative)
    try:
        # Double-fork an orphan that ignores TERM and binds a unique owned port.
        code="""import os,signal,socket,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
pid=os.fork()
if pid: os._exit(0)
os.setsid()
pid=os.fork()
if pid: os._exit(0)
s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen()
print(os.getpid(),s.getsockname()[1],flush=True)
while True: time.sleep(1)
"""
        parent=owned.spawn([sys.executable,'-c',code],out/'orphan.log')
        deadline=time.monotonic()+10
        while not (out/'orphan.log').read_text().strip():
            assert time.monotonic()<deadline; time.sleep(.1)
        orphan,port=map(int,(out/'orphan.log').read_text().split())
        live=owned.spawn([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)'],out/'live.log')
        time.sleep(.3); own_before=sorted(cgroup_pids(scope)-owned.baseline)
        assert orphan in own_before and live.pid in own_before
        started=time.monotonic(); owned.cleanup(); elapsed=time.monotonic()-started
        assert protected.poll() is None
        remaining={str(p):Path(f'/proc/{p}/stat').read_text().rsplit(')',1)[1].split()[0] for p in cgroup_pids(scope)-owned.baseline if Path(f'/proc/{p}/stat').exists()}
        assert all(v=='Z' for v in remaining.values())
        with socket.socket() as s: assert s.connect_ex(('127.0.0.1',port))!=0
        after_gpu=gpu_sample(); assert before_gpu['processes']==after_gpu['processes']
        result=dict(status='OWN_SCOPE_LINUX_CLEANUP_PASS',job_id=job,gpus_allocated=0,scope=str(scope),protected_baseline_pid=protected.pid,protected_alive=True,orphan_pid=orphan,killed_live_pid=live.pid,closed_owned_port=port,own_before=own_before,residual=remaining,cleanup_residual=owned.cleanup_residual,wall_seconds=elapsed,gpu_before=before_gpu,gpu_after=after_gpu,helper_sha256=sha(ROOT/'tools/tree_shared_owned_runtime_fixed.py'),probe_sha256=sha(__file__))
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
    finally:
        if protected.poll() is None: protected.terminate()
        protected.wait(timeout=10)
if __name__=='__main__': main()
