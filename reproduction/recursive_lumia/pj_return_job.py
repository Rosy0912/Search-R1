"""One registered Slurm allocation: independent cold restore then native continuation."""
import json, os, subprocess, sys, time
from pathlib import Path
from .pj_return import load, validate_allocation, validate_probe, dump_new


def main():
    path=Path(os.environ['RECURSIVE_RETURN_BINDING']); b,r,m=load(path,verify_members=True)
    job=validate_allocation(b,r)
    stat=os.statvfs(path.parent)
    if stat.f_bavail*stat.f_frsize < 600_000_000_000:
        raise RuntimeError('Lumia lacks full-state write headroom; no training started')
    result=path.parent/('return-probe-'+job+'.json')
    if result.exists():raise ValueError('Probe already attempted in this allocation')
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc-per-node=8','-m','reproduction.recursive_lumia.pj_return_probe'],check=True)
    validate_probe(json.loads(result.read_text()),b,m,job)
    command=[sys.executable,'-m','search_r1.baselines.preflight','--',sys.executable,
        '-m','reproduction.recursive_lumia.pj_return_entry','--receipt',str(path.parent/'return-configuration.json')]
    child=subprocess.Popen(command); rc=child.wait()
    dump_new(path.parent/('return-parent-wait-'+job+'.json'),dict(at=time.time(),pid=child.pid,
        job=job,rc=rc,source=b['source_commit'],command=command,scope='Actual child wait return'))
    raise SystemExit(rc if rc>=0 else 128-rc)

if __name__=='__main__':main()
