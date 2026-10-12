"""Produce OBSERVED200 inventory from real immutable artifacts; never a receipt."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import types


def load_common(expected_sha):
    p=Path(__file__).resolve().parent/'common.py'
    if not (re.fullmatch('[0-9a-f]{64}',expected_sha) and p.resolve(strict=True)==p and p.stat().st_uid==os.getuid()):raise ValueError('Exact owned common source required')
    before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd);raw=b''
        while True:
            chunk=os.read(fd,1024**2)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    fields=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    if not stat.S_ISREG(opened.st_mode) or any(tuple(getattr(x,k) for k in fields)!=tuple(getattr(before,k) for k in fields) for x in (opened,after,p.stat())) or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Common changed before exact-byte compilation')
    module=types.ModuleType('final200_common');module.__file__=str(p)
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),module.__dict__)
    module.read_file(p,expected_sha)
    return module


def observe(a,c):
    own=c.read_file(Path(__file__).resolve())[1]
    execution=c.execution_identity(a.execution_job_id,a.execution_step_id)
    context=c.accepted_context(a.accepted5_receipt_sha256);records=dict(context['records'])
    def inventory(expected):
        result={}
        for role,path in sorted(expected.items()):
            _,rec=c.read_file(path,root=c.RUN);result[role]=rec;records[str(path)]=rec
        return result
    actor=c.RUN/'checkpoints/actor/global_step_200';state=actor/'training_state';driver=c.RUN/'full_state/global_step_200'
    c.require({p.name for p in actor.iterdir()}==c.HF_NAMES|{'training_state'},'HF200 directory has extra/missing members')
    c.require({p.name for p in state.iterdir()}=={f'rank_{rank}.{suffix}' for rank in range(8) for suffix in ('pt','json')},'Native200 directory has extra/missing members')
    c.require({p.name for p in driver.iterdir()}=={'driver.pt','driver.json'},'Driver200 directory has unexpected members')
    for outer in range(1,201):
        directory=c.RUN/'trajectories'/f'outer_{outer:06d}'
        c.require(directory.resolve(strict=True)==directory and {p.name for p in directory.iterdir()}==c.ARCHIVE_NAMES,'Raw diagnostic member inventory differs')
    checkpoint=inventory(c.checkpoint_paths());archives=inventory(c.archive_paths())
    for role,rec in archives.items():
        if role in context['manifest']['files']:
            c.require(dict(path=rec['physical_path'],sha256=rec['sha256'])==context['manifest']['files'][role],'Observed1..5 archive changed since historical signature')
    c.require(len({(r['device'],r['inode']) for r in list(checkpoint.values())+list(archives.values())})==len(checkpoint)+len(archives),'Observed200 member inode aliases')
    ranks=[]
    for rank,w in enumerate(context['workers']):
        role=f'actor/training_state/rank_{rank}.json';rec=checkpoint[role]
        report,_=c.read_json(rec['physical_path'],rec['sha256'],c.RUN);c.rank_report(report,rank,200,'FULL_STATE_SERIALIZED_OK',w['pid'])
        c.require(report['object_ids']==w['object_ids'],'Stored200 objects changed since signed5')
        pt=checkpoint[f'actor/training_state/rank_{rank}.pt']
        c.require(report['path']==pt['physical_path'] and report['sha256']==report['state_sha256']==pt['sha256'],'Serialized200 PT/report binding differs')
        ranks.append(report)
    rec=checkpoint['driver/driver.json'];report,_=c.read_json(rec['physical_path'],rec['sha256'],c.RUN)
    for key,target in dict(status='DRIVER_STATE_SERIALIZED_OK',step=200,pid=context['manifest']['driver_pid'],uid=os.getuid(),job_id='3304').items():c.typed(report.get(key),target,'Serialized200 driver header differs: '+key)
    c.require(report['state_sha256']==checkpoint['driver/driver.pt']['sha256'],'Serialized200 driver PT binding differs')
    records[own['physical_path']]=own
    common=c.read_file(Path(c.__file__),a.common_sha256)[1];records[common['physical_path']]=common
    result=dict(version=1,kind=c.KIND,source_commit=c.SOURCE,config_sha256=c.CONFIG,job_id='3304',run_root=str(c.RUN),source_root=str(c.REPO),profile='multitraj200',checkpoint_step=200,horizon=200,warmup_steps=57,world_size=8,
        utc=c.utc(),accepted5=context['accepted5'],checkpoint_files=checkpoint,archive_files=archives,
        rank_reports=ranks,driver_report=report,raw_statuses=dict(ranks=[r['status'] for r in ranks],driver=report['status']),
        source_and_historical_anchor_records=context['records'],observer_source_sha256=own['sha256'],common_source_sha256=common['sha256'],execution_identity=execution,
        scope='Physical original file inventory only; no native payload/HF load/restore/effect/terminal qualification')
    c.validate_inventory(result);c.recheck(records)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('common-sha256','accepted5-receipt-sha256','execution-job-id','execution-step-id','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();c=load_common(a.common_sha256);result=observe(a,c);out=c.write_exclusive(a.output,result)
    print(c.canonical(dict(kind=c.KIND,output=out,scope=result['scope'])).decode())


if __name__=='__main__':main()
