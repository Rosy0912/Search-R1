"""Bounded CPU-only four-range relay; never prepares or submits a GPU job."""
from pathlib import Path
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback


def command(ssh, python, source, code):
    return ssh+['PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(source)+' '+shlex.quote(python)+' -c '+shlex.quote(code)]


def run(configfile):
    cfg=json.loads(Path(configfile).read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==cfg['controller_sha256']
    root=Path(cfg['local_run']);root.mkdir(exist_ok=False)
    lock=(root/'lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(at=time.time(),pid=os.getpid(),phase='VERIFY_TRANSFER_ONLY_BINDINGS',no_gpu_submission=True,no_auto_retry=True,source_commit=cfg['controller_commit'],target=cfg['target'],archive_sha256=cfg['proof']['sha256'])
    def save():
        state['at']=time.time();target=Path(cfg['state']);tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(target)
    save()
    def remote(side,code,timeout=90):
        params=cfg[side];x=subprocess.run(command(params['ssh'],params['python'],params['source'],code),text=True,capture_output=True,timeout=timeout);x.check_returncode();return x.stdout
    try:
        for side in ['source','destination']:
            check='import hashlib,json\nfrom pathlib import Path\nbase=Path('+repr(cfg[side]['source'])+')\nexpected='+repr(cfg['transport_files'])+'\nfor n,h in expected.items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h,n\nprint("VERIFIED")'
            assert remote(side,check).strip()=='VERIFIED'
        prefix=json.loads(remote('destination','import json\nfrom reproduction.seven_dataset.resume_stream import inspect_prefix\nprint(json.dumps(inspect_prefix('+repr(cfg['target'])+',0)))'))
        assert prefix==cfg['prefix'] and prefix['prefix_bytes']==0 and prefix['size']==0
        paths=[cfg['target']+'.range-'+str(i) for i in range(4)]
        state.update(phase='FOUR_VERIFIED_RANGE_STREAMS',prefix=prefix,parts=paths,transfer_started=time.time());save()
        def relay(item):
            index=item['index'];part=paths[index]
            send='import sys\nfrom reproduction.seven_dataset.parallel_archive import send\nsend('+repr(cfg['archive'])+','+repr(cfg['proof']['source_fingerprint'])+','+repr(item)+',sys.stdout.buffer)'
            receive='import sys,json\nfrom reproduction.seven_dataset.parallel_archive import receive\nprint(json.dumps(receive('+repr(part)+','+repr(item)+',sys.stdin.buffer)))'
            a=b=None
            with (root/(str(index)+'.source.err')).open('xb') as source_err,(root/(str(index)+'.target.err')).open('xb') as target_err,(root/(str(index)+'.target.out')).open('xb') as out:
                try:
                    s=cfg['source'];d=cfg['destination']
                    a=subprocess.Popen(command(s['ssh'],s['python'],s['source'],send),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=source_err)
                    b=subprocess.Popen(command(d['ssh'],d['python'],d['source'],receive),stdin=a.stdout,stdout=out,stderr=target_err);a.stdout.close()
                    br=b.wait(timeout=8000);ar=a.wait(timeout=30);assert ar==br==0,('Preserve failed partial',index,ar,br)
                finally:
                    for proc in [b,a]:
                        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
            with (root/(str(index)+'.receipt.json')).open('x') as f:json.dump(dict(item,source_returncode=ar,target_returncode=br),f,indent=2)
            return item
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:items=list(pool.map(relay,cfg['proof']['ranges']))
        state.update(phase='APPEND_FOUR_VERIFIED_RANGES',ranges=items);save()
        combine='import json\nfrom reproduction.seven_dataset.parallel_archive import combine\nprint(json.dumps(combine('+repr(cfg['target'])+','+repr(prefix)+','+repr(cfg['proof'])+','+repr(paths)+')))'
        result=json.loads(remote('destination',combine,timeout=360));assert result['sha256']==cfg['proof']['sha256'] and result['bytes']==cfg['proof']['total']
        (root/'full-target-sha.json').open('x').write(json.dumps(result,indent=2))
        state.update(phase='TRANSFER_COMPLETE_FULL_SHA_VERIFIED_NO_GPU_SUBMISSION',transfer=result);save()
    except Exception as exc:
        state.update(phase='FAILED_PRESERVE_PARTIALS_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':run(sys.argv[1])
