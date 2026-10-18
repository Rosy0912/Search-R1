"""Inspected CPU transport replacement; four ranges, one fixed GPU submission set."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from .final150_dispatch import digest, remote, parse_submissions, write_new


def run(configfile):
    cfg = json.loads(Path(configfile).read_text()); p = Path(cfg['local'])
    old = json.loads((p/'pj-budget5-stream-config-20261018T1640.json').read_text())
    export = json.loads((p/'pj-budget5-stream-20261018T1640/export.json').read_text())
    retirement = json.loads(Path(cfg['retirement']).read_text())
    assert retirement['pid'] == 65089 and retirement['phase'] == 'CPU_STREAM_RETIRED_PARTIAL_PRESERVED'
    assert retirement['checkpoint_identity'] == export['checkpoint_identity']
    try: os.kill(65089, 0)
    except ProcessLookupError: pass
    else: raise RuntimeError('Old CPU stream still present')
    lock = open(p/'pj-ablation-dispatch-20261018T1103/lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, expected in cfg['files'].items(): assert digest(p/name) == expected, name
    run = Path(cfg['run']); run.mkdir(exist_ok=False)
    state = dict(at=time.time(), pid=os.getpid(), source=cfg['source'],
        prior_dispatcher=65089, original_dispatcher=36239, group='budget-5',
        checkpoint_identity=export['checkpoint_identity'], no_auto_retry=True,
        phase='INDEPENDENT_ZERO_SUBMISSION_AND_PREFIX_CHECK')
    def save():
        state['at'] = time.time(); f = Path(cfg['state']); t = f.with_suffix('.tmp')
        t.write_text(json.dumps(state, indent=2)+'\n'); t.replace(f)
    def helper(name):
        assert digest(p/name) == cfg['files'][name]; return (p/name).read_text()
    ls, ps, lp, pp, sdk = (old[k] for k in ['lumia_ssh','pj_ssh','lumia_python','pj_python','pj_sdk'])
    lumia, pj, source = old['lumia'], old['pj'], cfg['source']
    dest = retirement['destination']
    def call(ssh, py, code, base, timeout=180):
        return remote(ssh, py, code, base+'/sources/'+source, timeout)
    def command(ssh, py, code, base):
        return ssh+['PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(base+'/sources/'+source)
                    +' '+shlex.quote(py)+' -c '+shlex.quote(code)]
    save()
    try:
        zero_code = ('CID='+repr(export['checkpoint_identity'])+'\nB='+repr(pj)+'\n'
            +helper('source/reproduction/seven_dataset/pj_name_inspection.py')+'''
from pathlib import Path
import importlib.util
s=importlib.util.spec_from_file_location('pt','/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
names=['zhangjunming-ttt-b5-150-r'+str(i)+'-'+CID[:4] for i in (1,2,3)]+['zhangjunming-ttt-e-b5-150-r'+str(i)+'-'+CID[:8]+'-pj' for i in (1,2,3)]
z=prove_absent(m.get_cluster().client,names)
b=Path(B)
assert not (b/'ablation-imports'/CID).exists()
assert not (b/'ablation-seven-evaluations'/CID).exists()
assert not list((b/'receipts').glob('ablation-pj-budget-5-150-r*.json'))
print('ZERO='+json.dumps(z))
''')
        z=call(ps,sdk,zero_code,pj); (run/'zero.output').write_text(z)
        write_new(run/'zero.json',json.loads(z.split('ZERO=',1)[1]))
        code='from reproduction.seven_dataset.resume_stream import inspect_prefix\nimport os,json\np='+repr(dest)+'\nprint(json.dumps(inspect_prefix(p,os.stat(p).st_size)))'
        prefix=json.loads(call(ps,pp,code,pj)); assert prefix==retirement['prefix']
        code='from reproduction.seven_dataset.parallel_archive import manifest\nimport json\nprint(json.dumps(manifest('+repr(export['archive'])+','+str(prefix['prefix_bytes'])+','+str(export['archive_bytes'])+','+repr(export['archive_sha256'])+')))'
        proof=json.loads(call(ls,lp,code,lumia,300))
        assert proof['prefix_sha256']==prefix['prefix_sha256']
        write_new(run/'source-ranges.json',proof); write_new(run/'target-prefix.json',prefix)
        paths=[dest+'.parallel-'+cfg['tag']+'-'+str(i) for i in range(4)]
        state.update(phase='FOUR_VERIFIED_RANGE_STREAMS', prefix_bytes=prefix['prefix_bytes'], transfer_started=time.time())
        save()
        def copy(item):
            i=item['index']; part=paths[i]
            send='import sys\nfrom reproduction.seven_dataset.parallel_archive import send\nsend('+repr(export['archive'])+','+repr(proof['source_fingerprint'])+','+repr(item)+',sys.stdout.buffer)'
            receive='import sys,json\nfrom reproduction.seven_dataset.parallel_archive import receive\nprint(json.dumps(receive('+repr(part)+','+repr(item)+',sys.stdin.buffer)))'
            a=b=None
            with (run/(str(i)+'.source.err')).open('wb') as se, (run/(str(i)+'.target.err')).open('wb') as te, (run/(str(i)+'.target.out')).open('wb') as out:
                try:
                    a=subprocess.Popen(command(ls,lp,send,lumia),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=se)
                    b=subprocess.Popen(command(ps,pp,receive,pj),stdin=a.stdout,stdout=out,stderr=te);a.stdout.close()
                    br=b.wait(timeout=6500); ar=a.wait(timeout=30)
                    assert ar==br==0, ('Range failed; preserve all partials',i,ar,br)
                finally:
                    for proc in (b,a):
                        if proc is not None and proc.poll() is None: proc.kill();proc.wait()
            write_new(run/(str(i)+'.receipt.json'),dict(item,source_rc=ar,target_rc=br))
            return item
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(copy,proof['ranges']))
        state.update(phase='APPEND_VERIFIED_LOCAL_RANGES', ranges=results);save()
        code='import json\nfrom reproduction.seven_dataset.parallel_archive import combine\nprint(json.dumps(combine('+repr(dest)+','+repr(prefix)+','+repr(proof)+','+repr(paths)+')))'
        result=json.loads(call(ps,pp,code,pj,300));write_new(run/'full-target-sha.json',result)
        state.update(phase='PJ_FROZEN_INPUT_PREFLIGHT',transfer_finished=time.time(),transfer=result);save()
        worker=old['worker_source']
        raw=remote(ps,pp,'PROOF='+repr(export)+'\nARCHIVE='+repr(dest)+'\nCOMMIT='+repr(worker)+'\n'+helper(old['stage_helper']),pj+'/sources/'+worker,1200)
        (run/'preflight.output').write_text(raw);preflight=json.loads(raw.strip().splitlines()[-1]);write_new(run/'preflight.json',preflight)
        state.update(phase='SUBMITTING_FIXED_THREE_ONCE');save()
        raw=remote(ps,sdk,'PREFLIGHT='+repr(preflight)+'\n'+helper(cfg['submit_helper']),pj+'/sources/'+worker,600)
        (run/'submission.output').write_text(raw);submissions=parse_submissions(raw)
        for item in submissions:write_new(p/('seven-ablation-pj-budget-5-150-r'+str(item['repeat_index'])+'-submission.json'),item)
        state.update(phase='THREE_SUBMITTED_NO_REPLAY',submissions=submissions);save()
    except Exception as exc:
        state.update(phase='FAILED_PRESERVE_EVIDENCE_NO_AUTO_RETRY',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':run(sys.argv[1])
