"""Bounded final150 CPU replacement: frozen workers, stream, no replay."""
import fcntl,hashlib,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
from .stream_copy import transfer

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_retirement(old,r):
    if (old['pid']!=13526 or old['groups']['budget-5']!={'phase':'WAITING_FINAL150'} or r.get('retired_pid')!=old['pid'] or not r.get('zero_mutation_verified') or not r.get('confirmed_exited') or r.get('child_pids')):raise ValueError('Not an inspected idle CPU retirement')
    try:os.kill(old['pid'],0)
    except ProcessLookupError:return
    raise ValueError('Prior dispatcher still exists')
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def remote(ssh,py,code,source,timeout=120):
    cmd='PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH='+shlex.quote(source)+' '+shlex.quote(py)+' -c '+shlex.quote(code)
    r=subprocess.run(ssh+[cmd],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr[-4000:]+'\n'+r.stdout[-3000:])
    return r.stdout

def parse_submissions(raw):
    ss=[json.loads(x.split('=',1)[1]) for x in raw.splitlines() if x.startswith('SUBMISSION=')]
    if len(ss)!=3 or {s['repeat_index'] for s in ss}!={1,2,3} or len({s['job_id'] for s in ss})!=3 or any(s['group']!='budget-5' or s['outer']!=150 for s in ss):raise ValueError('Incomplete/wrong/duplicate submissions; inspect intents, no replay')
    return ss

def run(file):
    c=json.loads(Path(file).read_text());p=Path(c['local']);run=Path(c['run']);old=json.loads((p/'pj-ablation-dispatch.latest.json').read_text())
    validate_retirement(old,json.loads(Path(c['retirement']).read_text()))
    lock=open(p/'pj-ablation-dispatch-20261018T1103/lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in c['helper_hashes'].items():
        if digest(p/n)!=h:raise ValueError('Frozen helper changed: '+n)
    if digest(__file__)!=c['dispatcher_sha256']:raise ValueError('Dispatcher source changed')
    run.mkdir(exist_ok=False)
    L=c['lumia'];B=c['pj'];C=c['worker_source'];LS=c['lumia_ssh'];PS=c['pj_ssh'];LP=c['lumia_python'];PP=c['pj_python'];SDK=c['pj_sdk']
    s=dict(at=time.time(),pid=os.getpid(),source=c['source'],worker_source=C,group='budget-5',prior_dispatcher=old['pid'],phase='WAITING_FINAL150',deadline=time.time()+16*3600,no_auto_retry=True)
    def save():
        s['at']=time.time();f=Path(c['state']);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(f)
    def helper(n):
        if digest(p/n)!=c['helper_hashes'][n]:raise ValueError('Frozen helper changed')
        return (p/n).read_text()
    save()
    try:
        while time.time()<s['deadline']:
            code='from pathlib import Path\nimport json\nb=Path('+repr(L)+');p=b/"runs/budget-5/checkpoints/training_state/global_step_150/manifest.json"\nprint(json.dumps({"ready":p.exists() and (b/"runs/budget-5/checkpoints/actor/global_step_150/model.safetensors.index.json").exists()}))'
            if json.loads(remote(LS,'python3',code,L+'/sources/'+C,45))['ready']:break
            save();time.sleep(60)
        else:raise TimeoutError('Bounded wait expired')
        s['phase']='EXPORTING_ON_LUMIA';save()
        raw=remote(LS,LP,'BASE='+repr(L)+'\nGROUP="budget-5"\nCOMMIT='+repr(C)+'\n'+helper(c['export_helper']),L+'/sources/'+C,1200)
        (run/'export.output').write_text(raw);proof=json.loads(raw.strip().splitlines()[-1]);write_new(run/'export.json',proof)
        cid=proof['checkpoint_identity'];dest=B+'/receipts/ablation-import-budget-5-150-20261018.tar'
        s.update(checkpoint_identity=cid,archive=dest,archive_bytes=proof['archive_bytes'],archive_sha256=proof['archive_sha256'],phase='CHECKING_ZERO_SUBMISSION');save()
        raw=remote(PS,SDK,'CID='+repr(cid)+'\nB='+repr(B)+'\nDEST='+repr(dest)+'\n'+helper(c['zero_helper']),B+'/sources/'+C,90);(run/'zero-submission.output').write_text(raw)
        s.update(phase='STREAMING_NO_LOCAL_ARCHIVE',transfer_started=time.time());save()
        alarm='import signal,sys,json\ndef expired(*a):raise TimeoutError("Transport deadline; preserve partial")\nsignal.signal(signal.SIGALRM,expired)\nsignal.alarm(7200)\n'
        send=alarm+'from reproduction.seven_dataset.resume_stream import inspect_prefix,send_suffix\np='+repr(proof['archive'])+'\ne=inspect_prefix(p,0)\nassert e["size"]=='+str(proof['archive_bytes'])+'\nsend_suffix(p,e,sys.stdout.buffer)'
        receive=alarm+'from reproduction.seven_dataset.resume_stream import inspect_prefix,receive_suffix\np='+repr(dest)+'\nopen(p,"xb").close()\ne=inspect_prefix(p,0)\nprint(json.dumps(receive_suffix(p,e,'+str(proof['archive_bytes'])+','+repr(proof['archive_sha256'])+',sys.stdin.buffer)))'
        def command(ssh,py,code,base):return ssh+['PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(base+'/sources/'+C)+' '+shlex.quote(py)+' -c '+shlex.quote(code)]
        s['transfer']=transfer(command(LS,LP,send,L),command(PS,PP,receive,B),7260);s.update(phase='PJ_FULL_INPUT_PREFLIGHT',transfer_finished=time.time());save()
        raw=remote(PS,PP,'PROOF='+repr(proof)+'\nARCHIVE='+repr(dest)+'\nCOMMIT='+repr(C)+'\n'+helper(c['stage_helper']),B+'/sources/'+C,1200)
        (run/'preflight.output').write_text(raw);pre=json.loads(raw.strip().splitlines()[-1]);write_new(run/'preflight.json',pre);s.update(phase='SUBMITTING_UNIQUE_THREE',preflight=str(run/'preflight.json'));save()
        raw=remote(PS,SDK,'PREFLIGHT='+repr(pre)+'\n'+helper(c['submit_helper']),B+'/sources/'+C,300);(run/'submission.output').write_text(raw);subs=parse_submissions(raw)
        for x in subs:write_new(p/('seven-ablation-pj-budget-5-150-r'+str(x['repeat_index'])+'-submission.json'),x)
        s.update(phase='THREE_SUBMITTED_NO_REPLAY',submissions=subs);save()
    except Exception as e:s.update(phase='FAILED_PRESERVE_EVIDENCE_NO_AUTO_RETRY',error=str(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':run(sys.argv[1])
