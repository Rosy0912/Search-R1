"""Explicit held-owner terminal handshake; never interprets an UNKNOWN SM window."""
import hashlib,json,os,time
from pathlib import Path


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path,value):
    data=json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n'
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        view=memoryview(data)
        while view:
            written=os.write(fd,view);require(written>0,'Terminal short write');view=view[written:]
        os.fsync(fd)
    finally:os.close(fd)
    directory=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def process_identity(pid):
    root=Path('/proc')/str(pid)
    stat=(root/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
    return dict(pid=int(pid),starttime_ticks=int(tail[19]),uid=root.stat().st_uid,
                cgroup=(root/'cgroup').read_text().splitlines())


def facts(folder,child_pid,parent_pid,job_id):
    folder=Path(folder)
    child_identity=process_identity(child_pid);parent_identity=process_identity(parent_pid)
    require(child_identity['uid']==parent_identity['uid']==os.getuid()
            and child_identity['cgroup']==parent_identity['cgroup']
            and any('/job_'+str(job_id)+'/' in line for line in child_identity['cgroup']),
            'Terminal must bind actual same-job parent/child identities')
    result=json.loads((folder/'result.json').read_text())
    rows=[json.loads(line) for line in (folder/'qa.jsonl').read_text().splitlines()]
    require(result['status']=='COMPLETE64_UNREVIEWED_NO_TRAINING' and len(rows)==64
            and result['quality']['denominator']==64 and result['metrics']['val/qa_count/all']==64,
            'Terminal requires all64 original results')
    require(result['qa_sha256']==sha(folder/'qa.jsonl') and result['final_sha256']==sha(folder/'final.safetensors'),
            'Terminal raw result binding drift')
    require(isinstance(result['producer_epoch'],(float,int)) and 0<result['producer_epoch']<=time.time(),
            'Actual result producer epoch absent/future')
    return dict(scope='HELD_GPU_OWNER_TERMINAL_ONLY_NOT_QUALITY_APPROVAL',child_pid=int(child_pid),
                parent_pid=int(parent_pid),job_id=str(job_id),child_identity=child_identity,parent_identity=parent_identity,result_sha256=sha(folder/'result.json'),
                qa_sha256=sha(folder/'qa.jsonl'),final_sha256=sha(folder/'final.safetensors'),
                producer_epoch=result['producer_epoch'])


def hold_owner(folder,engine,timeout_seconds=60):
    # engine is deliberately referenced until parent has drained active sampling.
    require(engine is not None,'Actual engine must remain strongly referenced')
    parent=int(os.environ['SEARCH_R1_EVAL64_SUPERVISOR_PID']);job=os.environ['SLURM_JOB_ID']
    require(parent==os.getppid(),'Terminal parent identity differs')
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix in ('.json','.jsonl','.safetensors'):
            with path.open('rb') as raw:os.fsync(raw.fileno())
    receipt=facts(folder,os.getpid(),parent,job)
    nonce=os.environ['SEARCH_R1_EVAL64_TERMINAL_NONCE']
    durable(Path(folder)/'terminal-held.json',dict(receipt,held_epoch=time.time()))
    deadline=time.monotonic()+timeout_seconds;permit=Path(folder)/'terminal-authorized.json'
    while not permit.exists():
        require(os.getppid()==parent,'Terminal parent exited before authorization')
        require(time.monotonic()<deadline,'Terminal authorization timeout')
        time.sleep(.05)
    authorization=json.loads(permit.read_text())
    require(authorization['receipt']==receipt and authorization['nonce']==nonce
            and authorization['monitor_stopped'] is True and authorization['sampler_errors']==[],
            'Invalid terminal authorization')
    durable(Path(folder)/'terminal-child-ack.json',dict(receipt,ack_epoch=time.time(),
            authorized_sha256=sha(permit)))
    return receipt


def authorize(folder,child_pid,parent_pid,job_id,nonce,stop_monitor,sampler_errors):
    expected=facts(folder,child_pid,parent_pid,job_id)
    held=json.loads((Path(folder)/'terminal-held.json').read_text())
    require({k:held[k] for k in expected}==expected and expected['producer_epoch']<=held['held_epoch']<=time.time(),
            'Held terminal process/result identity drift')
    stopped=stop_monitor()  # Must join actual in-flight monitor before inspecting errors.
    require(stopped['stopped'] is True and not sampler_errors,'Terminal monitor not drained or failed')
    durable(Path(folder)/'terminal-authorized.json',dict(receipt=expected,nonce=nonce,
            monitor_stopped=True,sampler_errors=list(sampler_errors),authorized_epoch=time.time()))
    return dict(receipt=expected,monitor_stop=stopped,authorization_sha256=sha(Path(folder)/'terminal-authorized.json'))


def verify_ack(folder,authorization):
    permit=Path(folder)/'terminal-authorized.json'
    ack=json.loads((Path(folder)/'terminal-child-ack.json').read_text())
    require({k:ack[k] for k in authorization['receipt']}==authorization['receipt']
            and ack['authorized_sha256']==sha(permit)==authorization['authorization_sha256']
            and ack['ack_epoch']>=json.loads(permit.read_text())['authorized_epoch'],
            'Terminal acknowledgment process/raw binding drift')
    return dict(path=str(Path(folder)/'terminal-child-ack.json'),sha256=sha(Path(folder)/'terminal-child-ack.json'))
