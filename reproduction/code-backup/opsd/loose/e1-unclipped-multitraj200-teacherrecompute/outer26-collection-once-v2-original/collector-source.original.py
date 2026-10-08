"""Fixed same-node archive reader: stdout gzip/tar; no remote file writes."""
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tarfile

SOURCE='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG='0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
COMMON='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
RUN=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304')
REPO=RUN.parent.parent/'repo-fcecbea5d939'
OUTER=26
DIR=RUN/'trajectories/outer_000026'
PAIR_SHA='6326d57977268284050ae5c3d9591e80c6204850a0fe17abd53ad1b61ec6c0f8'
PROGRESS_SHA='8d3a143091427c45924848bb6d327010cbaab0dac3f8f97bbeb29a9c82bf838b'
AUTHORITY_SHA='b64e39864be651423dee39b423191a3d87a6c401d496356330c37d186e6b0fd9'
SOURCE_BINDING='9a420576b556c1194d81e22886beb2734fd19b79a561faa5e791b2e0e51c64c8'
DRIVER=(2548319,1834784786)
WORKERS=[(2548974,1834785562),(2549227,1834786108),(2549228,1834786110),
         (2549229,1834786111),(2549231,1834786113),(2549232,1834786114),
         (2549230,1834786112),(2549233,1834786115)]
NAMES=['questions.json','student.jsonl','student-manifest.json','paired-manifest.json']+[
    f'teacher-rank-{i}{suffix}' for i in range(8) for suffix in ('.jsonl','.manifest.json')]
MAX_RAW=1024**3
ATTRS=('st_dev','st_ino','st_uid','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def sig(value):return tuple(getattr(value,key) for key in ATTRS)


def pairs(values):
    result={}
    for key,value in values:
        need(key not in result,'Duplicate JSON key');result[key]=value
    return result


def parse(raw):
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def opened(path):
    path=Path(path);need(path.is_absolute() and path.resolve(strict=True)==path,'Nonphysical path')
    before=path.stat();need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()==0
        and before.st_nlink==1 and before.st_size<=MAX_RAW,'Member owner/type/size differs')
    handle=os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb')
    need(sig(os.fstat(handle.fileno()))==sig(before),'Open identity race')
    return before,handle


def finish(path,before,handle,pin,size):
    need(sig(os.fstat(handle.fileno()))==sig(before)==sig(Path(path).stat())
         and Path(path).resolve(strict=True)==Path(path) and size==before.st_size,'Member changed during read')
    return dict(origin_path=str(path),sha256=pin,bytes=size,device=before.st_dev,inode=before.st_ino,
                uid=before.st_uid,mode=stat.S_IMODE(before.st_mode),mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns)


def same_stat(record,value):
    return all(record[key]==expected for key,expected in dict(device=value.st_dev,inode=value.st_ino,uid=value.st_uid,
        mode=stat.S_IMODE(value.st_mode),bytes=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns).items())


def read_small(path,pin=None):
    before,handle=opened(path);need(before.st_size<16*1024**2,'Small original exceeds limit')
    with handle:
        raw=handle.read();record=finish(path,before,handle,sha(raw),len(raw))
    if pin is not None:need(record['sha256']==pin,'Pinned small original differs')
    return raw,record


def hash_member(path,expected):
    before,handle=opened(path);need(stat.S_IMODE(before.st_mode)==0o444,'Published member mode differs')
    digest=hashlib.sha256();size=0;lines=0;tail=b''
    with handle:
        while True:
            block=handle.read(8*1024**2)
            if not block:break
            digest.update(block);size+=len(block);lines+=block.count(b'\n');tail=block[-1:]
        record=finish(path,before,handle,digest.hexdigest(),size)
    need(record['sha256']==expected['sha256'] and size==expected['bytes'],'Full member digest/bytes differs')
    name=Path(path).name
    if name.endswith('.jsonl'):
        need(tail==b'\n' and lines==(2560 if name=='student.jsonl' else 64),'JSONL EOF/row count differs')
    else:parse(read_small(path,record['sha256'])[0])
    return record


def process_identity(pid,start):
    proc=Path('/proc')/str(pid);status=proc.joinpath('status').read_text();cgroup=proc.joinpath('cgroup').read_text();rawstat=proc.joinpath('stat').read_text()
    need(int(rawstat.rsplit(')',1)[1].split()[19])==start,'Original PID was reused')
    uid=re.search(r'^Uid:\s+(.*)$',status,re.M);need(uid is not None and uid.group(1).split()==['0']*4,'Original process UID differs')
    need(any(re.search(r'/job[_-]3304(?:\.scope)?/',line.split(':',2)[-1]) for line in cgroup.splitlines()),'Original process job differs')
    return dict(pid=pid,start_ticks=start,status=status,cgroup=cgroup,stat=rawstat)


def commands():
    result={}
    for key,args in [('head',['rev-parse','HEAD']),('clean',['status','--porcelain'])]:
        argv=['git','--no-optional-locks','-C',str(REPO)]+args
        value=subprocess.run(argv,capture_output=True,text=True,timeout=25)
        result[key]=dict(argv=argv,exit_code=value.returncode,stdout=value.stdout,stderr=value.stderr)
        need(value.returncode==0 and not value.stderr,'Source Git read failed')
    need(result['head']['stdout'].strip()==SOURCE and result['clean']['stdout']=='','Frozen source differs')
    return result


def context():
    need(socket.gethostname()=='h100-10' and os.getuid()==0,'Only own H100 reader is allowed')
    source_raw,source_record=read_small(RUN/'bindings/source.json',SOURCE_BINDING);source=parse(source_raw)
    need(source['repo']==str(REPO) and source['source_commit']==SOURCE and source['clean'] is True and len(source['source_files'])==19,'Frozen source binding differs')
    source_records={}
    for relative,pin in source['source_files'].items():
        need(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Source relative path differs')
        _,source_records[relative]=read_small(REPO/relative,pin)
    _,config_record=read_small(RUN/'full_state/global_step_5/resolved-config.json',CONFIG)
    return dict(driver=process_identity(*DRIVER),workers=[process_identity(*pair) for pair in WORKERS],
        git=commands(),source_binding=source_record,source_records=source_records,resolved_config=config_record)


def assert_no_own_open_writer(records):
    identities={(r['device'],r['inode']) for r in records.values()}
    for pid,_ in [DRIVER]+WORKERS:
        for fd in (Path('/proc')/str(pid)/'fd').iterdir():
            try:
                st=fd.stat();flags=(Path('/proc')/str(pid)/'fdinfo'/fd.name).read_text()
            except FileNotFoundError:continue
            flag=re.search(r'^flags:\s+([0-7]+)$',flags,re.M)
            need(flag is not None,'FD flags unreadable')
            if (st.st_dev,st.st_ino) in identities:
                need(int(flag.group(1),8)&os.O_ACCMODE==os.O_RDONLY,'Original writer still holds archive writable FD')


def expected_members(pair,student):
    expected={'paired-manifest.json':dict(path=str(DIR/'paired-manifest.json'),sha256=PAIR_SHA,bytes=5638),
              'questions.json':pair['request'],'student-manifest.json':pair['student_manifest'],'student.jsonl':student['student']}
    need(type(pair['teacher_rank_members']) is list and len(pair['teacher_rank_members'])==8,'Teacher rank members differ')
    for rank,item in enumerate(pair['teacher_rank_members']):
        need(type(item['rank']) is int and item['rank']==rank,'Teacher rank order differs')
        expected[f'teacher-rank-{rank}.jsonl']=item['teacher'];expected[f'teacher-rank-{rank}.manifest.json']=item['manifest']
    need(set(expected)==set(NAMES),'Exact20 roles differ')
    for name,record in expected.items():
        need(set(record)=={'path','sha256','bytes'} and record['path']==str(DIR/name)
             and type(record['bytes']) is int and 0<record['bytes']<=MAX_RAW
             and re.fullmatch('[0-9a-f]{64}',record['sha256']),'Archive path/size/SHA differs')
    need(sum(r['bytes'] for r in expected.values())<=MAX_RAW,'Raw20 exceed1GiB')
    return expected


def identity_ok(pair):
    identity=pair['identity']
    need(identity['source_commit']==SOURCE and identity['config_sha256']==CONFIG and identity['common_manifest_sha256']==COMMON
         and identity['job_id']=='3304' and identity['driver_pid']==DRIVER[0] and type(identity['outer_step']) is int
         and identity['outer_step']==26 and identity['directory']==str(DIR) and identity['profile']=='multitraj200'
         and pair['student_count']==2560 and pair['teacher_count']==512
         and pair['teacher_diagnostic_answers_used_as_targets'] is False,'Paired observed26 identity differs')


class HashingReader:
    def __init__(self,handle):self.handle=handle;self.digest=hashlib.sha256();self.size=0
    def read(self,size):
        block=self.handle.read(size);self.digest.update(block);self.size+=len(block);return block


def collect(output,authority):
    need(sha(canonical(authority))==AUTHORITY_SHA,'ActualV11 full authority differs')
    before=context();raw,_=read_small(DIR/'paired-manifest.json',PAIR_SHA);pair=parse(raw);identity_ok(pair)
    student_raw,_=read_small(DIR/'student-manifest.json',pair['student_manifest']['sha256']);student=parse(student_raw)
    need(student['identity']==pair['identity'] and student['request']==pair['request'],'Student metadata identity differs')
    expected=expected_members(pair,student);records={name:hash_member(DIR/name,expected[name]) for name in NAMES}
    need(len({(r['device'],r['inode']) for r in records.values()})==20,'Archive inode aliases')
    assert_no_own_open_writer(records)
    with gzip.GzipFile(fileobj=output,mode='wb',mtime=0,compresslevel=3) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
            for name in NAMES:
                path=DIR/name;st,handle=opened(path);need(same_stat(records[name],st),'Copy physical identity differs')
                with handle:
                    wrapped=HashingReader(handle);header=tarfile.TarInfo(name);header.size=st.st_size;header.mode=0o444
                    archive.addfile(header,wrapped);second=finish(path,st,handle,wrapped.digest.hexdigest(),wrapped.size)
                need(second==records[name],'Copied bytes/physical member differs')
            after=context();assert_no_own_open_writer(records)
            need(before['git']==after['git'] and before['source_records']==after['source_records']
                 and before['source_binding']==after['source_binding'] and before['resolved_config']==after['resolved_config'],'Source/config changed across copy')
            need(before['driver']['cgroup']==after['driver']['cgroup']
                 and [p['cgroup'] for p in before['workers']]==[p['cgroup'] for p in after['workers']], 'Original process allocation changed')
            inventory=dict(version=1,kind='OBSERVED_COMPLETED_OUTER26_ARCHIVE_INVENTORY_NOT_TRAINING_CHECKPOINT_MANIFEST',
                scope='ORIGINAL20_BYTE_COPY_WITH_STABLE_TWO_READS_NOT_FUTURE_WRITER_LIFETIME_OR_EFFECT_ACCEPTANCE',
                identity=pair['identity'],authority=authority,members=records,raw_bytes=sum(r['bytes'] for r in records.values()),
                original_context_before=before,original_context_after=after,
                collection=dict(pid=os.getpid(),uid=os.getuid(),hostname=socket.gethostname(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    no_gpu_calls=True,no_remote_file_writes=True,original_training_job_id='3304'))
            blob=canonical(inventory)+b'\n';header=tarfile.TarInfo('observed-inventory.json');header.size=len(blob);header.mode=0o444
            archive.addfile(header,io.BytesIO(blob))
    return inventory


if __name__=='__main__':
    authority=parse(sys.stdin.buffer.read());collect(sys.stdout.buffer,authority)
