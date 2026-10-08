"""Fixed read-only postterminal source/binding originals; no native/model/GPU acceptance."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import types

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
RUN = ROOT / 'runs/formal-3304'
REPO = ROOT / 'repo-fcecbea5d939'
TOOLS = ROOT / 'tools/formal-71d2501d4512'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
COMMON_PATH = ROOT / 'tools/final200-f11580b6a7bc/common.py'
LOGS_PATH = ROOT / 'tools/final200-f11580b6a7bc/consume_final200_logs_terminal.py'
COMMON_SHA = 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
LOGS_SHA = '509df9332effd1196c6cb07a6a9d67612d9efdd75498f39a2bdb36726dcd7306'
FORMAL_SHA = '71d2501d45120c546d5a94cc9add1a60fb7afbf70dd4ed7a5333e51955ab1028'
SUPERVISOR_SHA = 'ffb3bdc64f91c4a60905cd15994132c63c6494952c4ca0818c0cf7eee119d017'
OWNED_SHA = 'f69418c892f3d0a644e06003773bbfdfeafe9355a48dcc013f94c54af65ab1fd'
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
RECEIPT = 'd1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
OUTPUT = RUN / 'evidence/postterminal-source-bindings-once-v1'
REQUEST_KEYS = {'version','kind','source_commit','config_sha256','stored_training_job_id','accepted5_receipt_sha256','ready','terminal'}
TERMINAL_ROLES = {'training_child_exit','wrapper_exit','scheduler_original'}
SMALL_SUFFIXES = {'.py','.json','.sh','.yaml','.yml'}


def require(ok, message):
    if not ok: raise ValueError(message)


def bootstrap_common():
    # Verify and compile the same owned physical FD bytes; never consult pyc.
    p=COMMON_PATH
    require(p.is_absolute() and p.resolve(strict=True)==p and not p.is_symlink() and p.is_file(),'Common physical path differs')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size<=1024**2,'Common regular FD/UID/size differs')
        raw=b''
        while True:
            chunk=os.read(fd,65536)
            if not chunk:break
            raw+=chunk
        after=os.fstat(fd)
    finally:os.close(fd)
    keys=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    require(tuple(getattr(before,k) for k in keys)==tuple(getattr(after,k) for k in keys)==tuple(getattr(p.stat(),k) for k in keys),'Common changed during read')
    require(hashlib.sha256(raw).hexdigest()==COMMON_SHA,'Common exact source differs')
    c=types.ModuleType('postterminal_source_exact_common');c.__file__=str(p);c.__package__=''
    exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),c.__dict__)
    return c


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_bytes(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    fsync_directory(Path(path).parent)


def emit(path, value):
    write_bytes(path,json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')


def reserve(path):
    p=Path(path)
    require(p.is_absolute() and p.parent.resolve(strict=True)==p.parent and p.parent.stat().st_uid==os.getuid(),'Output physical own parent differs')
    p.mkdir()  # Existing/unknown attempts must never be retried or overwritten.
    fsync_directory(p.parent);fsync_directory(p)
    return p


def validate_request(value,c):
    require(type(value) is dict and set(value)==REQUEST_KEYS,'Postterminal request schema differs')
    expected=dict(version=1,kind='POSTTERMINAL_SOURCE_BINDING_READONLY_REQUEST',source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id='3304',accepted5_receipt_sha256=RECEIPT)
    for key,target in expected.items():c.typed(value.get(key),target,'Request identity differs: '+key)
    require(type(value['terminal']) is dict and set(value['terminal'])==TERMINAL_ROLES,'Three actual terminal originals required')
    c.typed(value['ready'].get('physical_path'),str(RUN/'approval-ready.json'),'Ready original path differs')
    for role,path in [('training_child_exit',RUN/'evidence/train-process-exit.json'),('wrapper_exit',RUN/'job-exit.txt')]:
        c.typed(value['terminal'][role].get('physical_path'),str(path),'Literal terminal source path differs: '+role)


def copy_record(c,record,destination):
    raw,current=c.read_file(record['physical_path'],record['sha256'],keep_bytes=True,limit=2*1024**2)
    require(current==record,'Physical original changed before copy')
    write_bytes(destination,raw);c.verify_record(record)
    return dict(original=record,copy=c.read_file(destination,record['sha256'])[1])


def terminal_first(c,logs,request,out,started_ns):
    raws={};records={};copies={}
    for role in sorted(TERMINAL_ROLES):
        rec=request['terminal'][role];c.verify_record(rec,RUN)
        require(rec['mtime_ns']<=started_ns and rec['ctime_ns']<=started_ns,'Terminal original was created/changed after this inspection began')
        raw,current=c.read_file(rec['physical_path'],rec['sha256'],keep_bytes=True,limit=2*1024**2)
        require(current==rec,'Terminal identity changed');raws[role]=raw;records[rec['physical_path']]=rec
        write_bytes(out/(role+'.original'),raw);copies[role]=c.read_file(out/(role+'.original'),rec['sha256'])[1]
    layers=logs.terminal_layers(raws,c)
    require(layers['training_child_exit']['status']=='OBSERVED_LITERAL_EXIT' and layers['wrapper_exit']['status']=='OBSERVED_LITERAL_EXIT' and layers['scheduler']['status']=='OBSERVED_LITERAL_SCHEDULER_TERMINAL','Actual terminal prerequisite is blocked')
    emit(out/'terminal-inputs.original.json',dict(records=records,copies=copies,layers=layers,scope='Literal supplied terminal originals only; freshness/writer/release require separate independent qualification'))
    return layers,records


@contextlib.contextmanager
def validation_environment():
    values={'E1_FORMAL_MANIFEST_SHA256':FORMAL_SHA,'E1_ENTRY_SCRIPT':str(TOOLS/'formal_job.sh'),'SEARCH_R1_REPO':str(REPO)}
    previous={k:os.environ.get(k) for k in values};os.environ.update(values)
    try:yield values
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def load_formal(c):
    require('owned_runtime' not in sys.modules,'Refuse a preloaded owned_runtime module')
    owned,orec=c.exact_module(TOOLS/'owned_runtime.py',OWNED_SHA,'owned_runtime')
    sys.modules['owned_runtime']=owned
    formal,frec=c.exact_module(TOOLS/'formal_supervisor.py',SUPERVISOR_SHA,'postterminal_exact_formal')
    require(formal.__file__==str(TOOLS/'formal_supervisor.py') and formal.SOURCE==SOURCE and formal.ROOT==ROOT,'Frozen formal physical source differs')
    return formal,{orec['physical_path']:orec,frec['physical_path']:frec}


def git_original(c,out,phase,verb):
    argv=['git','-C',str(REPO)]+({'head':['rev-parse','HEAD'],'status':['status','--porcelain']}[verb])
    stem=out/(phase+'-git-'+verb)
    emit(str(stem)+'.argv.original.json',dict(argv=argv,scope='Additional independently waited Git capture; frozen helper command behavior is unchanged'))
    result=None;error=None
    with open(str(stem)+'.stdout.original','xb') as stdout,open(str(stem)+'.stderr.original','xb') as stderr:
        try:result=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=20,check=False)
        except BaseException as exc:error=exc
        finally:
            stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
            emit(str(stem)+'.wait.original.json',dict(waited=result is not None,returncode=None if result is None else result.returncode,error=None if error is None else dict(type=type(error).__name__,message=str(error))))
    if error is not None:raise error
    require(type(result.returncode) is int and result.returncode==0,'Actual Git child did not succeed')
    raw=Path(str(stem)+'.stdout.original').read_bytes()
    require(raw.strip()==(SOURCE.encode() if verb=='head' else b''),'Additional Git HEAD/status differs')
    captures={role:c.read_file(str(stem)+suffix)[1] for role,suffix in [('argv','.argv.original.json'),('stdout','.stdout.original'),('stderr','.stderr.original'),('wait','.wait.original.json')]}
    return dict(argv=argv,returncode=result.returncode,waited=True,originals=captures)


def inspect_sources(c,request,out,layers,terminal_records,own_record,common_record):
    records=dict(terminal_records);records[own_record['physical_path']]=own_record;records[common_record['physical_path']]=common_record
    ready,ready_record=c.read_json(request['ready']['physical_path'],request['ready']['sha256'],RUN)
    require(ready_record==request['ready'],'Ready physical identity differs');records[ready_record['physical_path']]=ready_record
    for key,target in dict(source_commit=SOURCE,job_id='3304',uid=os.getuid(),run_root=str(RUN)).items():c.typed(ready.get(key),target,'Ready original differs: '+key)
    formal,formal_records=load_formal(c);records.update(formal_records)
    git=[]
    for verb in ('head','status'):git.append(git_original(c,out,'before',verb))
    before=c.accepted_context(RECEIPT);records.update(before['records'])
    c.typed(ready['binding_sha256'],before['manifest']['files']['launch_binding']['sha256'],'Original ready/signed launch binding differs')
    with validation_environment() as validation_env:
        repo,manifest=formal.sources(TOOLS);binding=formal.verify_binding(RUN,ready)
    require(repo==REPO and manifest['source_commit']==SOURCE and binding==c.read_json(before['manifest']['files']['launch_binding']['path'],before['manifest']['files']['launch_binding']['sha256'])[0],'Frozen function/context results differ')
    require(len(before['closure']['modules'])==211,'Physical211 inventory differs')
    copies={};not_copied=[];total=0
    small=out/'small-originals';small.mkdir();fsync_directory(out);fsync_directory(small)
    for path,rec in sorted(records.items()):
        if Path(path).suffix in SMALL_SUFFIXES and rec['bytes']<=2*1024**2:
            total+=rec['bytes'];require(total<=32*1024**2,'Small raw source/binding copy budget exceeded')
            name=hashlib.sha256(path.encode()).hexdigest()+Path(path).suffix
            copies[path]=copy_record(c,rec,small/name)
        else:not_copied.append(rec)
    after=c.accepted_context(RECEIPT)
    require(before['records']==after['records'],'Current source/context physical fingerprints changed')
    for verb in ('head','status'):git.append(git_original(c,out,'after',verb))
    for capture in git:
        for rec in capture['originals'].values():c.verify_record(rec)
    c.recheck(records)
    return dict(version=1,kind='POSTTERMINAL_SOURCE_AND_BINDING_ORIGINALS_NOT_NATIVE_OR_TERMINAL_ACCEPTANCE',status='CURRENT_SOURCE_BINDINGS_CHECKED_INDEPENDENT_QUALIFICATION_REQUIRED',source_commit=SOURCE,config_sha256=CONFIG,stored_training_job_id='3304',terminal_layers=layers,accepted5=before['accepted5'],current_source_records=records,physical_import_inventory_members=211,frozen_function_calls=['sources(tools)','verify_binding(run,ready)'],frozen_formal_manifest_sha256=FORMAL_SHA,adapter_constructed_validation_environment=validation_env,git_captures=git,small_original_copies=copies,raw_not_copied=not_copied,small_copied_bytes=total,scope='Current full SHA/stat of the existing211 physical inventory, source/tool bytes, config and launch/official/data/retrieval manifest bytes; no current native imports, model/index asset content reload, training200/effect/writer-lifetime/resource-release qualification',actual_outer_updates=None,native_model_load=None,resource_release=None,whole_resource_bad66='RETAINED_UNCHANGED_NOT_CONSUMED_OR_RECLASSIFIED')


def collect(c,request,request_record,own_record,common_record,identity,out,started_ns):
    out=reserve(out)
    emit(out/'intent.original.json',dict(status='ONCE_RESERVED_NO_RETRY',request=request_record,collector_source=own_record,common_source=common_record,execution=identity,started_ns=started_ns,argv=sys.argv))
    try:
        copy_record(c,request_record,out/'request.original.json')
        logs,logs_record=c.exact_module(LOGS_PATH,LOGS_SHA,'postterminal_exact_logs')
        layers,terminals=terminal_first(c,logs,request,out,started_ns)
        require('torch' not in sys.modules,'Adapter must not import Torch or initialize CUDA')
        report=inspect_sources(c,request,out,layers,terminals,own_record,common_record)
        report.update(execution=identity,collector_source=own_record,common_source=common_record,terminal_parser_source=logs_record,request=request_record)
        c.verify_record(logs_record);c.verify_record(request_record);c.verify_record(own_record);c.verify_record(common_record)
        require('torch' not in sys.modules,'Adapter must not import Torch or initialize CUDA')
        emit(out/'source-bindings.original.json',report)
        emit(out/'controller-return.original.json',dict(returncode=0,scope='Adapter result only; not scheduler, training, native200 or effect completion'))
        return report
    except BaseException as exc:
        emit(out/'failure.original.json',dict(type=type(exc).__name__,message=str(exc),scope='Once remains reserved; partial raw originals remain; no automatic retry or qualification'))
        raise


def main():
    started_ns=time.time_ns();p=argparse.ArgumentParser()
    for arg in ('request','request-sha256','self-sha256','execution-job-id','execution-step-id','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();c=bootstrap_common()
    require(sys.flags.optimize==0 and sys.version_info[:2]==(3,10) and sys.executable==PYTHON,'Use the fixed unoptimized native Python3.10 interpreter')
    own_record=c.read_file(Path(__file__),a.self_sha256)[1];common_record=c.read_file(COMMON_PATH,COMMON_SHA)[1]
    identity=c.execution_identity(a.execution_job_id,a.execution_step_id)
    require(identity['uid']==0 and identity['execution_job_id']!='3304','Requires a distinct actual own CPU execution job after stored3304 terminal')
    require(Path(a.output)==OUTPUT,'Only fixed exclusive output is allowed')
    request,request_record=c.read_json(a.request,a.request_sha256,RUN);validate_request(request,c)
    report=collect(c,request,request_record,own_record,common_record,identity,Path(a.output),started_ns)
    require('torch' not in sys.modules,'Adapter must not import Torch or initialize CUDA')
    print(json.dumps(dict(kind=report['kind'],status=report['status'],output=str(OUTPUT),scope=report['scope']),sort_keys=True),flush=True)


if __name__=='__main__':main()
