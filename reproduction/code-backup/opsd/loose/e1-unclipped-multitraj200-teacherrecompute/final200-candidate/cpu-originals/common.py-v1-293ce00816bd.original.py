"""Owned physical CPU evidence helpers; no network, GPU, restore or signing."""
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import types

SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
MODEL = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
COMMON_DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CLOSURE = 'e4fb0f6c4b2dcf62837e2276ba58e29f5e198a1a0252975b7578debb6209cc4c'
FORMAL = '71d2501d45120c546d5a94cc9add1a60fb7afbf70dd4ed7a5333e51955ab1028'
ACCEPTED5 = '30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
PUBLIC = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
MATH_HELPER = '77afa00b2b3e0fdcf674f5022f8b6cc39e12818bc7e825f8e58128fecab0f68b'
HF_HELPER = 'b2e02be33d04d925af04a6a8f4efea4ba93997146197d54729d0a9715754c419'
FULL_STATE = 'b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6'
GATE = '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
RUN = ROOT / 'runs/formal-3304'
REPO = ROOT / 'repo-fcecbea5d939'
NUMEL = 3085938688
KIND = 'OBSERVED_MULTITRAJ200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST'
RECORD_KEYS = {'physical_path','sha256','bytes','device','inode','uid','mtime_ns','ctime_ns'}
HF_NAMES = {'added_tokens.json','config.json','generation_config.json','merges.txt',
            'model.safetensors.index.json','special_tokens_map.json','tokenizer.json',
            'tokenizer_config.json','vocab.json'} | {f'model-{i:05d}-of-00003.safetensors' for i in (1,2,3)}
ARCHIVE_NAMES = {'questions.json','student.jsonl','student-manifest.json','paired-manifest.json'} | {
    f'teacher-rank-{rank}'+suffix for rank in range(8) for suffix in ('.jsonl','.manifest.json')}
STAT_FIELDS = ('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')


def require(ok, message):
    if not ok: raise ValueError(message)


def typed(value, target, message):
    require(type(value) is type(target) and value == target, message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def number(raw):
        value = float(raw)
        require(math.isfinite(value), 'Nonfinite JSON float')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False).encode()


def valid_sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value), 'Exact SHA256 required')
    return value


def physical(path, root=None):
    p = Path(path)
    require(p.is_absolute() and p.resolve(strict=True) == p, 'Physical absolute path required')
    require(p.is_file() and not p.is_symlink() and p.stat().st_uid == os.getuid(), 'Owned regular file required')
    if root is not None: require(p.is_relative_to(Path(root)), 'File outside run root')
    return p


def identity(value):
    return tuple(getattr(value, key) for key in STAT_FIELDS)


def read_file(path, expected_sha=None, root=None, keep_bytes=False, limit=None):
    """Hash the same opened bytes that are parsed/compiled; reject replace races."""
    p = physical(path,root); before=p.stat(); digest=hashlib.sha256(); parts=[]
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and identity(before)==identity(opened), 'Opened FD identity changed')
        if limit is not None: require(opened.st_size<=limit, 'Evidence exceeds size limit')
        size=0
        while True:
            chunk=os.read(fd,8*1024**2)
            if not chunk:break
            digest.update(chunk);size+=len(chunk)
            if keep_bytes:parts.append(chunk)
        finished=os.fstat(fd)
    finally:os.close(fd)
    after=p.stat()
    require(identity(before)==identity(finished)==identity(after) and size==opened.st_size, 'Input changed during stable full read')
    actual=digest.hexdigest()
    if expected_sha is not None:require(actual==valid_sha(expected_sha), 'Pinned bytes differ: '+str(p))
    record=dict(physical_path=str(p),sha256=actual,bytes=size,device=opened.st_dev,inode=opened.st_ino,
                uid=opened.st_uid,mtime_ns=opened.st_mtime_ns,ctime_ns=opened.st_ctime_ns)
    return (b''.join(parts) if keep_bytes else None),record


def read_json(path, expected_sha=None, root=None):
    raw,record=read_file(path,expected_sha,root,True,32*1024**2)
    return strict_json(raw),record


def verify_record(record, root=None):
    require(type(record) is dict and set(record)==RECORD_KEYS, 'Physical file record schema differs')
    require(all(type(record[k]) is int and record[k]>=0 for k in RECORD_KEYS-{'physical_path','sha256'}), 'Physical file metadata type differs')
    _,actual=read_file(record['physical_path'],record['sha256'],root)
    require(actual==record,'Observed physical file identity changed')
    return Path(record['physical_path'])


def exact_module(path, sha, name):
    raw,record=read_file(path,sha,keep_bytes=True,limit=32*1024**2)
    module=types.ModuleType(name);module.__file__=record['physical_path'];module.__package__=name.rpartition('.')[0]
    exec(compile(raw,record['physical_path'],'exec',dont_inherit=True,optimize=0),module.__dict__)
    verify_record(record)
    return module,record


def parse_execution_identity(job,step,platform,node,env,uid,pid,stat_raw,status,cgroup):
    require(platform=='linux' and node.split('.')[0]=='h100-10','Native CPU must execute on h100-10')
    require(type(job) is str and re.fullmatch('[1-9][0-9]*',job) and env.get('SLURM_JOB_ID')==job,'Actual CPU job identity differs')
    require(type(step) is str and re.fullmatch(r'(?:[0-9]+|batch|extern)',step) and env.get('SLURM_STEP_ID')==step,'Actual CPU step identity differs')
    require(env.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be explicitly hidden')
    fields=stat_raw[stat_raw.rfind(')')+2:].split();start=int(fields[19])
    match=re.search(r'^Uid:\s+(.*)$',status,re.M)
    require(match is not None,'Actual UID missing');uids=[int(v) for v in match.group(1).split()]
    require(type(uid) is int and type(pid) is int and pid>0 and uids==[uid]*4 and start>0,'Actual CPU process identity invalid')
    for label,value in [('job',job),('step',step)]:
        require(re.search(r'(?:^|/)'+label+r'[_-]'+re.escape(value)+r'(?:\.scope)?(?:/|$)',cgroup,re.M),'Actual CPU cgroup '+label+' differs')
    return dict(node=node,pid=pid,uid=uid,uids=uids,start_ticks=start,cgroup=cgroup,
                execution_job_id=job,execution_step_id=step,stored_training_job_id='3304',python=sys.executable,python_version=sys.version)


def execution_identity(job,step):
    return parse_execution_identity(job,step,sys.platform,socket.gethostname(),os.environ,os.getuid(),os.getpid(),
        Path('/proc/self/stat').read_text(),Path('/proc/self/status').read_text(),Path('/proc/self/cgroup').read_text())


def checkpoint_paths(root=RUN):
    root=Path(root);paths={f'actor/{name}':root/'checkpoints/actor/global_step_200'/name for name in HF_NAMES}
    paths.update({f'actor/training_state/rank_{rank}.{suffix}':root/'checkpoints/actor/global_step_200/training_state'/f'rank_{rank}.{suffix}' for rank in range(8) for suffix in ('pt','json')})
    paths.update({'driver/driver.pt':root/'full_state/global_step_200/driver.pt','driver/driver.json':root/'full_state/global_step_200/driver.json','algorithm':root/'checkpoints/algorithm/global_step_200/state.json'})
    return paths


def archive_paths(root=RUN):
    return {f'trajectory_outer_{outer:06d}/{name}':Path(root)/'trajectories'/f'outer_{outer:06d}'/name for outer in range(1,201) for name in ARCHIVE_NAMES}


def object_ids(value):
    require(type(value) is dict and set(value)=={'model','optimizer','scheduler','teacher'} and all(type(v) is int and v>0 for v in value.values()),'Training object identities differ')


def rank_report(report,rank,step,status,pid=None):
    for key,target in dict(version=3,algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',rank=rank,world_size=8,step=step,status=status,scheduler_last_epoch=step,optimizer_state_entries=37,model_dtensor_entries=435).items():
        typed(report.get(key),target,'Rank report differs: '+key)
    require(type(report.get('pid')) is int and report['pid']>0,'Stored rank PID missing')
    if pid is not None:typed(report['pid'],pid,'Stored rank PID changed')
    object_ids(report['object_ids'])


def clock():
    return dict(global_step=200,next_global_step=201,adam_steps=2000,scheduler_last_epoch=200,scheduler_step_count=201,lr=1e-6,teacher_cuda_updates=200,cursor=102400,epoch=0)


def source_files(repo,closure):
    repo=Path(repo)
    require(repo==REPO and repo.resolve(strict=True)==repo,'Frozen source physical path differs')
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Frozen source HEAD is not clean')
    require(type(closure) is dict and set(closure)=={'modules'} and len(closure['modules'])==211,'Native211 import inventory differs')
    records={}
    for name,rec in closure['modules'].items():
        require(type(rec) is dict and set(rec)=={'path','sha256'},'Native closure schema differs')
        _,records[rec['path']]=read_file(rec['path'],rec['sha256'])
        if name.startswith(('search_r1.','verl.','reproduction.')):
            stem=repo.joinpath(*name.split('.'));expected=stem.with_suffix('.py')
            if not expected.is_file():expected=stem/'__init__.py'
            require(Path(rec['path'])==expected,'Native source module origin differs')
    return records


def validate_accepted_headers(manifest,payload):
    expected=dict(version=1,method='pure_opsd',run_root=str(RUN),source_commit=SOURCE,config_sha256=CONFIG,global_step=5,actual_outer_updates=5,next_global_step=6,horizon=200,warmup_steps=57,world_size=8,job_id='3304',uid=os.getuid())
    for key,value in expected.items():
        typed(manifest.get(key),value,'Historical manifest identity differs: '+key)
        # Frozen receipt writer uses its explicit IDENTITY keys; unlike the
        # manifest it does not add a version field to the signed payload.
        if key!='version':typed(payload.get(key),value,'Signed5 payload identity differs: '+key)
    typed(payload.get('status'),'PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW','Historical signed5 status differs')
    require(payload['checkpoint_manifest_sha256']==ACCEPTED5 and canonical(payload['checkpoint_files'])==canonical(manifest['files']),'Signed5 file map differs')
    for key in ('driver_pid','pause_nonce'):
        typed(payload.get(key),manifest[key],'Signed5 historical driver/nonce differs')
    require(type(manifest['driver_pid']) is int and manifest['driver_pid']>0,'Historical driver PID missing')


def accepted_context(receipt_sha, repo=REPO):
    """Offline historical signature/context; never impersonates the old driver."""
    manifest,mrec=read_json(RUN/'full_state/global_step_5/manifest.json',ACCEPTED5,RUN)
    receipt,rrec=read_json(RUN/'acceptance-5-receipt.json',valid_sha(receipt_sha),RUN)
    public,prec=read_json(RUN/'bindings/public.json',PUBLIC,RUN)
    gate,grec=exact_module(Path(repo)/'search_r1/baselines/checkpoint_gate.py',GATE,'final200_exact_gate')
    require(type(receipt) is dict and set(receipt)=={'payload','signature_hex'},'Signed5 envelope differs')
    gate.verify_rsa_signature(receipt['payload'],receipt['signature_hex'],public);payload=receipt['payload']
    validate_accepted_headers(manifest,payload)
    records={r['physical_path']:r for r in (mrec,rrec,prec,grec)}
    def anchor(role):
        rec=manifest['files'][role];require(type(rec) is dict and set(rec)=={'path','sha256'},'Signed anchor schema differs')
        value,actual=read_json(rec['path'],rec['sha256'],RUN);records[actual['physical_path']]=actual
        return value
    config=anchor('resolved_config');typed(hashlib.sha256(canonical(config)).hexdigest(),CONFIG,'Signed actual resolved config differs')
    closure=anchor('import_closure');typed(manifest['files']['import_closure']['sha256'],CLOSURE,'Signed source closure differs')
    records.update(source_files(repo,closure))
    launch=anchor('launch_binding')
    typed(launch.get('source_commit'),SOURCE,'Original launch source differs');typed(launch.get('run_root'),str(RUN),'Original launch root differs')
    require(set(launch['files'])=={'allocation','source','import_closure','data_manifest','model_manifest','tool_manifest','environment'},'Original launch roles differ')
    for role,rec in launch['files'].items():
        require(canonical(rec)==canonical(manifest['files'][role]),'Launch/signed5 role differs: '+role)
        anchor(role)
    workers=[]
    for rank in range(8):
        report=anchor(f'actor/training_state/rank_{rank}.json');rank_report(report,rank,5,'EXACT_FULL_STATE_RELOAD_OK')
        workers.append(dict(rank=rank,pid=report['pid'],object_ids=report['object_ids'],report=report))
    require(len({w['pid'] for w in workers})==8 and manifest['driver_pid'] not in {w['pid'] for w in workers},'Historical worker PIDs alias')
    audit_rec=payload['evidence']['rank_audit'];require(set(audit_rec)=={'path','sha256'},'Signed native5 witness record differs')
    audit,actual=read_json(audit_rec['path'],audit_rec['sha256'],RUN);records[actual['physical_path']]=actual
    require(audit['passed'] is True and audit['scope']=='NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE' and audit['source_commit']==SOURCE and audit['job_id']=='3304' and audit['checkpoint_manifest_sha256']==ACCEPTED5 and len(audit['ranks'])==8,'Signed native5 witness scope differs')
    for rank,w in enumerate(workers):
        prior=audit['ranks'][rank]
        require(type(prior['rank']) is int and prior['rank']==rank and prior['pid']==w['pid'] and prior['object_ids']==w['object_ids'],'Signed native5 worker differs')
    prep_rec=payload['evidence']['preparation_audit'];require(set(prep_rec)=={'path','sha256'},'Signed preparation audit record differs')
    preparation,actual=read_json(prep_rec['path'],prep_rec['sha256'],RUN);records[actual['physical_path']]=actual
    for key,target in dict(version=1,source_commit=SOURCE,job_id='3304',manifest_sha256=ACCEPTED5).items():typed(preparation.get(key),target,'Signed preparation audit identity differs: '+key)
    def historical_process(value,pid):
        require(type(value) is dict,'Historical process must be a mapping')
        typed(value.get('pid'),pid,'Historical PID differs');typed(value.get('uid'),os.getuid(),'Historical UID differs')
        require(type(value.get('start')) is str and re.fullmatch('[1-9][0-9]*',value['start']),'Historical start_ticks must retain actual writer string type')
        require(type(value.get('cgroup')) is str and re.search(r'(?:^|/)job_3304(?:/|$)',value['cgroup'],re.M),'Historical job cgroup differs')
        return value
    driver_identity=historical_process(preparation['actual_live_driver'],manifest['driver_pid'])
    require(type(preparation['actual_live_workers']) is list and len(preparation['actual_live_workers'])==8,'Signed historical8 worker witnesses incomplete')
    for w,live in zip(workers,preparation['actual_live_workers']):w['historical_process']=historical_process(live,w['pid'])
    service_identity=historical_process(preparation['actual_live_service'],preparation['actual_live_service']['pid'])
    require(service_identity['pid'] not in {manifest['driver_pid']}|{w['pid'] for w in workers},'Historical service aliases training')
    for role,pin in [('data_manifest',COMMON_DATA),('model_manifest',MODEL)]:typed(manifest['files'][role]['sha256'],pin,'Shared official data/model manifest changed')
    formal,frec=read_json(ROOT/'tools/formal-71d2501d4512/manifest.json',FORMAL);records[frec['physical_path']]=frec
    typed(formal['source_commit'],SOURCE,'Frozen formal source differs')
    for name,pin in formal['source_files'].items():
        _,rec=read_file(Path(repo)/name,pin);records[rec['physical_path']]=rec
    for name,pin in formal['tools'].items():
        _,rec=read_file(ROOT/'tools/formal-71d2501d4512'/name,pin);records[rec['physical_path']]=rec
    return dict(manifest=manifest,payload=payload,config=config,closure=closure,workers=workers,prior=audit,preparation=preparation,driver_identity=driver_identity,service_identity=service_identity,records=records,gate=gate,
                accepted5=dict(manifest_path=mrec['physical_path'],manifest_sha256=ACCEPTED5,receipt_path=rrec['physical_path'],receipt_sha256=receipt_sha,public_key_path=prec['physical_path'],public_key_sha256=PUBLIC))


def validate_inventory(value):
    require(type(value) is dict,'Observed inventory must be a dict')
    for key,target in dict(version=1,kind=KIND,source_commit=SOURCE,config_sha256=CONFIG,job_id='3304',run_root=str(RUN),source_root=str(REPO),profile='multitraj200',checkpoint_step=200,horizon=200,warmup_steps=57,world_size=8).items():
        typed(value.get(key),target,'Observed200 identity differs: '+key)
    require(set(value['checkpoint_files'])==set(checkpoint_paths()) and set(value['archive_files'])==set(archive_paths()),'Complete observed200 member inventory differs')
    require(value['raw_statuses']==dict(ranks=['FULL_STATE_SERIALIZED_OK']*8,driver='DRIVER_STATE_SERIALIZED_OK'),'Raw SERIALIZED status changed')
    require(type(value['rank_reports']) is list and len(value['rank_reports'])==8,'Observed200 raw rank reports incomplete')
    for rank,report in enumerate(value['rank_reports']):rank_report(report,rank,200,'FULL_STATE_SERIALIZED_OK')
    typed(value['driver_report'].get('status'),'DRIVER_STATE_SERIALIZED_OK','Raw driver scope changed')
    for key,target in dict(step=200,job_id='3304',uid=os.getuid()).items():typed(value['driver_report'].get(key),target,'Raw driver header differs')
    return value


def verified_members(inventory, include_archives=True):
    validate_inventory(inventory);paths={};inodes=set()
    categories=[('checkpoint_files',checkpoint_paths())]
    if include_archives:categories.append(('archive_files',archive_paths()))
    for category,expected in categories:
        for role,rec in inventory[category].items():
            typed(rec['physical_path'],str(expected[role]),'Observed200 member path differs')
            p=verify_record(rec,RUN);inode=(rec['device'],rec['inode'])
            require(inode not in inodes,'Observed200 members alias an inode');inodes.add(inode);paths[role]=p
    return paths


def cpu_modules(context):
    """Reuse the frozen actual CPU import checker, without its launcher/main."""
    closure=context['closure'];records=dict(context['records'])
    sys.path[:]=[str(REPO)]+[p for p in sys.path if p and Path(p).resolve()!=REPO]
    importlib.import_module('search_r1.baselines')
    modules={}
    for leaf in ('holdout','sharedholdout','e1_profile','full_state','checkpoint_gate','trajectory_archive'):
        name='search_r1.baselines.'+leaf;rec=closure['modules'][name]
        if leaf=='full_state':typed(rec['sha256'],FULL_STATE,'Frozen full_state source differs')
        if leaf=='checkpoint_gate':typed(rec['sha256'],GATE,'Frozen checkpoint gate source differs')
        module,physical_rec=exact_module(rec['path'],rec['sha256'],name)
        sys.modules[name]=module;setattr(sys.modules['search_r1.baselines'],leaf,module);modules[leaf]=module
        records[physical_rec['physical_path']]=physical_rec
    name='reproduction.baselines.e1_import_bootstrap';rec=closure['modules'][name]
    checker,physical_rec=exact_module(rec['path'],rec['sha256'],name);sys.modules[name]=checker
    records[physical_rec['physical_path']]=physical_rec
    actual,code_count=checker.verify_imports(REPO,closure,required=())
    require(code_count>0,'No actual native CPU functions audited')
    source_map={}
    for namespace in ('search_r1','verl'):
        for p in sorted((REPO/namespace).rglob('*.py')):
            _,record=read_file(p);records[str(p)]=record;source_map[str(p.relative_to(REPO))]=record['sha256']
    return modules,records,source_map,dict(module_count=len(actual['modules']),function_code_count=code_count)


def write_exclusive(path,value):
    p=Path(path);require(p.is_absolute() and p.parent.resolve(strict=True)==p.parent and not p.is_symlink(),'Physical output parent required')
    raw=canonical(value)+b'\n'
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return read_file(p,hashlib.sha256(raw).hexdigest())[1]


def recheck(records):
    for rec in records.values():verify_record(rec)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
