"""Remote-only source/target peer. No Torch, credentials, signals or GPU calls."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

SOURCE='/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'
TARGET='/data/shaoshitong/search-e1-opsd-signal-weight-backup-20261010'
VARIANTS=('G_POS','G_NEG','H_POS','H_NEG')
NAME=re.compile(r'update-(\d{8})-outer-(\d{5})')
DIRECT_SSH_OPTIONS=['-B','en0','-o','ProxyCommand=none','-o','ProxyJump=none',
    '-o','ControlMaster=no','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=10']


def require(ok,message):
    if not ok:raise ValueError(message)


def stat(path):
    s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def proof(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Require an actual regular file')
    path=path.resolve();before=stat(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    require(before==stat(path),'Source changed while hashing');return dict(path=str(path),sha256=h.hexdigest(),**before)


def checked(ref):
    actual=proof(ref['path']);require(actual['sha256']==ref['sha256'],'File SHA differs')
    if 'bytes' in ref:require(actual['bytes']==ref['bytes'],'File size differs')
    return actual


def read(ref):
    actual=checked(ref);before=stat(actual['path']);value=json.loads(Path(actual['path']).read_text());require(before==stat(actual['path']),'JSON changed during read')
    return value,actual


def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def encode(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();require(len(raw)<=64<<10,'Small transport envelope exceeds64KiB')
    return base64.b64encode(raw).decode()


def decode(value):
    raw=base64.b64decode(value,validate=True);require(len(raw)<=64<<10,'Small transport envelope exceeds64KiB');return json.loads(raw)


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return proof(path)


def no_cuda(node,expected_uid):
    require(os.getuid()==expected_uid and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Wrong approved remote UID or nonempty CUDA visibility')
    require('torch' not in sys.modules,'Remote backup peer must not import Torch')
    return dict(cluster='HKUST',node=node,uid=os.getuid(),hostname=socket.gethostname(),pid=os.getpid(),torch_imported=False)


def science_contract(config):
    return dict(direction='student_to_target_reverse',target_version='opsd-signal-weight-reverse-v1',
        variant=config['variant'],precision=config['precision']['mode'],mixture_alpha=0.,
        signal_version='opsd-five-sibling-centered-signal-weights-v1',weight_amplitude=.9,world_size=4)


def verify_checkpoint_declarations(config,cfg,checkpoint,expected_updates,expected_outer):
    """Pure JSON consistency. This intentionally does not deserialize state/RNG."""
    require(config['precision']=={'mode':'bf16'} and config['baseline_variant']=='R'
        and all(config['scientific'].get(k)==v for k,v in dict(world_size=4,micro_batch=1,global_batch=32,
            gradient_accumulation_full_batch_per_rank=8,original_eight_rank_accumulation=4,
            learning_rate=1e-6,component_clip=10,loss_reduction='trajectory_mean',teacher_frozen=True).items()),
        'Configured four-rank R scientific settings differ')
    require(checkpoint['schema']=='opsd-online-checkpoint-v1' and checkpoint['accepted'] is True
        and checkpoint['run_id']==config['run_id'] and checkpoint['config_sha256']==cfg['sha256']
        and checkpoint['optimizer_updates']==expected_updates and checkpoint['cursor']['outer_index']==expected_outer
        and checkpoint['runtime_identity']==config['runtime_manifest']['sha256']
        and checkpoint['source_model_identity']==config['source_model_manifest']['sha256']
        and checkpoint['contract']==science_contract(config)
        and checkpoint['common_initial_adapter']==config['common_initial_adapter_manifest'],
        'Checkpoint declared runtime/source/science/commoninitial identity differs')
    require(checkpoint['complete_rank_count']==4 and checkpoint['equal_optimizer_steps'] is True
        and checkpoint['optimizer_storage']=='shared_exact_four_rank_optimizer_v1',
        'Actual committed four-rank checkpoint/run/count differs')
    for key in ('all_rank_optimizer_tensor_sha256','all_rank_adapter_sha256'):
        values=checkpoint[key]
        require(len(values)==4 and len(set(values))==1 and all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in values),
            'Declared four-rank digest identity differs: '+key)
    scalers=checkpoint['all_rank_scaler_state']
    require(len(scalers)==4 and all(x==scalers[0] for x in scalers),'Declared four-rank scaler states differ')
    require(checkpoint['all_rng']==['python','numpy','torch_cpu','cuda_all']
        and checkpoint['scheduler']==dict(kind='none',warmup_updates=0),'Declared full RNG/no scheduler differs')
    cursor=checkpoint['cursor']
    require(type(checkpoint['optimizer_updates']) is int and checkpoint['optimizer_updates']>=0
        and type(checkpoint['cumulative_pair_exposure']) is int and checkpoint['cumulative_pair_exposure']>=0
        and type(cursor['outer_index']) is int and type(cursor['next_pair_cursor']) is int and type(cursor['usable_pair_count']) is int
        and 0<=cursor['next_pair_cursor']<=cursor['usable_pair_count']<=512
        and (cursor['next_pair_cursor']==cursor['usable_pair_count'] or cursor['next_pair_cursor']%32==0),
        'Declared cumulative exposure/committed usable pair cursor differs')


def inventory(config_ref,checkpoint_ref,variant,expected_updates,expected_outer,peer_sha):
    identity=no_cuda('HKUST4',1001);require(proof(__file__)['sha256']==peer_sha,'Source peer differs from published code')
    require(variant in VARIANTS and type(expected_updates) is int and expected_updates>=0 and type(expected_outer) is int and 0<=expected_outer<25,'Explicit true update/outer identity required')
    config,cfg=read(config_ref);checkpoint,cp=read(checkpoint_ref)
    root=Path(SOURCE).resolve();config_path=Path(cfg['path']);directory=Path(cp['path']).parent
    require(config_path.is_relative_to(root/('group-'+variant)/'receipts') and Path(cp['path']).name=='manifest.json'
        and directory.parent==root/('group-'+variant)/'runs/checkpoints' and NAME.fullmatch(directory.name),
        'Only this authorized remote study explicit committed checkpoint may be copied')
    require(config['schema']=='opsd-signal-weight-config-v1' and config['variant']==variant and config['owner']['uid']==1001
        and config['scientific']['outer_iterations']==25,'Configured new study identity differs')
    match=NAME.fullmatch(directory.name)
    require(checkpoint['optimizer_updates']==expected_updates==int(match[1]) and checkpoint['cursor']['outer_index']==expected_outer==int(match[2]),
        'Actual committed four-rank checkpoint/run/count differs')
    verify_checkpoint_declarations(config,cfg,checkpoint,expected_updates,expected_outer)
    require(expected_updates in (1,5) or checkpoint['cursor']['next_pair_cursor']==checkpoint['cursor']['usable_pair_count'],
        'Only protected actualu1/u5 or complete outer-end checkpoint is authorized')
    refs=[cp]+[r['file'] for r in checkpoint['ranks']]+[checkpoint['shared_optimizer']]+checkpoint['adapter']['files']
    require({r['rank'] for r in checkpoint['ranks']}==set(range(4)) and len(checkpoint['ranks'])==4,'Missing/duplicate rank states')
    require(all(Path(r['file']['path']).resolve()==directory/('rank-%d-state.pt'%r['rank']) for r in checkpoint['ranks'])
        and len({(Path(r['file']['path']).stat().st_dev,Path(r['file']['path']).stat().st_ino) for r in checkpoint['ranks']})==4,
        'Rank file names or independent inode identity differ')
    require(Path(checkpoint['shared_optimizer']['path']).resolve()==directory/'optimizer-shared.pt',
        'Shared optimizer is outside this exact checkpoint root')
    adapter_root=directory/'adapter'
    require(Path(checkpoint['adapter']['path']).resolve()==adapter_root
        and all(Path(ref['path']).resolve().parent==adapter_root for ref in checkpoint['adapter']['files'])
        and {'adapter_config.json','adapter_model.safetensors'}<= {Path(ref['path']).name for ref in checkpoint['adapter']['files']},
        'Declared adapter directory/two necessary files differ')
    members={}
    for ref in refs:
        actual=checked(ref);path=Path(actual['path']);require(path.is_relative_to(directory),'Checkpoint payload outside its own immutable directory')
        relative=str(path.relative_to(directory));require(relative not in members,'Repeated checkpoint member')
        members[relative]=actual
    actual_files={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    require(actual_files==set(members) and not any(p.is_symlink() for p in directory.rglob('*')),'Checkpoint actual inventory contains unexplained/symlink files')
    raw=Path(cfg['path']).read_bytes();require(len(raw)<32<<10,'Configuration summary is unexpectedly large')
    result=dict(schema='opsd-signal-pinned-backup-inventory-v1',accepted=True,cluster='HKUST',source_node='HKUST4',source_uid=1001,
        source_hostname=identity['hostname'],peer_source_sha256=peer_sha,variant=variant,run_id=config['run_id'],study_id=config['study_id'],
        config=cfg,config_bytes_base64=base64.b64encode(raw).decode(),checkpoint=cp,checkpoint_name=directory.name,
        source_checkpoint_directory=str(directory),optimizer_updates=expected_updates,outer_index=expected_outer,
        cursor=checkpoint['cursor'],protection_reason=checkpoint.get('protection_reason',[]),members=members,
        member_count=len(members),member_bytes=sum(x['bytes'] for x in members.values()),
        numeric_scope='Fresh immutable byte SHA only; no optimizer/tensor deserialization',large_outputs_remote_only=True)
    encode(result);return result


def send(value,peer_sha,stream=None):
    no_cuda('HKUST4',1001);stream=stream or sys.stdout.buffer
    actual=inventory(value['config'],value['checkpoint'],value['variant'],value['optimizer_updates'],value['outer_index'],peer_sha)
    require(actual==value,'Original pinned inventory/config/payload changed before transfer')
    command=['tar','--format=ustar','-cf','-','-C',value['source_checkpoint_directory'],'--']+sorted(value['members'])
    child=subprocess.Popen(command,stdout=stream,stderr=sys.stderr.buffer)
    code=child.wait();require(code==0,'Actual source tar wait returncode differs from0')
    after=inventory(value['config'],value['checkpoint'],value['variant'],value['optimizer_updates'],value['outer_index'],peer_sha)
    require(after==value,'Source checkpoint changed during transfer')
    result=dict(schema='opsd-signal-source-stream-terminal-v1',accepted=True,inventory_sha256=object_sha(value),
        source_tar_actual_wait_returncode=code,before_after_inventory_equal=True,member_count=value['member_count'],member_bytes=value['member_bytes'],
        actual_source_peer=no_cuda('HKUST4',1001),source_peer_sha256=peer_sha,no_signals_sent=True)
    return result


def target_root(value,peer_sha):
    no_cuda('HKUST1',1005);require(proof(__file__)['sha256']==peer_sha==value['peer_source_sha256'],'Target/source peer publication differs')
    require(value['schema']=='opsd-signal-pinned-backup-inventory-v1' and value['accepted'] is True and value['cluster']=='HKUST'
        and value['source_node']=='HKUST4' and value['source_uid']==1001 and value['variant'] in VARIANTS
        and re.fullmatch('[0-9a-f]{32}',value['run_id']) is not None and NAME.fullmatch(value['checkpoint_name']),
        'Unapproved transfer envelope')
    require(value['member_count']==len(value['members']) and value['member_bytes']==sum(x['bytes'] for x in value['members'].values()),'Target inventory accounting differs')
    for name,ref in value['members'].items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts and str(rel)==name and name not in ('','.','..'),
            'Unsafe checkpoint member relative path')
        require(type(ref['bytes']) is int and ref['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',ref['sha256']) is not None,'Bad member size/SHA')
    root=Path(TARGET);require(not root.is_symlink(),'Target root is a symlink')
    destination=root/value['variant']/value['run_id']/'checkpoints'/value['checkpoint_name']
    require(not any(p.is_symlink() for p in [destination,*destination.parents] if p.exists()),'Target ancestor is a symlink')
    return destination


def extract_ustar(path,destination,members):
    """Exact regular-file USTAR, safe exclusive extraction, streamed full SHA."""
    require(not destination.exists(),'Refuse overwrite of extraction evidence');destination.mkdir(parents=True)
    seen=set();result={}
    with Path(path).open('rb') as source:
        while True:
            header=source.read(512);require(len(header)==512,'Truncated USTAR header')
            if header==bytes(512):
                second=source.read(512);require(second==bytes(512),'Require actual two-zero-block USTAR terminator')
                while True:
                    rest=source.read(1<<20)
                    if not rest:break
                    require(not any(rest),'Trailing nonzero USTAR content')
                require(Path(path).stat().st_size%512==0,'Tar stream was not block aligned');break
            require(header[257:263]==b'ustar\0' and header[263:265]==b'00' and header[156:157] in (b'0',b'\0'),'Only canonical regular-file USTAR is admitted')
            checksum=int(header[148:156].rstrip(b'\0 ').lstrip(b' '),8)
            require(checksum==sum(header[:148])+8*32+sum(header[156:]),'USTAR header checksum differs')
            name=header[:100].split(b'\0',1)[0].decode();prefix=header[345:500].split(b'\0',1)[0].decode()
            if prefix:name=prefix+'/'+name
            require(name in members and name not in seen,'Unexpected/duplicate USTAR member')
            size=int(header[124:136].rstrip(b'\0 ').lstrip(b' ') or b'0',8);require(size==members[name]['bytes'],'USTAR declared size differs from actual source inventory')
            target=destination/Path(name);require(target.resolve().is_relative_to(destination.resolve()),'Extraction escaped its new evidence root')
            target.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();remaining=size
            with target.open('xb') as sink:
                while remaining:
                    block=source.read(min(4<<20,remaining));require(block,'Truncated checkpoint member');sink.write(block);digest.update(block);remaining-=len(block)
                sink.flush();os.fsync(sink.fileno())
            require(digest.hexdigest()==members[name]['sha256'],'Received checkpoint member SHA differs')
            padding=source.read((-size)%512);require(len(padding)==(-size)%512 and not any(padding),'Missing/nonzero tar member padding')
            seen.add(name);result[name]=proof(target)
    require(seen==set(members),'Complete source checkpoint file coverage is missing')
    return result


def verify_wait_evidence(value,peer_sha,source_terminal,transport):
    require(source_terminal['schema']=='opsd-signal-source-stream-terminal-v1' and source_terminal['accepted'] is True
        and source_terminal['inventory_sha256']==object_sha(value) and source_terminal['source_tar_actual_wait_returncode']==0
        and source_terminal['before_after_inventory_equal'] is True and source_terminal['source_peer_sha256']==peer_sha,
        'Actual source tar/send terminal is missing or nonzero')
    require(transport['source_ssh_actual_wait_returncode']==transport['target_ssh_actual_wait_returncode']==0
        and transport['source_alias']=='hkust4' and transport['target_alias']=='hkust1'
        and transport['direct_ssh_options']==DIRECT_SSH_OPTIONS and transport['connection_mode']=='direct_no_proxy_no_multiplex'
        and transport['local_checkpoint_payload_files_created']==0,'Actual authorized direct dual SSH wait0/no-local-payload proof missing')


def verify_existing(value,destination):
    marker=destination/'complete-sha-verified.json'
    if not marker.exists():return dict(accepted=False,state='NOT_COMPLETE',destination=str(destination))
    receipt,rref=read(proof(marker));require(receipt['schema']=='opsd-signal-pinned-backup-complete-v1'
        and receipt['accepted'] is True and receipt['inventory']==value and receipt['inventory_sha256']==object_sha(value)
        and receipt['member_count']==value['member_count'] and receipt['member_bytes']==value['member_bytes']
        and receipt['large_outputs_remote_only'] is True and receipt['actual_tar_and_transfer_returncodes_zero'] is True,
        'Existing accepted checkpoint belongs to another pinned input or incomplete transport')
    verify_wait_evidence(value,value['peer_source_sha256'],receipt['source_terminal'],receipt['actual_transport'])
    payload=destination/'payload'
    require(not payload.is_symlink() and {str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file()}==set(value['members'])
        and not any(p.is_symlink() for p in payload.rglob('*')),'Accepted target inventory contains unexpected/symlink payload')
    for name,entry in value['members'].items():
        target=checked(dict(path=str(destination/'payload'/name),sha256=entry['sha256'],bytes=entry['bytes']))
    require(proof(destination/'source-config.json')['sha256']==value['config']['sha256'],'Accepted target config differs')
    return dict(accepted=True,state='COMPLETE_FRESH_SHA_READBACK',receipt=rref,destination=str(destination),member_count=value['member_count'],member_bytes=value['member_bytes'])


def receive(value,peer_sha,stream=None):
    destination=target_root(value,peer_sha);destination.mkdir(parents=True,exist_ok=True);stream=stream or sys.stdin.buffer
    require(shutil.disk_usage(destination).free>=2*value['member_bytes']+(100<<30),
        'Remote target lacks100GiB reserve plus retained tar/payload space; do not affect other ongoing jobs')
    attempt=destination/('attempt-'+uuid.uuid4().hex);attempt.mkdir()
    lock=(destination/'backup.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        require(not (destination/'complete-sha-verified.json').exists(),'Target already has accepted payload; use fresh probe instead')
        tar=attempt/'checkpoint.tar.partial';limit=value['member_bytes']+value['member_count']*1024+(20<<10);count=0;digest=hashlib.sha256()
        with tar.open('xb') as output:
            while True:
                block=stream.read(4<<20)
                if not block:break
                count+=len(block);require(count<=limit,'Tar stream exceeds exact inventory overhead bound')
                output.write(block);digest.update(block)
            output.flush();os.fsync(output.fileno())
        members=extract_ustar(tar,attempt/'payload',value['members'])
        result=dict(schema='opsd-signal-target-receive-pending-v1',accepted=True,inventory=value,inventory_sha256=object_sha(value),
            actual_target_peer=no_cuda('HKUST1',1005),target_peer_source_sha256=peer_sha,tar=proof(tar),tar_stream_sha256=digest.hexdigest(),
            member_fresh_readback=members,member_count=len(members),member_bytes=value['member_bytes'],
            complete=False,needs_actual_source_tar_and_both_ssh_wait0=True,no_signals_sent=True)
        ref=write_once(attempt/'receive-pending.json',result)
        return dict(accepted=True,pending_receipt=ref,destination=str(destination),member_count=len(members),member_bytes=value['member_bytes'],complete=False)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def commit(value,peer_sha,pending_ref,source_terminal,transport):
    destination=target_root(value,peer_sha);pending,pr=read(pending_ref)
    require(Path(pr['path']).parent.parent==destination and Path(pr['path']).name=='receive-pending.json','Pending evidence outside this target checkpoint')
    require(pending['accepted'] is True and pending['inventory']==value and pending['inventory_sha256']==object_sha(value)
        and pending['target_peer_source_sha256']==peer_sha,'Pending target identity differs')
    verify_wait_evidence(value,peer_sha,source_terminal,transport)
    lock=(destination/'backup.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        existing=verify_existing(value,destination)
        if existing['accepted']:return existing
        final_payload=destination/'payload'
        payload=final_payload if final_payload.exists() else Path(pr['path']).parent/'payload'
        require(not payload.is_symlink() and {str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file()}==set(value['members'])
            and not any(p.is_symlink() for p in payload.rglob('*')),'Unaccepted payload differs from exact pinned regular-file inventory')
        readback={}
        for name,entry in value['members'].items():readback[name]=checked(dict(path=str(payload/name),sha256=entry['sha256'],bytes=entry['bytes']))
        if not final_payload.exists():os.rename(payload,final_payload)
        raw=base64.b64decode(value['config_bytes_base64'],validate=True);require(hashlib.sha256(raw).hexdigest()==value['config']['sha256'],'Original configuration byte SHA differs')
        config_path=destination/'source-config.json'
        if config_path.exists():require(proof(config_path)['sha256']==value['config']['sha256'],'Unaccepted original config differs; preserve it')
        else:
            with config_path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        directory_fd=os.open(str(destination),os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        final=dict(schema='opsd-signal-pinned-backup-complete-v1',accepted=True,inventory_sha256=object_sha(value),inventory=value,
            original_target_receive=pr,source_terminal=source_terminal,actual_transport=transport,
            member_readback={n:proof(destination/'payload'/n) for n in value['members']},source_config=proof(destination/'source-config.json'),
            member_count=value['member_count'],member_bytes=value['member_bytes'],large_outputs_remote_only=True,
            actual_tar_and_transfer_returncodes_zero=True,no_signals_sent=True,accepted_unix=time.time(),
            restore_scope='Original checkpoint/config byte copies. Manifest paths still name HKUST4 original layout; dependencies/debug/source weights remain preserved there. No standalone numeric restore claimed.')
        marker=write_once(destination/'complete-sha-verified.json',final)
        return dict(accepted=True,state='COMPLETE_FRESH_SHA_READBACK',receipt=marker,destination=str(destination),member_count=value['member_count'],member_bytes=value['member_bytes'])
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-sha256',required=True)
    sub=p.add_subparsers(dest='action',required=True);inv=sub.add_parser('inventory')
    for key in ('config','config-sha256','checkpoint','checkpoint-sha256','variant'):inv.add_argument('--'+key,required=True)
    inv.add_argument('--expected-updates',type=int,required=True);inv.add_argument('--expected-outer',type=int,required=True)
    for name in ('probe','send','receive','commit'):
        q=sub.add_parser(name);q.add_argument('--inventory-base64',required=True)
        if name=='commit':
            for key in ('pending-receipt','pending-receipt-sha256','source-terminal-base64','transport-base64'):q.add_argument('--'+key,required=True)
    a=p.parse_args()
    if a.action=='inventory':result=inventory(dict(path=a.config,sha256=a.config_sha256),dict(path=a.checkpoint,sha256=a.checkpoint_sha256),a.variant,a.expected_updates,a.expected_outer,a.peer_sha256)
    else:
        value=decode(a.inventory_base64)
        if a.action=='probe':result=verify_existing(value,target_root(value,a.peer_sha256))
        elif a.action=='send':
            result=send(value,a.peer_sha256);sys.stderr.write(json.dumps({'source_stream_terminal':result},sort_keys=True)+'\n');sys.stderr.flush();return
        elif a.action=='receive':result=receive(value,a.peer_sha256)
        else:result=commit(value,a.peer_sha256,dict(path=a.pending_receipt,sha256=a.pending_receipt_sha256),decode(a.source_terminal_base64),decode(a.transport_base64))
    print(json.dumps(result,sort_keys=True));require('torch' not in sys.modules,'Backup peer imported Torch')


if __name__=='__main__':main()
