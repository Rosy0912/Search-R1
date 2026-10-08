"""External CPU bootstrap; only retire-finalize may SIGTERM one bound resident PID.

Never imports CUDA, launches inference/training, changes old sources or sends a
process-group signal. Root publishes/reviews/deploys this separate entry first.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import tarfile
import time
from urllib.request import urlopen

RESIDENT_BASE=Path('/data/shaoshitong/search-e1-retriever-resident-20261008')
ORACLE_BASE=Path('/data/shaoshitong/search-e1-retriever-segmented-20261007')
RESIDENT=dict(pid=4189880,uid=1005,starttick=922361711,run_id='resident-09a3c264ec76419f9c5da1c6eb14b364')
ORACLE=dict(pid=2239529,uid=1005,starttick=914664099)
RESIDENT_DIGEST='c1265ec3f5884e1b3cc61ca52dfedcbf21992ad50bcbf144deb0c9fd85d5152e'
TRANSFER_SHA='08b005d8788fbbbc6c57cf40af4c08202b59c0d89160a6f030fad8d555fc007d'
DATA_SHA={'train':'a334d768230fd85d3465e251aee906060150628775b78ab7ce15e5e789f7489f',
          'validation':'75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76'}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(4<<20),b''):value.update(block)
    return value.hexdigest()


def load_helpers(args):
    """Verify published bytes before executing any helper import."""
    root=Path(args.helper_root).resolve();publication=json.loads(Path(args.publication_json).read_text())
    manifest_path=root/'source-manifest.json';manifest=json.loads(manifest_path.read_text())
    require(publication.get('accepted') is True and publication.get('branch')=='search-e1'
            and publication.get('source_commit')==publication.get('remote_readback_sha')==args.sourcecommit,
            'Unaccepted publication/source commit')
    require(re.fullmatch('[0-9a-f]{40}',args.sourcecommit),'Invalid published commit')
    with manifest_path.open('rb') as stream:require(digest(stream)==publication['source_manifest_sha256'],'Helper manifest mismatch')
    for name,entry in manifest['files'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe helper path')
        with (root/relative).open('rb') as stream:require(digest(stream)==entry['sha256'],'Published helper changed: '+name)
    require({str(p.relative_to(root)) for p in root.rglob('*.py')}=={n for n in manifest['files'] if n.endswith('.py')},'Unpublished helper Python')
    sys.path.insert(0,str(root))
    modules=[importlib.import_module(name) for name in ('prepare','checkpoint','acceptance')]
    require(all(Path(m.__file__).resolve().parent==root for m in modules),'Wrong cached helper import')
    return modules


def human_admission(value,run_id,durable_target):
    require(value.get('cluster')=='HKUST' and value.get('run_id')==run_id
            and value.get('component_clip')==10 and value.get('outer_iterations')==200
            and value.get('explicit_exception') is True and value.get('eight_gpu_execution') is True,
            'Exact latest clip10/outer200/HKUST-local/eight-GPU authorization missing')
    require(value.get('text') and value.get('thread_id') and value.get('observed_unix',0)>0,'Human provenance missing')
    require(value.get('durable_target')==durable_target and Path(durable_target).is_absolute(),'Durable target not explicitly bound')


def health(port):
    with urlopen('http://10.120.17.101:'+str(port)+'/health',timeout=10) as stream:return json.load(stream)


def oracle_receipt(checkpoint,acceptance,config):
    control,control_ref=checkpoint.checked_json(checkpoint.fileproof(ORACLE_BASE/'receipts/admit-requests.json'))
    current=acceptance.process(ORACLE['pid']);require(acceptance.same_process(current,ORACLE),'Original oracle identity changed')
    result=dict(accepted=True,retriever_url=config['paths']['retriever_url'],**ORACLE,
                source_identity_sha256=control['source_identity_sha256'],original_admission=control_ref,
                full_gpu_gate_path=control['full_gpu_http_gate_path'],full_gpu_gate_sha256=control['full_gpu_http_gate_sha256'])
    acceptance.retriever_admission(config,result)
    return result


def verify_backup(value,checkpoint,base):
    require(value.get('accepted') is True and value.get('pid')==RESIDENT['pid'] and value.get('uid')==RESIDENT['uid']
            and value.get('tick',value.get('starttick'))==RESIDENT['starttick'],'Backup belongs to another resident')
    archive=Path(value.get('node_archive',str(base/'receipts/resident-backup.tar.gz')))
    proof=checkpoint.checked_file(dict(path=str(archive),sha256=value['archive_sha256']))
    sources=set();logs=0
    with tarfile.open(archive,'r:*') as tar:
        members={m.name.removeprefix('./'):m for m in tar.getmembers()}
        require(len(members)==len(tar.getmembers()),'Duplicate backup archive member')
        for item in value['files']:
            relative=Path(item['relative_path']);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe backup member')
            path=Path(item['path']).resolve();require(path==RESIDENT_BASE.resolve()/relative,'Backup path escaped exact resident')
            member=members[str(relative)];require(member.isfile() and member.size==item['bytes'],'Backup member missing/aliased/size changed')
            with tar.extractfile(member) as stream:require(digest(stream)==item['sha256'],'Uploaded archive member differs')
            if relative.parts[0]=='logs':
                logs+=1;require(path.stat().st_size>=item['bytes'],'Preserved log prefix was truncated')
                value_hash=hashlib.sha256();remaining=item['bytes']
                with path.open('rb') as stream:
                    while remaining:
                        block=stream.read(min(4<<20,remaining));require(block,'Short log prefix');value_hash.update(block);remaining-=len(block)
                require(value_hash.hexdigest()==item['sha256'],'Live log differs from preserved prefix')
            else:
                checkpoint.checked_file(dict(path=str(path),sha256=item['sha256'],bytes=item['bytes']))
                if relative.parts[0]=='source' and relative.suffix=='.py':sources.add(str(relative))
    require(len(sources)>=10 and {'source/server.py','source/launch_cpu.py'}<=sources and logs>0,'Complete source/log backup absent')
    require(value.get('local_archive') and value.get('local_extracted'),'Retained Mac archive/readback locations absent')
    return dict(accepted=True,uploaded_archive=proof,source_files=len(sources),preserved_log_files=logs,
                local_archive=value['local_archive'],local_extracted=value['local_extracted'],
                scope='Uploaded backup verified by bytes; original growing logs verified as preserved prefixes, originals retained')


def prepare(args,p,c,a):
    base=Path(args.base).resolve();receipts=base/'receipts';receipts.mkdir(parents=True,exist_ok=True)
    require(re.fullmatch('[0-9a-f]{32}',args.run_id) and os.getuid()==1005,'Wrong new run/UID')
    config=p.template(base);require(not Path(config['paths']['output_root']).exists(),'Preserve existing runs; fresh bootstrap cannot restart')
    human,human_ref=c.checked_json(c.fileproof(args.human_authorization_json));human_admission(human,args.run_id,args.durable_target)
    config.update(run_id=args.run_id,source_commit=args.sourcecommit)
    config['paths']['helper_root']=str(Path(args.helper_root).resolve())
    config['scientific'].update(component_clip=10,outer_iterations=200);config['retention']['authorized']=True
    config['authorization']={name:human['text'] for name in config['authorization']}
    config['helper_source_manifest']=c.fileproof(Path(args.helper_root)/'source-manifest.json')
    config['runtime_manifest']=c.fileproof(Path(p.OLD)/'receipts/preparation-3-runtime.sha256.json')
    require(config['runtime_manifest']['sha256']==p.RUNTIME_SHA,'Original305 identity changed')
    destination=Path(config['paths']['runtime'])
    if not destination.exists():p.copy_runtime(Path(p.OLD)/'runtime/opsd-preparation3-xformers',config['runtime_manifest'],destination)
    runtime_proof=p.verify_runtime(config)
    transfer,transfer_ref=c.checked_json(dict(path=str(Path(p.OLD)/'receipts/archive100-transfer-accepted.json'),sha256=TRANSFER_SHA))
    require(transfer.get('accepted') is True and Path(transfer['target']).resolve()==Path(config['paths']['model']).resolve() and len(transfer['files'])==12,'Original source100 transfer mismatch')
    files=[c.checked_file(dict(path=str(Path(config['paths']['model'])/name),**proof)) for name,proof in sorted(transfer['files'].items())]
    config['source_model_manifest']=c.exclusive_json(receipts/'source100-model-files.json',dict(files=files,original_transfer=transfer_ref,source_grpo_step=100))
    for name,count in [('train',169103),('validation',512)]:
        proof=c.checked_file(dict(path=config['paths'][name],sha256=DATA_SHA[name]))
        with Path(proof['path']).open('rb') as stream:require(sum(1 for line in stream if line.endswith(b'\n'))==count,'Dataset row count changed')
        require(c.stat_identity(proof['path'])=={k:proof[k] for k in c.stat_identity(proof['path'])},'Dataset changed while counting')
        config[name+'_identity']=proof
    storage=dict(accepted=True,cluster='HKUST',mode='human_hkust_local_exception',output_root=config['paths']['output_root'],
        human_authorization=human,human_authorization_receipt=human_ref,
        backup_policy=dict(durable_target=args.durable_target,synchronize_completed_checkpoints=True,
                           preserve_raw_debug=True,policy='Root synchronizes completed manifests/files and verifies target SHA before node release; no shared-storage claim'))
    a.storage_admission(config,storage)
    config['gates']['storage']=c.exclusive_json(receipts/'storage-local-human-exception.json',storage)
    config['gates']['publication']=c.fileproof(args.publication_json)
    a.publication_admission(config,a.bound(config['gates']['publication']))
    config['gates']['retriever_admission']=c.exclusive_json(receipts/'oracle18004-admission.json',oracle_receipt(c,a,config))
    config['bootstrap_human_authorization']=human_ref
    config['bootstrap_launch_environment']=dict(E1_BASE=str(base),E1_PROFILE='opsd-online-archive100',E1_RUN_ID=args.run_id,
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONPATH=str(destination/'upstream/Search-R1')+os.pathsep+str(destination),
        HF_HOME=str(Path(p.OLD)/'hf-cache'),HF_DATASETS_CACHE=str(Path(p.OLD)/'hf-datasets'),
        XDG_CACHE_HOME=str(Path(p.OLD)/'cache'),TMPDIR=str(Path(p.OLD)/'tmp'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
        E1_INFERENCE_UTIL='0.85',WANDB_MODE='offline',VLLM_ATTENTION_BACKEND='XFORMERS')
    backup,backup_ref=c.checked_json(c.fileproof(args.resident_backup_json));backup_proof=verify_backup(backup,c,base)
    config['bootstrap_resident_backup']=backup_ref
    return c.exclusive_json(receipts/'config.preliminary.json',config),c.exclusive_json(receipts/'bootstrap-prepared.json',
        dict(accepted=True,run_id=args.run_id,runtime=runtime_proof,resident_backup=backup_proof,launch_allowed=False,no_signals_sent=True))


def inventory_admission(config,a,resident_present):
    devices,apps=a.inventory();require([d['index'] for d in devices]==list(range(8)) and [d['uuid'] for d in devices]==config['owner']['gpu_uuids'],'GPU allocation changed')
    a.require(a.same_process(a.process(ORACLE['pid']),ORACLE),'Oracle identity changed')
    resident_cards=set()
    for app in apps:
        if app['pid']==ORACLE['pid']:require(app['gpu_uuid']==devices[7]['uuid'] and app['used_mib']<=1024,'Oracle sharing limit failed')
        elif resident_present and app['pid']==RESIDENT['pid']:resident_cards.add(app['gpu_uuid']);require(app['gpu_uuid'] in {d['uuid'] for d in devices[:4]},'Resident GPU scope changed')
        else:raise RuntimeError('Foreign GPU process; no signal authorized: '+str(app['pid']))
    require(not resident_present or resident_cards=={d['uuid'] for d in devices[:4]},'Resident does not own exactly original four GPUs')
    require(all(d['actual_free_mib']>=(6144 if resident_present and d['index']<4 else 20480) for d in devices),'Direct memory.free gate failed')
    return dict(devices=devices,applications=apps,checked_unix=time.time())


def retire_finalize(args,p,c,a):
    base=Path(args.base).resolve();config,preliminary=c.checked_json(c.fileproof(base/'receipts/config.preliminary.json'))
    require(config['run_id']==args.run_id and config['source_commit']==args.sourcecommit and not Path(config['paths']['output_root']).exists(),'Wrong/used preliminary run')
    human,_=c.checked_json(config['bootstrap_human_authorization']);human_admission(human,args.run_id,args.durable_target)
    backup,_=c.checked_json(config['bootstrap_resident_backup']);backup_proof=verify_backup(backup,c,base)
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Exact pidfd signaling unavailable; no unsafe PID fallback')
    with ExitStack() as stack:
        lock_paths=[str(base/'online-experiment.lock')]+config['resources']['shared_gpu_lease_paths']
        for path in lock_paths:
            handle=stack.enter_context(open(path,'a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=a.process(RESIDENT['pid']);require(a.same_process(identity,RESIDENT) and identity['run_id']==RESIDENT['run_id']
            and str(RESIDENT_BASE/'source/server.py') in identity['argv'],'Resident identity/entry changed; no signal')
        h=health(18005);require(h.get('pid')==RESIDENT['pid'] and h.get('uid')==1005 and h.get('starttick')==RESIDENT['starttick']
            and h.get('source_identity_sha256')==RESIDENT_DIGEST and h.get('pending_requests')==0,'Resident busy or health/source changed; no signal')
        before=inventory_admission(config,a,True);a.retriever_admission(config,a.bound(config['gates']['retriever_admission']))
        before_ref=c.exclusive_json(base/'receipts/resident-before-retirement.json',dict(accepted=True,run_id=args.run_id,
            owner=identity,health=h,inventory=before,backup=backup_proof,human_authorization=config['bootstrap_human_authorization'],locks=lock_paths))
        fd=os.pidfd_open(RESIDENT['pid'])
        try:
            require(a.same_process(a.process(RESIDENT['pid']),RESIDENT),'PID changed after pidfd open; no signal')
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            deadline=time.monotonic()+args.timeout
            while Path('/proc/'+str(RESIDENT['pid'])).exists():
                if time.monotonic()>=deadline:
                    c.exclusive_json(base/'receipts/resident-retirement-timeout.json',dict(accepted=False,run_id=args.run_id,before=before_ref,signal='SIGTERM',sigkill_sent=False))
                    raise RuntimeError('Resident still exists; no SIGKILL, no finalized config, preserve and inspect')
                time.sleep(1)
        finally:os.close(fd)
        after=inventory_admission(config,a,False);a.retriever_admission(config,a.bound(config['gates']['retriever_admission']))
        transition=dict(accepted=True,cluster='HKUST',new_run_id=args.run_id,gpu_uuids=config['owner']['gpu_uuids'],
            before=before_ref,retired_resident=RESIDENT,resident_terminal='ABSENT',after=after,allowed_gpu7_oracle=ORACLE,
            preserved_backup=config['bootstrap_resident_backup'],locks=lock_paths,human_authorization=config['bootstrap_human_authorization'],
            superseded_unstarted_scopes=['M3 counterfactual GPU','M4 native512 GPU'],signal='pidfd SIGTERM only exact resident',sigkill_sent=False,
            no_old_grpo_or_opsd_restarted=True,old_artifacts_retained=True,k4_untouched=True,checked_unix=time.time())
        config['gates']['resource_transition']=c.exclusive_json(base/'receipts/resource-transition.json',transition)
        p.validate_config(config,launch=True);a.publication_admission(config,a.bound(config['gates']['publication']));a.storage_admission(config,a.bound(config['gates']['storage']))
        return c.exclusive_json(base/'receipts/config.json',config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('helper-root','base','run-id','publication-json','sourcecommit','human-authorization-json','resident-backup-json','durable-target'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=['prepare','retire-finalize'],required=True);parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args();require(0<args.timeout<=600,'Timeout must be positive and at most600 seconds')
    p,c,a=load_helpers(args)
    result=prepare(args,p,c,a) if args.mode=='prepare' else retire_finalize(args,p,c,a)
    print(json.dumps(dict(mode=args.mode,result=result,gpu_started=False)))


if __name__=='__main__':main()
