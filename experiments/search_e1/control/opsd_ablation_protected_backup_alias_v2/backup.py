"""Independent Mac protected u0/u1/u5 backup, without any training lease.

Actual missing states -> PENDING; insufficient space -> BACKLOG. These are
backup states and never authorization to hold, stop or restart GPU work.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
PREFIX='experiments/search_e1/control/opsd_ablation_protected_backup_alias_v2/'
SSH=['/usr/bin/ssh','-oBatchMode=yes','-oConnectTimeout=10','hkust1']
PYTHON='/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python'


def load_operation(path,sha):
    path=Path(path)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('Operation source differs')
    ns=dict(__name__='_protected_operation',__file__=str(path))
    exec(compile(raw,str(path),'exec'),ns)
    return raw,ns


def publication(b,root,manifest_sha,pub_path,pub_sha,commit,run):
    manifest,mp=b['checked_json'](root/'source-manifest.json',manifest_sha)
    actual={p.name for p in root.iterdir() if p.name!='source-manifest.json'}
    b['require'](actual==set(manifest['files']), 'Independent helper inventory differs')
    proofs=[]
    for name,row in manifest['files'].items():
        b['safe_name'](name);b['require']('/' not in name,'Nested helper source refused')
        proofs.append(dict(relative_path=name,**b['fileproof'](root/name,dict(path=str(root/name),**row))))
    pub,pp=b['checked_json'](pub_path,pub_sha)
    b['require'](pub.get('accepted') is True and pub.get('branch')=='search-e1'
                 and re.fullmatch('[0-9a-f]{40}',commit or '')
                 and pub.get('source_commit')==pub.get('remote_readback_sha')==commit
                 and pub.get('source_manifest_sha256')==mp['sha256']
                 and (pub.get('run_id')==run or run in pub.get('run_ids',{}).values()),
                 'Exact published source/readback/run binding absent')
    for proof in proofs+[dict(relative_path='source-manifest.json',**mp)]:
        rows=[x for x in pub['files'] if x['path']==PREFIX+proof['relative_path']]
        b['require'](len(rows)==1 and rows[0].get('equal') is True
                     and rows[0]['sha256']==proof['sha256'] and rows[0]['bytes']==proof['bytes'],
                     'Payload differs from remote publication readback')
    for proof in proofs+[mp,pp]:b['stable'](proof)
    return dict(manifest=mp,publication=pp,source_commit=commit,files=proofs)


def remote_command(operation,mode,payload,sha):
    encoded=base64.b64encode(payload).decode('ascii')
    noun='identity' if mode=='pin' else 'scope'
    remote=shlex.join(['env','-u','E1_RUN_ID','CUDA_VISIBLE_DEVICES=',
        'PYTHONDONTWRITEBYTECODE=1',PYTHON,'-B','-c',operation.decode(),mode,
        '--'+noun+'-base64',encoded,'--'+noun+'-sha256',sha])
    return SSH+[remote]


def collect(b,destination,label,argv,data=None):
    result=subprocess.run(argv,input=data,capture_output=True,check=False)
    for suffix,raw in [('stdout',result.stdout),('stderr',result.stderr)]:
        with (destination/(label+'.'+suffix)).open('xb') as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
    proof=b['write_json'](destination/(label+'-execution.json'),dict(argv=argv,
        actual_returncode=result.returncode,stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr).hexdigest(),observed_unix=time.time(),
        no_node_writes=True,no_gpu_or_training_lock=True))
    b['require'](result.returncode==0, 'Read-only '+label+' transport failed; preserved actual result')
    return result.stdout,proof


def require_space(b,value,available):
    need=b['required_space'](value,5<<30)
    return dict(accepted=available>=need,status='READY' if available>=need else 'BACKLOG',
                actual_free_bytes=available,required_free_bytes=need,
                archive_plus_extraction_plus_margin=True,margin_bytes=5<<30,
                backup_is_training_dependency=False)


def verify_after(b,scope_sha,inventory_ref,value,after):
    b['require'](after.get('schema')=='opsd-ablation-protected-source-after-v1'
        and after.get('accepted') is True and after['scope_sha256']==scope_sha
        and after['inventory_sha256']==inventory_ref['sha256'] and after['identity']==value['identity']
        and after['protected']==value['protected'] and after['files']==42
        and after['all_source_metadata_stable'] is True and after['all_core_file_sets_unchanged'] is True,
        'Source-after proof differs from exact frozen scope/inventory')


def backup(args):
    root=Path(__file__).absolute().parent
    raw,op=load_operation(root/'source-operation.py',args.operation_sha256)
    b=op['primitives'](args.source_helper)
    goal,gp=b['checked_json'](args.goal,args.goal_sha256)
    groups=b['validate_goal'](goal)
    matches=[g for g in groups if g['variant']==args.variant]
    b['require'](len(matches)==1 and args.variant in op['VARIANTS'],'Only exact R/W/P/J goal identity')
    group=matches[0]
    pub=publication(b,root,args.source_manifest_sha256,args.publication,args.publication_sha256,
                    args.source_commit,group['run_id'])
    b['require'](next(x['sha256'] for x in pub['files'] if x['relative_path']=='source-operation.py')
                 ==args.operation_sha256,'Executed operation differs from published payload')
    identity=dict(schema='opsd-ablation-protected-identity-v1',variant=group['variant'],run_id=group['run_id'],
        config=group['config'],output_root=group['output_root'],
        goal=dict(path=args.remote_goal,sha256=args.goal_sha256))
    op['identity_shape'](identity)
    destination=Path(args.destination)
    b['require'](destination.is_absolute() and not destination.exists() and not destination.is_symlink(),
                 'New independent backup attempt destination required')
    b['checked_path'](destination.parent,directory=True)
    destination.mkdir(mode=0o700);b['fsync_dir'](destination.parent)
    archive=destination/'protected-checkpoints.tar.partial'
    try:
        b['write_json'](destination/'attempt.json',dict(accepted=False,identity=identity,source_publication=pub,
            no_node_writes=True,backup_is_training_dependency=False,started_unix=time.time()))
        iraw=b['canonical'](identity);isha=hashlib.sha256(iraw).hexdigest()
        sraw,pinexec=collect(b,destination,'pin',remote_command(raw,'pin',iraw,isha))
        scope=json.loads(sraw)
        if scope.get('accepted') is not True:
            b['require'](scope.get('schema')=='opsd-ablation-protected-pin-v1' and scope.get('status')=='PENDING'
                and scope['identity']==identity and bool(scope['missing_updates']), 'Invalid missing-state response')
            b['write_json'](destination/'pending.json',dict(status='PENDING',identity=identity,
                missing_updates=scope['missing_updates'],pin_execution=pinexec,backup_is_training_dependency=False))
            return dict(accepted=False,status='PENDING',missing_updates=scope['missing_updates'],destination=str(destination))
        op['scope_shape'](scope);b['require'](scope['identity']==identity,'Discovered scope identity differs')
        # Preserve the exact original bytes of the newly frozen three-manifest
        # scope. All following stages carry this same SHA, never mutable latest.
        with (destination/'scope.json').open('xb') as handle:
            handle.write(sraw);handle.flush();os.fsync(handle.fileno())
        ssha=hashlib.sha256(sraw).hexdigest();sp=b['fileproof'](destination/'scope.json')
        vraw,vexec=collect(b,destination,'inventory',remote_command(raw,'inventory',sraw,ssha))
        value=json.loads(vraw);expected=op['validate_inventory'](b,scope,ssha,value)
        ip=b['fileproof'](destination/'inventory.stdout')
        space=require_space(b,value,shutil.disk_usage(destination).free)
        b['write_json'](destination/'space.json',space)
        if not space['accepted']:
            b['write_json'](destination/'backlog.json',dict(status='BACKLOG',identity=identity,scope=sp,
                inventory=ip,space=space,node_originals_retained=True,backup_is_training_dependency=False))
            return dict(accepted=False,status='BACKLOG',space=space,destination=str(destination))
        command=remote_command(raw,'stream',sraw,ssha)
        with archive.open('xb') as out,(destination/'transfer.stderr').open('xb') as err:
            child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=out,stderr=err)
            error=None
            try:
                b['write_json'](destination/'transfer-start.json',dict(actual_pid=child.pid,argv=command,
                    inventory=ip,scope=sp,started_unix=time.time(),backup_is_training_dependency=False))
                child.communicate(vraw)
            except BaseException as exc:
                error=type(exc).__name__+': '+str(exc)
                raise
            finally:
                if child.stdin is not None and not child.stdin.closed:child.stdin.close()
                rc=child.wait();out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
                b['write_json'](destination/'transfer-terminal.json',dict(actual_returncode=rc,local_error=error,
                    ended_unix=time.time(),status_not_inferred_from_absence=True))
        b['require'](rc==0 and error is None,'Actual stream failed; partial retained')
        araw,aexec=collect(b,destination,'source-after',remote_command(raw,'after',sraw,ssha),vraw)
        after=json.loads(araw);verify_after(b,ssha,ip,value,after)
        ap=b['fileproof'](destination/'source-after.stdout')
        local,arch=b['extract_verified'](archive,destination/'verified-files',expected)
        for proof in (gp,sp,ip,ap):b['stable'](proof)
        final=destination/'protected-checkpoints.tar';b['require'](not final.exists(),'Existing final refused')
        os.rename(archive,final);b['fsync_dir'](destination)
        finalref=b['fileproof'](final,dict(path=str(final),sha256=arch['sha256'],bytes=arch['bytes']))
        complete=dict(schema='opsd-ablation-protected-local-backup-v1',accepted=True,
            identity=identity,protected=scope['protected'],scope=sp,inventory=ip,source_after=ap,
            source_publication=pub,archive=finalref,files=local,file_count=len(local),core_file_count=39,
            closure_file_count=3,actual_transfer_returncode=rc,independent_full_local_disk_readback=True,
            all_source_full_SHA_before=True,all_source_metadata_stable_after=True,
            source_after_execution=aexec,inventory_execution=vexec,pin_execution=pinexec,
            initial_optimizer_file_copied=True,node_originals_retained=True,
            numerical_state_acceptance=False,backup_is_training_dependency=False,completed_unix=time.time())
        proof=b['write_json'](destination/'complete-sha-verified.json',complete)
        return dict(accepted=True,status='COMPLETE',complete=proof,files=len(local),bytes=value['total_bytes'])
    except BaseException as error:
        b['write_json'](destination/'failed.json',dict(accepted=False,status='FAILED_BACKUP_ONLY',
            identity=identity,reason=type(error).__name__+': '+str(error),partial_retained=archive.exists(),
            node_originals_retained=True,backup_is_training_dependency=False,observed_unix=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('goal','goal-sha256','remote-goal','variant','destination','source-helper',
                 'operation-sha256','source-manifest-sha256','publication','publication-sha256','source-commit'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();result=backup(a)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
