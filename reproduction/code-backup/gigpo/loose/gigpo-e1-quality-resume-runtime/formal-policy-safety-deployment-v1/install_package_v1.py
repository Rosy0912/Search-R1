"""Template: isolated immutable install only; signed inventory, no productionREADY or sbatch.

Traversal/link/overwrites/unknown inventory reject before mkdir. Primitives and
bounded archive limits follow the independently reviewed calibrationv9 installer.
"""
import json
import os
from pathlib import Path,PurePosixPath
import socket
import stat
import tarfile
from deployment_contract_v1 import (ROOT,HELPERS,CONTROLS,CANDIDATE,SCOPE,INSTALL_INTENT,INSTALL_RECEIPT,
    RUNTIME_MANIFEST_SHA,RUNTIME_REVIEW_SHA,require,sha,record,read_bound,write_once)


def digest(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def read_archive(archive,expected_sha,members):
    require(sha(archive)==expected_sha,'Package SHA changed')
    expected={item['member']:item for item in members}
    require(len(expected)==len(members) and members,'Empty/duplicate signed package inventory')
    values={}
    with tarfile.open(archive,'r:gz') as stream:
        raw=stream.getmembers();require(len(raw)==len(expected) and len({m.name for m in raw})==len(raw),'Unknown/duplicate tar inventory')
        total=0
        for item in raw:
            p=PurePosixPath(item.name)
            require(not p.is_absolute() and len(p.parts)==2 and '..' not in p.parts and '.' not in p.parts
                and str(p)==item.name and item.name in expected,'Unsafe or unknown tar path')
            require(item.isfile() and not item.issym() and not item.islnk() and not item.issparse(),'Link/sparse/non-file member forbidden')
            require(0<=item.size<=12*1024*1024,'Oversized tar member');total+=item.size
            require(total<=64*1024*1024,'Oversized uncompressed package')
            data=stream.extractfile(item).read();bound=expected[item.name]
            require(len(data)==item.size==bound['bytes'] and digest(data)==bound['sha256'],'Tar member bytes/SHA changed')
            values[item.name]=data
    require(set(values)==set(expected),'Missing package member');return values


def installation_plan(archive,plan_record,review_record):
    plan=read_bound(plan_record);review=read_bound(review_record)
    require(plan['status']=='DRAFT_IMMUTABLE_INSTALL_PLAN_NOT_PRODUCTION_READY' and plan['scope']==SCOPE
        and plan['source_commit']=='52befe968c856162d1bc79cd0171bb38309f4f9e' and plan['ready'] is False and plan['submitted'] is False
        and plan['runtime_manifest_sha256']==RUNTIME_MANIFEST_SHA and plan['runtime_static_review_sha256']==RUNTIME_REVIEW_SHA,
        'Installation plan cannot establish training readiness')
    require(review['status']=='PASS' and review['scope']=='GIGPO_FORMAL_IMMUTABLE_INSTALL_ONLY_NOT_PRODUCTION'
        and review['plan_sha256']==plan_record['sha256'] and review['package_sha256']==plan['package']['sha256']
        and review['installer_sha256']==sha(Path(__file__)) and review['actual_formal_GPU_accepted'] is False,
        'Exact independent installer/package review missing')
    payloads=read_archive(archive,plan['package']['sha256'],plan['members'])
    candidate_target=Path(plan['candidate_target'])
    import re
    require(candidate_target.parent==ROOT/'logs' and re.fullmatch(r'FORMAL-PROTECTION-CANDIDATE-(?:TEMPLATE|FINAL)-52BEFE9-v[1-9][0-9]*\.json',candidate_target.name) is not None,'Wrong new versioned candidate target')
    raw=payloads['candidate/'+candidate_target.name];require(digest(raw)==plan['candidate']['sha256'],'Wrong package candidate')
    c=json.loads(raw)
    require(c['status']=='CANDIDATE_NOT_READY' and c['helpers']==str(HELPERS) and c['controls']==str(CONTROLS)
        and c['training_submitted'] is False and c['extra_AB_diagnostic_Adam_required']==0,'Wrong isolated NOT_READY candidate')
    files={r['path']:r for r in c['files']};proofs={r['path']:r for r in c['evidence']}
    require(len(files)==len(c['files']) and len(proofs)==len(c['evidence']),'Duplicate candidate file/evidence inventory')
    manifest_raw=payloads['proof/FORMAL-PROTECTION-DRAFT-IMPLEMENTATION-v4.json']
    require(digest(manifest_raw)==RUNTIME_MANIFEST_SHA,'Frozen formalv4 manifest differs')
    review_raw=payloads['proof/FORMAL-PROTECTION-DRAFT-INDEPENDENT-REVIEW-v4.json']
    require(digest(review_raw)==RUNTIME_REVIEW_SHA and c['runtime_manifest']['sha256']==RUNTIME_MANIFEST_SHA
        and c['runtime_static_review']['sha256']==RUNTIME_REVIEW_SHA,'Reviewed runtime proof original changed')
    static=json.loads(review_raw)
    require(static['status']=='PASS' and static['scope']=='LOCAL_STATIC_AND_CPU_DRAFT_ONLY_NOT_DEPLOYMENT_READY'
        and static['runtime_manifest_sha256']==RUNTIME_MANIFEST_SHA and static['actual_formal_GPU_hook_accepted'] is False,
        'Static scope overstated')
    manifest=json.loads(manifest_raw);frozen={Path(r['path']).name:r for r in manifest['files']}
    runtime=[r for r in plan['members'] if r['member'].startswith('runtime/')]
    controls=[r for r in plan['members'] if r['member'].startswith('control/')]
    require(len(runtime)==len(frozen)==29 and len(controls)==7,'Not exact reviewed runtime/control inventory')
    for item in plan['members']:
        category,name=item['member'].split('/',1);destination=Path(item['path'])
        bound={k:item[k] for k in ('path','bytes','sha256')}
        if category=='runtime':
            require(destination==HELPERS/name and files.get(str(destination))==bound
                and {k:frozen[name][k] for k in ('bytes','sha256')}=={k:item[k] for k in ('bytes','sha256')}
                and item['reuse_same_bytes_allowed'] is False,'Frozen runtime/new directory mismatch')
        elif category=='control':require(destination==CONTROLS/name and files.get(str(destination))==bound and item['reuse_same_bytes_allowed'] is False,'Wrong new control path')
        elif category=='proof':require(destination==ROOT/'logs'/name and proofs.get(str(destination))==bound and item['reuse_same_bytes_allowed'] is True,'Proof aliases/bytes not closed')
        elif category=='candidate':require(destination==candidate_target and name==candidate_target.name and bound['sha256']==plan['candidate']['sha256'] and item['reuse_same_bytes_allowed'] is True,'Wrong versioned candidate')
        else:raise ValueError('Unknown archive category')
    transferred={r['path'] for r in plan['members']}
    expected=[r for r in c['evidence'] if r['path'] not in transferred]
    require(plan['required_existing_evidence']==expected,'Existing proof path/bytes/SHA closure changed')
    return plan,payloads


def directory(path):
    info=os.lstat(path)
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and Path(path).resolve()==Path(path),'Parent missing/not directory/symlink')


def preflight(plan):
    require(socket.gethostname().split('.')[0]=='h100-8','Installation only on authorized h100-8')
    for p in (ROOT,ROOT/'tools',ROOT/'logs'):directory(p)
    require(not os.path.lexists(HELPERS) and not os.path.lexists(CONTROLS),'New helper/control path already exists; no overwrite')
    require(not os.path.lexists(INSTALL_INTENT) and not os.path.lexists(INSTALL_RECEIPT),'Existing/uncertain installation: inspect, never repeat')
    for item in plan['required_existing_evidence']:
        require(Path(item['path']).is_file() and not Path(item['path']).is_symlink() and record(item['path'])==item,'Changed/missing required existing proof')
    reused=[]
    for item in plan['members']:
        path=Path(item['path'])
        if os.path.lexists(path):
            require(item['reuse_same_bytes_allowed'] and path.is_file() and not path.is_symlink()
                and record(path)=={k:item[k] for k in ('path','bytes','sha256')},'Existing target never overwritten')
            reused.append(str(path))
    return reused


def write_new(path,data):
    fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    parent=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(parent)
    finally:os.close(parent)


def install(archive,plan_record,review_record):
    plan,payloads=installation_plan(archive,plan_record,review_record);reused=preflight(plan)
    write_once(INSTALL_INTENT,dict(status='CLAIMED_NO_OVERWRITE_OR_RETRY',plan=plan_record,review=review_record,package_sha256=plan['package']['sha256']))
    installed=[]
    try:
        os.mkdir(HELPERS,0o755);os.mkdir(CONTROLS,0o755)
        for item in plan['members']:
            if item['path'] not in reused:write_new(item['path'],payloads[item['member']]);installed.append(item['path'])
            require(record(item['path'])=={k:item[k] for k in ('path','bytes','sha256')},'Installed byte identity changed')
        result=dict(status='INSTALLED_IMMUTABLE_DRAFT_NOT_PRODUCTION_READY',installed=installed,reused_same_bytes=reused,plan=plan_record,
            package_sha256=plan['package']['sha256'],ready=False,submitted=False,actual_formal_GPU_accepted=False)
        write_once(INSTALL_RECEIPT,result);return result
    except BaseException as error:
        if not os.path.lexists(INSTALL_RECEIPT):write_once(INSTALL_RECEIPT,dict(status='PARTIAL_OR_UNCERTAIN_INSTALLATION_NO_RETRY',installed=installed,
            error_type=type(error).__name__,error=str(error),ready=False,submitted=False))
        raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--plan-record',required=True);p.add_argument('--review-record',required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--verify-only',action='store_true');mode.add_argument('--install',action='store_true');a=p.parse_args()
    plan=json.loads(Path(a.plan_record).read_text());review=json.loads(Path(a.review_record).read_text())
    result=install(a.archive,plan,review) if a.install else installation_plan(a.archive,plan,review)[0]
    print(json.dumps(result,indent=2,allow_nan=False))
