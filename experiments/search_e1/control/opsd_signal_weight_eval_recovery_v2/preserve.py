"""Separate, published preservation action. No generation/optimizer/foreign signals."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from recovery_common import checked_json, checked, require, tree, install_original
from failure_gate import FAILURE_SHA, validate_observed_failure, prove_old_absent, fresh_resources, required_mutable_paths


def file_ref(path):
    path=Path(path).resolve();before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    after=path.stat();fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    require(all(getattr(before,k)==getattr(after,k) for k in fields),'Failure asset changed while hashing')
    return dict(path=str(path),sha256=digest.hexdigest(),bytes=before.st_size)


def exclusive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,sort_keys=True,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    return file_ref(path)


def archive_tree(source,destination,output,fields,metadata_refs=()):
    source=Path(source);destination=Path(destination);output=Path(output)
    require(not source.is_symlink() and not destination.is_symlink() and not output.is_symlink(),
            'Root symlinks in preservation paths refused')
    source=source.resolve();destination=destination.resolve();output=output.resolve()
    require(source.is_dir() and not source.is_symlink() and not destination.exists() and not output.exists(),
            'Only existing failed tree into a new independent destination; never overwrite')
    require(not any(p.is_symlink() for p in source.rglob('*')),'Symlink in failed evidence tree refused')
    files=[file_ref(p) for p in sorted(source.rglob('*')) if p.is_file()]
    require(files,'Failed evidence tree cannot be empty')
    destination.parent.mkdir(parents=True,exist_ok=True)
    metadata_root=destination.with_name(destination.name+'.metadata')
    require(not metadata_root.exists() and not metadata_root.is_symlink(),'Metadata destination must also be new')
    metadata_originals=[]
    for reference in metadata_refs:
        checked(reference);original=file_ref(reference['path'])
        require(original['sha256']==reference['sha256'],'Mutable metadata differs from pinned failure')
        metadata_originals.append(original)
    require(len({r['path'] for r in metadata_originals})==len(metadata_originals),'Repeated preservation metadata path')
    pending=output.with_name(output.name+'.before-rename.json')
    exclusive(pending,dict(original_root=str(source),preserved_root=str(destination),original_files=files,
                           metadata_originals=metadata_originals,rename_not_yet_complete=True,**fields))
    metadata_mapping=[]
    if metadata_originals:
        metadata_root.mkdir()
        for index,original in enumerate(metadata_originals):
            target=metadata_root/('%02d-'%index+Path(original['path']).name)
            raw=checked(original)
            with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            preserved=file_ref(target)
            require(preserved['sha256']==original['sha256'] and preserved['bytes']==original['bytes'],
                    'Failure mutable metadata copy changed')
            metadata_mapping.append(dict(original=original,preserved=preserved))
    os.rename(source,destination)
    mapping=[]
    for old in files:
        target=destination/Path(old['path']).relative_to(source);new=file_ref(target)
        require(new['sha256']==old['sha256'] and new['bytes']==old['bytes'],'Moved failed evidence changed')
        mapping.append(dict(original=old,preserved=new))
    return exclusive(output,dict(schema='opsd-signal-failed-evaluation-preservation-v2',accepted=True,
        formal_endpoint_accepted=False,original_root=str(source),preserved_root=str(destination),files=mapping,
        metadata_files=metadata_mapping,metadata_file_count=len(metadata_mapping),
        file_count=len(mapping),bytes=sum(r['original']['bytes'] for r in mapping),
        original_json_absolute_references_unchanged=True,
        old_absolute_references_must_resolve_with_this_mapping=True,
        original_worker_unknown_waits_not_promoted=True,no_performance_selection=True,
        no_checkpoint_training_or_source_mutation=True,**fields))


def run(plan_ref):
    plan=checked_json(plan_ref);root=Path(plan['helper_root']).resolve()
    require(plan['schema']=='opsd-signal-evaluation-preservation-plan-v2' and root==Path(__file__).resolve().parent
            and os.getuid()==1001 and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules,
            'Explicit published remote CPU preservation plan required')
    tree(root,plan['source_manifest']);pub=checked_json(plan['publication'])
    require(pub['accepted'] is True and pub['branch']=='search-e1'
            and pub['source_commit']==pub['remote_readback_sha']==plan['source_commit']
            and pub['source_manifest_sha256']==plan['source_manifest']['sha256'],'Preservation source not published')
    config=checked_json(plan['config']);ref=plan['config']
    require(ref['sha256']=='e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661'
            and plan['failure_metadata']['sha256']==FAILURE_SHA,'Only original G_POS/evaluator15 failure permitted')
    value=validate_observed_failure(checked_json(plan['failure_metadata']),config,ref)
    require({r['path'] for r in plan['mutable_metadata']}==required_mutable_paths(config),
            'Explicit original status/exits/study-status/CPU registry references required')
    for key in ('failed_status','failure_actual_exits'):
        require(any(r['path']==value[key]['reference']['path'] and r['sha256']==value[key]['reference']['sha256']
                    for r in plan['mutable_metadata']),'Pinned failed status/actual waits must retain original bytes')
    original=Path(config['paths']['output_root'])/'evaluations/outer-15'
    require(original==Path(value['original_request']['path']).parent.parent
            and not (original/'parent-complete.json').exists() and not (original/'child-complete.json').exists(),
            'Failure canonical tree contains a formal parent/child; do not archive accepted evaluation')
    study=Path(config['paths']['study_base']).resolve();destination=Path(plan['destination']).resolve();output=Path(plan['output']).resolve()
    require(destination.is_relative_to(study/'preserved') and output.is_relative_to(study/'receipts')
            and destination!=original and not destination.is_relative_to(original),'Preservation paths escaped own remote study')
    manifest=tree(config['paths']['helper_root'],config['helper_source_manifest']);install_original(Path(config['paths']['helper_root']),manifest)
    gates=importlib.import_module('gates');config=dict(config,_config_sha256=ref['sha256']);locks=[]
    paths=[str(study/'study-sequence.lock'),str(Path(config['paths']['base'])/'online-experiment.lock')]
    paths+=config['resources']['shared_gpu_lease_paths']+[config['resources']['study_lease_path']]
    try:
        for path in sorted(set(paths)):
            handle=Path(path).open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        prove_old_absent(value,gates);resource=fresh_resources(config,gates)
        checked(value['original_request']);checked(value['original_parent_log'])
        result=archive_tree(original,destination,output,dict(failure_metadata=plan['failure_metadata'],config=ref,
            resume_checkpoint=value['raw_checkpoint'],plan=plan_ref,source_manifest=plan['source_manifest'],publication=plan['publication'],
            actual_failed_evaluator_wait=1,actual_failed_controller_wait=1,actual_failed_supervisor_wait=1,
            old_all_actual_absent=True,current_only_oracle=True),metadata_refs=plan['mutable_metadata'])
        return result
    finally:
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();result=run(dict(path=args.plan,sha256=args.plan_sha256))
    print(json.dumps(dict(accepted=True,preservation=result,formal_endpoint_accepted=False,
                         large_products_remote_only=True,no_new_training_or_evaluation=True),sort_keys=True))


if __name__=='__main__':main()
