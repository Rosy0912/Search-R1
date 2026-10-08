"""Assemble V4 only after root supplies actual fixed core commit/file SHA pins.

No tests, reviews, GPU, remote calls or submission. V3 and its actual FAILED3322
evidence remain retained. Historical native proof identities stay historical.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

HERE=Path(__file__).resolve().parent
V3=HERE/'package-code-only-v3'
PACKAGE=HERE/'package-code-only-v4'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PARENT=ROOT/'instruct300'
BASE_SOURCE='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
AUTHORITY_NAME='DIRECT_TRAIN200_HUMAN_AUTHORITY-V4.json'
RELEASE_NAME='DIRECT_TRAIN200_RELEASE-V4.json'
DISPATCH_NAME='DIRECT_TRAIN200_DISPATCH-V4.sbatch'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def require(condition,message):
    if not condition:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source-commit','source-delta','source-delta-sha','predecessor-proof-sha','remote-candidate-name','once-intent-name'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}',args.source_commit) and args.source_commit!=BASE_SOURCE,
            'Root actual fixed core commit is required; draft cannot become a final package')
    delta_path=Path(args.source_delta)
    require(sha(delta_path)==args.source_delta_sha,'Root supplied source-delta metadata bytes changed')
    delta=json.loads(delta_path.read_text())
    require(delta['code_commit']==args.source_commit and type(delta['files']) is dict and delta['files'],
            'Root actual fixed core file descriptors are required before assembly')
    require(re.fullmatch('fresh200-direct-train-[A-Za-z0-9_-]+-v4',args.remote_candidate_name),
            'Explicit new V4 candidate namespace required')
    require(re.fullmatch('SUBMISSION_ONCE-fresh200-direct-[A-Za-z0-9_-]+-v4.json',args.once_intent_name),
            'Explicit new V4 once intent namespace required')
    require(re.fullmatch('[0-9a-f]{64}',args.predecessor_proof_sha),'Exact3322 native predecessor proof SHA required')
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old=json.loads((V3/'manifest.json').read_text())
    changed={}
    for relative,pin in delta['files'].items():
        name=Path(relative)
        require(not name.is_absolute() and '..' not in name.parts and str(name)==relative
                and name.suffix in ('.py','.yaml','.yml'), 'Root fix delta must contain ordinary code files only')
        source=Path(pin['path']);raw=source.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==pin['sha256'],'Root fixed core file bytes differ: '+relative)
        destination=PACKAGE/'source/repo'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(raw);changed[relative]=pin['sha256']
    contract=PACKAGE/'production_contract.py'
    text=contract.read_text();needle="SOURCE = '"+BASE_SOURCE+"'"
    require(text.count(needle)==1,'Draft source contract is not the expected a18 base')
    contract.write_text(text.replace(needle,"SOURCE = '"+args.source_commit+"'"))
    proofs={
        '3319':dict(path=str(ROOT/'runs/tree-fresh200-3319/direct-training-predecessor-proof.json'),
            sha256='a4b9656f02427090983405b91afe3757473103a91b727c09aa51384b2f876816'),
        '3321':dict(path=str(ROOT/'runs/tree-fresh200-train-3321/direct-training-predecessor-proof.json'),
            sha256='484b1f5485d63c4f082c6e520e3d17914fa12cb6fefd2457509b0e2b975d51ef'),
        '3322':dict(path=str(ROOT/'runs/tree-fresh200-train-3322/direct-training-predecessor-proof.json'),
            sha256=args.predecessor_proof_sha),
    }
    authority=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_HUMAN_AUTHORITY-a18f420-V3.json').read_text()))
    authority.update(packet_revision=4,code_commit=args.source_commit,core_source_bytes_changed=True,
        real_error_fix='Root supplied actual core fix after first actor entry FAULT_STOP in FAILED3322',
        root_fixed_core_files_sha256=changed,error_fix_observed_at_utc=observed)
    authority_path=HERE/AUTHORITY_NAME;write_json(authority_path,authority)
    manifest=copy.deepcopy(old)
    manifest.update(packet_revision=4,code_commit=args.source_commit,materialized_at_utc=observed,
        human_training_authority=dict(path=str(PARENT/AUTHORITY_NAME),sha256=sha(authority_path)),
        supersedes_failed_packet_manifest_sha256=sha(V3/'manifest.json'),
        predecessor_terminal_cleanup_proof=proofs['3322'],terminal_proof_closure=proofs,
        root_actual_error_fix=dict(code_commit=args.source_commit,base_source_commit=BASE_SOURCE,
            source_delta_metadata_sha256=args.source_delta_sha,changed_source_files_sha256=changed,
            new_tests_executed=False,new_reviews_executed=False),
        previous_failed_direct_runs=[old['previous_failed_direct_run'],dict(job_id=3322,status='FAILED',
            scheduler_exit_code='1:0',code_commit=BASE_SOURCE,native_proof=proofs['3322'],
            actual_optimizer_committed=None,scheduler_advanced=None,failed_unrecorded_partial_optimizer_writes_unknown=True,
            checkpoint_inheritance=False,scope='Original native null counters remain null; separately observed per-rank zero writes are not substituted for run-result fields')])
    manifest.pop('previous_failed_direct_run',None)
    manifest['dispatch']['once_intent']=str(PARENT/args.once_intent_name)
    files={str(path.relative_to(PACKAGE)):sha(path) for path in sorted(PACKAGE.rglob('*')) if path.is_file()}
    require(all('__pycache__' not in Path(name).parts and Path(name).suffix!='.pyc' for name in files),
            'Unselected local import caches must not enter the V4 archive')
    manifest['files']=files
    manifest['wrapper_files']={name:value for name,value in files.items() if not name.startswith('source/')}
    source_files={name.removeprefix('source/repo/'):value for name,value in files.items() if name.startswith('source/repo/')}
    manifest['source_selection']=dict(protocol='a18_code_base_plus_root_actual_error_fix_v4',base_source_commit=BASE_SOURCE,
        code_commit=args.source_commit,selected_files=source_files,selected_count=len(source_files),
        changed_source_files_sha256=changed,not_a_full_repository_asset_copy=True,
        contains_model_data_index_checkpoint_assets=False)
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest)
    manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_RELEASE-a18f420-V3.json').read_text()))
    release.update(packet_revision=4,code_commit=args.source_commit,manifest_sha256=manifest_sha,
        human_training_authority_sha256=sha(authority_path),dispatch_intent_path=manifest['dispatch']['once_intent'],
        dispatch_renderer_sha256=files['render_dispatch.py'],dispatch_submitter_sha256=files['submit_once.py'],
        predecessor_terminal_cleanup_proof=proofs['3322'],terminal_proof_closure=proofs,
        actual_error_fix='Root supplied fixed core bytes after native FAILED3322 first actor-entry cause')
    release_path=HERE/RELEASE_NAME;write_json(release_path,release)
    remote_package=PARENT/'candidates'/args.remote_candidate_name
    sys.path.insert(0,str(PACKAGE))
    from render_dispatch import render
    rendered=render(remote_package,manifest_sha,PARENT/RELEASE_NAME,sha(release_path),168)
    dispatch_path=HERE/DISPATCH_NAME
    with dispatch_path.open('x') as stream:stream.write(rendered)
    argv=[str(ROOT/'env/bin/python'),str(remote_package/'submit_once.py'),
        '--package',str(remote_package),'--manifest-sha',manifest_sha,
        '--release-path',str(PARENT/RELEASE_NAME),'--release-sha',sha(release_path),
        '--script',str(PARENT/DISPATCH_NAME),'--script-sha',sha(dispatch_path),
        '--predecessor-proof',proofs['3322']['path'],'--predecessor-proof-sha',args.predecessor_proof_sha]
    archive=HERE/(args.remote_candidate_name+'-code-only.tar.gz')
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted([*files,'manifest.json']):
            bundle.add(PACKAGE/relative,arcname='candidates/'+args.remote_candidate_name+'/'+relative,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_DIRECT200_V4_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=4,code_commit=args.source_commit,
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(remote_package),
        archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=manifest_sha,authority_path=str(authority_path),authority_sha256=sha(authority_path),
        release_path=str(release_path),release_sha256=sha(release_path),dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),
        once_submission_argv=argv,once_intent=manifest['dispatch']['once_intent'],terminal_proof_closure=proofs,
        changed_source_files_sha256=changed,new_tests_executed=False,new_reviews_executed=False,
        actual_slurm_submitted=False,contains_model_data_index_checkpoint_assets=False,
        retained_v3_failed_run_evidence=True,postload_held_ofd_identity_and_foreign15_percent_guards_preserved=True)
    write_json(HERE/'DIRECT_PACKET_BOUND_ARGV_V4.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
