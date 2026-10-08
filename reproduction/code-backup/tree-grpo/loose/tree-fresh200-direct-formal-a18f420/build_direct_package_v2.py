"""Materialize direct200 V2 with historical base-dev reuse; no tests/submit.

V1 remains retained as an obsolete, undeployed packet. This is a new immutable
artifact, not a second execution of the V1 writer.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
V1=HERE/'package-code-only'
PACKAGE=HERE/'package-code-only-v2'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PARENT=ROOT/'instruct300'
REMOTE_PACKAGE=PARENT/'candidates/fresh200-direct-train-a18f420'
AUTHORITY_NAME='DIRECT_TRAIN200_HUMAN_AUTHORITY-a18f420-V2.json'
RELEASE_NAME='DIRECT_TRAIN200_RELEASE-a18f420-V2.json'
DISPATCH_NAME='DIRECT_TRAIN200_DISPATCH-a18f420-V2.sbatch'
BASELINE_LOCAL=HERE.parent/'tree-fresh200-freeze-53082e-v12/actual-zero-dispatch/native3317/output/validation-cost/validation-1377126-1.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-proof-sha')
    args=parser.parse_args()
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old=json.loads((V1/'manifest.json').read_text())
    PACKAGE.mkdir()
    # Copy only the V1 selected code/metadata members, with the completed monitor
    # compatibility edit. Unselected import caches never enter the packet.
    for relative in old['files']:
        source=V1/relative;destination=PACKAGE/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write(source.read_bytes())
    report=json.loads(BASELINE_LOCAL.read_text());baseline_sha=sha(BASELINE_LOCAL)
    baseline_relative='historical-inputs/initial-dev3317.json'
    (PACKAGE/'historical-inputs').mkdir()
    with (PACKAGE/baseline_relative).open('xb') as stream:stream.write(BASELINE_LOCAL.read_bytes())
    baseline=dict(kind='HISTORICAL_OFFICIAL_FRESH_BASE_DEV512',job_id='3317',
        source_code_commit=report['code_commit'],model_identity=report['model_identity'],data_identity=report['data_identity'],
        actual_completed_outer=0,optimizer_writes_before_baseline=0,evaluated_rows=report['evaluated_rows'],
        per_dataset=dict(nq=256,hotpotqa=256),metrics={key:report['metrics'][key] for key in ('val/f1/all','val/em/all')},
        source_report=dict(path=str(ROOT/'runs/tree-fresh200-3317/output/validation-cost/validation-1377126-1.json'),sha256=baseline_sha),
        package_report_relative_path=baseline_relative,current_run_evaluation_executed=False,
        independent_acceptance=False,scope='Reuse an already measured official fresh-base score on the same fixed dev512; original job/source/config remain historical')
    authority=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_HUMAN_AUTHORITY-a18f420.json').read_text()))
    authority.update(packet_revision=2,trainer_val_before_train=False,startup_initial_validation_reused=True,
        historical_initial_validation=baseline,speed_change_observed_at_utc=observed,
        speed_change_authority='Latest explicit human instruction to start training and omit repeated validation; root directs prospective val_before_train=false')
    authority_path=HERE/AUTHORITY_NAME;write_json(authority_path,authority)
    manifest=copy.deepcopy(old)
    manifest['configuration_overrides']=[item for item in old['configuration_overrides'] if not item.startswith('trainer.val_before_train=')]
    manifest['configuration_overrides'].append('trainer.val_before_train=false')
    manifest.update(packet_revision=2,trainer_val_before_train=False,startup_initial_validation_reused=True,
        historical_initial_validation=baseline,human_training_authority=dict(path=str(PARENT/AUTHORITY_NAME),sha256=sha(authority_path)),
        materialized_at_utc=observed,supersedes_undeployed_packet_manifest_sha256=sha(V1/'manifest.json'))
    manifest['files']={str(path.relative_to(PACKAGE)):sha(path) for path in sorted(PACKAGE.rglob('*')) if path.is_file()}
    manifest['wrapper_files']={name:value for name,value in manifest['files'].items() if not name.startswith('source/')}
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest)
    manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_RELEASE-a18f420.json').read_text()))
    release.update(packet_revision=2,manifest_sha256=manifest_sha,human_training_authority_sha256=sha(authority_path),
        dispatch_renderer_sha256=manifest['files']['render_dispatch.py'],dispatch_submitter_sha256=manifest['files']['submit_once.py'],
        trainer_val_before_train=False,startup_initial_validation_reused=True)
    release_path=HERE/RELEASE_NAME;write_json(release_path,release)
    sys.path.insert(0,str(PACKAGE))
    from render_dispatch import render
    rendered=render(REMOTE_PACKAGE,manifest_sha,PARENT/RELEASE_NAME,sha(release_path),168)
    dispatch_path=HERE/DISPATCH_NAME
    with dispatch_path.open('x') as stream:stream.write(rendered)
    argv=[str(ROOT/'env/bin/python'),str(REMOTE_PACKAGE/'submit_once.py'),
        '--package',str(REMOTE_PACKAGE),'--manifest-sha',manifest_sha,
        '--release-path',str(PARENT/RELEASE_NAME),'--release-sha',sha(release_path),
        '--script',str(PARENT/DISPATCH_NAME),'--script-sha',sha(dispatch_path),
        '--predecessor-proof',str(ROOT/'runs/tree-fresh200-3319/direct-training-predecessor-proof.json')]
    if args.predecessor_proof_sha:argv.extend(['--predecessor-proof-sha',args.predecessor_proof_sha])
    archive=HERE/'fresh200-direct-train-a18f420-v2-code-only.tar.gz'
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted([*manifest['files'],'manifest.json']):
            bundle.add(PACKAGE/relative,arcname='candidates/fresh200-direct-train-a18f420/'+relative,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_DIRECT200_V2_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=2,
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(REMOTE_PACKAGE),
        archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=manifest_sha,authority_path=str(authority_path),authority_sha256=sha(authority_path),
        release_path=str(release_path),release_sha256=sha(release_path),dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),
        once_submission_argv=argv if args.predecessor_proof_sha else None,
        submission_argv_pending_predecessor_sha=argv if not args.predecessor_proof_sha else None,
        required_remaining_argv_option=None if args.predecessor_proof_sha else '--predecessor-proof-sha',
        once_intent=manifest['dispatch']['once_intent'],trainer_val_before_train=False,
        historical_initial_validation=baseline,new_tests_executed=False,new_reviews_executed=False,
        actual_slurm_submitted=False,contains_model_data_index_checkpoint_assets=False,
        retained_v1_archive_is_obsolete_and_undeployed=True)
    write_json(HERE/'DIRECT_PACKET_BOUND_ARGV_V2.json',receipt)
    write_json(HERE/'DIRECT_PACKET_V1_OBSOLETE.json',dict(status='OBSOLETE_UNDEPLOYED_V1_RETAINED',
        old_manifest_sha256=sha(V1/'manifest.json'),superseding_manifest_sha256=manifest_sha,
        superseding_packet='DIRECT_PACKET_BOUND_ARGV_V2.json',reason='Explicit prospective startup val_before_train=false with historical official-base dev512 reuse'))
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
