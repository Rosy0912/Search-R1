"""Materialize immutable direct200 V3 after real FAILED3321; never test/submit."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
V2=HERE/'package-code-only-v2'
PACKAGE=HERE/'package-code-only-v3'
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
PARENT=ROOT/'instruct300'
REMOTE_PACKAGE=PARENT/'candidates/fresh200-direct-train-a18f420-v3'
AUTHORITY_NAME='DIRECT_TRAIN200_HUMAN_AUTHORITY-a18f420-V3.json'
RELEASE_NAME='DIRECT_TRAIN200_RELEASE-a18f420-V3.json'
DISPATCH_NAME='DIRECT_TRAIN200_DISPATCH-a18f420-V3.sbatch'
PROOF_PATH=ROOT/'runs/tree-fresh200-train-3321/direct-training-predecessor-proof.json'
PROOF_SHA='484b1f5485d63c4f082c6e520e3d17914fa12cb6fefd2457509b0e2b975d51ef'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def main():
    observed=datetime.datetime.now(datetime.timezone.utc).isoformat()
    old=json.loads((V2/'manifest.json').read_text())
    authority=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_HUMAN_AUTHORITY-a18f420-V2.json').read_text()))
    authority.update(packet_revision=3,real_error_fix='FAILED3321 post-load whole-card idle gate replaced with held lease identity plus foreign15% wrapper',
        initial_whole_card_idle_admission_preserved=True,postload_guard_scope='Exact owned allocation/cgroup/OFD UUID+FD identity; original realtime foreignMEM/SM<=15%',
        core_source_bytes_changed=False,error_fix_observed_at_utc=observed)
    authority_path=HERE/AUTHORITY_NAME;write_json(authority_path,authority)
    manifest=copy.deepcopy(old)
    manifest.update(packet_revision=3,materialized_at_utc=observed,
        human_training_authority=dict(path=str(PARENT/AUTHORITY_NAME),sha256=sha(authority_path)),
        supersedes_failed_packet_manifest_sha256=sha(V2/'manifest.json'),
        predecessor_terminal_cleanup_proof=dict(path=str(PROOF_PATH),sha256=PROOF_SHA),
        previous_failed_direct_run=dict(job_id=3321,status='FAILED',scheduler_exit_code='1:0',
            run_result_sha256='052dd4d171c523d3258856ee48494c57919d2f073cc8cff12833eec4757af00a',
            cost_sha256='438435c67fa523367effd16c07e47c09e36fdabf97195488246c8ebe25ffb714',
            once_journal_sha256='086626300a58fc0bb3800d7043a3a3d9be90a75d97fdbe1bdcf76db2b2c5a35d',
            actual_optimizer_committed=None,scheduler_advanced=None,failed_unrecorded_partial_optimizer_writes_unknown=True,
            checkpoint_inheritance=False,scope='Retained real FAILED3321 evidence; missing partial writes stay unknown, never become completed credits'),
        postload_lease_wrapper=dict(mode_environment='TREE_DIRECT_OWNED_MODELS_LOADED',
            mode_absent_during_initial_preflight=True,initial_whole_card_idle_admission_preserved=True,
            identity_checked='own actual allocation/cgroup/exact8 physicalUUID/FDinode/heldOFD',
            resource_guard_checked='Realtime foreignMEMfraction<=0.15 and foreignSMpercent<=15',
            own_loaded_memory_and_sm_are_not_idle_criteria=True,core_preflight_source_bytes_unchanged=True))
    manifest['dispatch']['once_intent']=str(PARENT/'SUBMISSION_ONCE-fresh200-direct-a18f420-v3.json')
    members=[*old['files'],'owned_lease_identity.py']
    manifest['files']={relative:sha(PACKAGE/relative) for relative in sorted(members)}
    manifest['wrapper_files']={name:value for name,value in manifest['files'].items() if not name.startswith('source/')}
    manifest_path=PACKAGE/'manifest.json';write_json(manifest_path,manifest)
    manifest_sha=sha(manifest_path)
    release=copy.deepcopy(json.loads((HERE/'DIRECT_TRAIN200_RELEASE-a18f420-V2.json').read_text()))
    release.update(packet_revision=3,manifest_sha256=manifest_sha,human_training_authority_sha256=sha(authority_path),
        dispatch_intent_path=manifest['dispatch']['once_intent'],
        dispatch_renderer_sha256=manifest['files']['render_dispatch.py'],dispatch_submitter_sha256=manifest['files']['submit_once.py'],
        predecessor_terminal_cleanup_proof=dict(path=str(PROOF_PATH),sha256=PROOF_SHA),
        actual_error_fix='Post-load wrapper identity/foreign15% checks replace own-loaded whole-card idle criteria')
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
        '--predecessor-proof',str(PROOF_PATH),'--predecessor-proof-sha',PROOF_SHA]
    archive=HERE/'fresh200-direct-train-a18f420-v3-code-only.tar.gz'
    with tarfile.open(archive,'w:gz',format=tarfile.PAX_FORMAT) as bundle:
        for relative in sorted([*manifest['files'],'manifest.json']):
            bundle.add(PACKAGE/relative,arcname='candidates/fresh200-direct-train-a18f420-v3/'+relative,recursive=False)
        for path in (authority_path,release_path,dispatch_path):bundle.add(path,arcname=path.name,recursive=False)
    receipt=dict(status='CODE_ONLY_DIRECT200_V3_PACKET_MATERIALIZED_NOT_SUBMITTED',packet_revision=3,
        local_package=str(PACKAGE),remote_extract_destination=str(PARENT),remote_package=str(REMOTE_PACKAGE),
        archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=manifest_sha,authority_path=str(authority_path),authority_sha256=sha(authority_path),
        release_path=str(release_path),release_sha256=sha(release_path),dispatch_path=str(dispatch_path),dispatch_sha256=sha(dispatch_path),
        once_submission_argv=argv,once_intent=manifest['dispatch']['once_intent'],
        predecessor_terminal_cleanup_proof=dict(path=str(PROOF_PATH),sha256=PROOF_SHA),
        new_tests_executed=False,new_reviews_executed=False,actual_slurm_submitted=False,
        contains_model_data_index_checkpoint_assets=False,retained_v2_failed_run_evidence=True,
        initial_whole_card_idle_admission_preserved=True,core_source_bytes_changed=False,
        postload_held_ofd_identity_and_foreign15_percent_guards_preserved=True)
    write_json(HERE/'DIRECT_PACKET_BOUND_ARGV_V3.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
