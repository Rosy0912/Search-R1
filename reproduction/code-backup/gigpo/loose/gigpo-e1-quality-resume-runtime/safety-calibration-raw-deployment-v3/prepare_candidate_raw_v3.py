"""Prepare an immutable unlaunched raw calibration candidate, never GPU READY."""
import argparse, copy, gzip, io, json, os, tarfile
from pathlib import Path
from deployment_contract_raw_v3 import *
HERE=Path(__file__).resolve().parent
LOCAL=HERE.parents[2]
OUT=LOCAL/'outputs/gigpo-e1-policy-safety'
RUNTIME=HERE.with_name('safety-calibration-raw-v2')


def record(path,remote=None):
    p=Path(path)
    require(p.resolve()==p and p.is_file() and not p.is_symlink(), 'Only canonical original files')
    return dict(path=str(remote or p),bytes=p.stat().st_size,sha256=sha(p))


def bound_local(r):
    require(set(r)=={'path','bytes','sha256'}, 'Explicit local original path/bytes/SHA required')
    p=Path(r['path'])
    require(p.is_absolute() and p.is_relative_to(OUT) and record(p)==r, 'Local original changed or foreign')
    return p


def preview_config(candidate,config_directory=None):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    q=candidate['quality_selection_control']
    env=dict(GIGPO_E1_ROOT=str(ROOT),GIGPO_E1_RUN=str(ROOT/'runs/__ACTUAL_ISOLATED_RUN__'),AGENT_ROOT=str(ROOT),
        SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root',SEARCH_R1_BASELINE_OUTPUT=str(ROOT/'runs/__ACTUAL_ISOLATED_RUN__'),
        GIGPO_QUALITY_GATE='30',GIGPO_QUALITY_CHECKPOINT=CHECKPOINT,GIGPO_QUALITY_MANIFEST_SHA256=MANIFEST,
        GIGPO_QUALITY_SELECTION=q['selection_path'],GIGPO_QUALITY_SELECTION_SHA256=q['selection_sha256'],
        GIGPO_QUALITY_REVIEW=q['selection_review_path'],GIGPO_QUALITY_REVIEW_SHA256=q['selection_review_sha256'])
    previous={k:os.environ.get(k) for k in env};os.environ.update(env)
    try:
        directory=config_directory or LOCAL/'work/gigpo-e1-raw-action-identity-20261006/verl/trainer/config'
        with initialize_config_dir(config_dir=str(directory),version_base=None):
            config=compose(config_name='ppo_gigpo_instruct300_quality10_safe_raw')
        config.actor_rollout_ref.actor.policy_safety.mode='calibration'
        config.trainer.experiment_name='gigpo-e1-policy-safety-raw-action-calibration25'
        OmegaConf.resolve(config)
        return OmegaConf.to_container(config,resolve=True)
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v


def validate_preview(config):
    require(config['actor_token_identity']=='sampled_action_prefix_v1'
            and all(config['actor_rollout_ref'][role]['actor_token_identity']=='sampled_action_prefix_v1'
                    for role in ('actor','ref','rollout')), 'Raw scorer/generation identity differs')
    safety=config['actor_rollout_ref']['actor']['policy_safety']
    require(safety['enabled'] is True and safety['mode']=='calibration' and safety['dual_clip']==3.
            and safety['kl_log_bound']==30. and safety['monitor_first_global_rows']==256
            and safety['target_old_policy_kl'] is None and safety['negative_tail_limit'] is None,
            'Probe cannot introduce production thresholds or alter safety formulas')
    require(config['actor_rollout_ref']['actor']['ppo_mini_batch_size']==256
            and config['actor_rollout_ref']['actor']['ppo_micro_batch_size']==16
            and config['actor_rollout_ref']['actor']['ppo_epochs']==1, 'Authorized quality10 trajectory budget differs')
    return True


def make_archive(path,sources):
    require(len(sources)==len(set(sources)), 'Duplicate package member')
    with Path(path).open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz:
        with tarfile.open(fileobj=gz,mode='w') as archive:
            for name,p in sorted(sources.items()):
                require(not Path(name).is_absolute() and '..' not in Path(name).parts and not Path(p).is_symlink(), 'Unsafe package member')
                data=Path(p).read_bytes();info=tarfile.TarInfo(name)
                info.size,info.mode,info.mtime,info.uid,info.gid=len(data),0o644,0,0,0
                archive.addfile(info,io.BytesIO(data))


def main(spec_path,spec_sha):
    require(sha(spec_path)==spec_sha, 'Explicit proof inputs changed')
    spec=json.loads(Path(spec_path).read_text())
    require(set(spec)=={'parent_candidate','parent_physical','proofs'}, 'Wrong proof input roles')
    pp=bound_local(spec['parent_candidate']);ph=bound_local(spec['parent_physical'])
    require(sha(pp)==PARENT_CANDIDATE_SHA and sha(ph)==PARENT_PHYSICAL_SHA, 'Not exact new-source native lineage')
    parent=json.loads(pp.read_text());physical=json.loads(ph.read_text())
    require(set(spec['proofs'])==set(PROOF_ROLES)-{'config_preview'}, 'Raw implementation/support proof roles missing')
    paths={role:bound_local(r) for role,r in spec['proofs'].items()}
    require(sha(paths['runtime_manifest'])==RUNTIME_MANIFEST_SHA and sha(paths['runtime_review'])==RUNTIME_REVIEW_SHA,
            'Raw runtime static acceptance changed')
    manifest=json.loads(paths['runtime_manifest'].read_text());review=json.loads(paths['runtime_review'].read_text())
    require(manifest['code_commit']==SOURCE and manifest['file_count']==18
            and review['status'].startswith('PASS') and review['runtime_manifest_sha256']==RUNTIME_MANIFEST_SHA
            and review['code_commit']==SOURCE and review['gpu_used'] is False, 'Not exact raw-v2 static scope')
    sources={};files=copy.deepcopy(parent['files']);by=closure(files,'inherited files');runtime=[]
    for name,identity in RUNTIME_IDENTITIES.items():
        p=RUNTIME/name;require(record(p)==dict(path=str(p),**identity), 'Frozen local runtime changed')
        r=dict(path=str(HELPERS/name),**identity);require(r['path'] not in by,'Cannot overwrite native helper')
        runtime.append(r);files.append(r);sources['runtime/'+name]=p
    for name in NAMES:
        p=HERE/name;r=record(p,CONTROLS/name);require(r['path'] not in by,'Cannot overwrite old control')
        files.append(r);sources['control/'+name]=p
    candidate=dict(status='CANDIDATE_NOT_READY',code_commit=SOURCE,repo=str(REPO),node='h100-8',helpers=str(HELPERS),controls=str(CONTROLS),
        validation_scope=ADMISSION_SCOPE,calibration_scope=SCOPE,actor_token_identity='sampled_action_prefix_v1',capture_actual_runner_inputs=True,
        allowed_safety_modes=['calibration'],calibration_phase='probability_probe_then_wait',allowed_calibration_phases=['probability_probe_then_wait'],
        restore_step=25,restore_checkpoint=CHECKPOINT,restore_manifest_sha256=MANIFEST,common_manifest_sha256=DATA,
        calibration_actual_outer_budget=1,optimizer_steps_per_outer=10,retained_training_updates_contributed=0,
        training_submitted=False,calibration_submitted=False,source_native_acceptance=None,actual_probability_acceptance=None,
        target_old_policy_kl=None,negative_tail_limit=None,actor_recompute_abs_tolerance=None,actor_vllm_abs_tolerance=None,
        actor_recompute_joint_abs_tolerance=None,actor_vllm_joint_abs_tolerance=None,probability_tolerance_evidence=[],production_thresholds_selected=False,
        prefix_mismatch_limit=0,action_token_mismatch_limit=0,healthy_baseline_f1_floor=.3231979640745351/2,probability_review_wait_seconds=43200,
        runtime_helpers=runtime,files=files,resource_guard=parent['resource_guard'],resource_ownership=parent['resource_ownership'],
        quality_selection_control=QUALITY_CONTROL,native_restore_scope='OFFICIAL_INITIALIZATION_ONLY_NOT_TRAINED25',
        calibration_complete_restore25_required=True,actual_runner_observation_not_yet_proved=True)
    evidence=copy.deepcopy(parent['evidence']);evidence_by=closure(evidence,'inherited evidence')
    def add(p,remote,pack=False):
        r=record(p,remote)
        require(r['path'] not in evidence_by or evidence_by[r['path']]==r,'Proof substitution forbidden')
        if r['path'] not in evidence_by:evidence.append(r);evidence_by[r['path']]=r
        if pack:sources['evidence/'+Path(remote).name]=p
        return r
    candidate['native_parent_candidate']=add(pp,ROOT/'logs/ACTION-PREFIX-NATIVE-CANDIDATE-2262942-v2.json')
    candidate['inherited_physical']=add(ph,ROOT/'logs/ACTION-PREFIX-NATIVE-PHYSICAL-2262942-v2.json')
    candidate['raw_runtime_proofs']={role:add(p,ROOT/'logs/policy-safety-calibration-raw-proof-v3'/p.name,True) for role,p in paths.items()}
    resolved=preview_config(candidate);validate_preview(resolved)
    preview_path=OUT/'CALIBRATION-RAW-RESOLVED-CONFIG-PREVIEW-v3.json'
    write_once(preview_path,dict(status='LOCAL_HYDRA_PREVIEW_NOT_TARGET_EXECUTION',source_commit=SOURCE,resolved=resolved,
                              actual_gpu=False,source_config='ppo_gigpo_instruct300_quality10_safe_raw'))
    preview=add(preview_path,ROOT/'logs/policy-safety-calibration-raw-proof-v3'/preview_path.name,True)
    candidate['expected_config_preview']=preview;candidate['raw_runtime_proofs']['config_preview']=preview;candidate['evidence']=evidence
    validate_candidate(candidate);validate_inheritance(candidate,parent,physical)
    target=OUT/'CALIBRATION-RAW-CANDIDATE-2262942-v3.json';archive=HERE/'policy-safety-calibration-raw-2262942-v3-package.tar.gz'
    require(not target.exists() and not archive.exists(), 'Existing draft/package cannot be overwritten')
    write_once(target,candidate);sources['candidate/'+target.name]=target;sources['cpu/test_deployment_raw_v3.py']=HERE/'test_deployment_raw_v3.py'
    require(sum(Path(p).stat().st_size for p in sources.values())<=64*1024*1024
            and all(Path(p).stat().st_size<=20*1024*1024 for p in sources.values()), 'Bounded code/metadata-only package size exceeded')
    make_archive(archive,sources)
    receipt=OUT/'CALIBRATION-RAW-PACKAGE-RECEIPT-v3.json'
    write_once(receipt,dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',candidate=record(target),archive=record(archive),
        proof_inputs=record(Path(spec_path)),package_members=sorted(sources),file_count=len(files),evidence_count=len(evidence),
        inherited_files_count=len(parent['files']),runtime_files_count=18,control_files_count=5,source_native_acceptance=None,
        current_native_gpu_accepted=False,probability_accepted=False,production_thresholds_selected=False,submitted_job=None,
        existing_target_inputs=[candidate['native_parent_candidate'],candidate['inherited_physical']],
        target_side_installation_requires_no_traversal_symlink_overwrite=True))
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),ready=False,submitted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proof-inputs',required=True);parser.add_argument('--proof-inputs-sha',required=True)
    args=parser.parse_args();main(args.proof_inputs,args.proof_inputs_sha)
