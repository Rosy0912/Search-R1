"""Prepare v7 external initialization factory after exact runtime/failure closure; local only."""
import argparse
import ast
import copy
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

from deployment_contract_v7 import (ROOT, SOURCE, HELPERS, NATIVE_HELPERS, CONTROLS,
    MANIFEST, CHECKPOINT, STARTUP_SCOPE, STARTUP_PROOF_ROLES, FACTORY_SCOPE, FACTORY_CHANGED, FACTORY_PROOF_ROLES, require, sha, write_once, validate_candidate)

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parents[2]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
RUNTIME = HERE.with_name('safety-calibration-v7')
OLD_RUNTIME = HERE.with_name('safety-calibration-v6')
NATIVE_SHA = '7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
PREDECESSOR_SHA = 'f008ac97b89f3ca7dd1d4801de8c401a5fa1f75571d5e1b873ff3823212ab561'
BASE_RUNTIME_MANIFEST_SHA = '0f57728daf2fe50fb74681787b320fa16c8086f03bd7eadb30b3bbe8b4d82cbd'
FINAL_RUNTIME_MANIFEST_NAME = 'CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json'
CONTROL_NAMES = {'deployment_contract_v7.py','freeze_ready_v7.py','prepare_candidate_v7.py','submit_once_v7.py','verify_physical_v7.py'}
SHA_RE = re.compile(r'[0-9a-f]{64}')


def preview_config(candidate, config_directory=None):
    # Same reviewed configuration bytes/semantics; only the controller version changes.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    q = candidate['quality_selection_control']
    env = dict(GIGPO_E1_ROOT=str(ROOT), GIGPO_E1_RUN=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'),
        AGENT_ROOT=str(ROOT), SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root',
        SEARCH_R1_BASELINE_OUTPUT=str(ROOT / 'runs/__ACTUAL_ISOLATED_RUN__'), GIGPO_QUALITY_GATE='30',
        GIGPO_QUALITY_CHECKPOINT=CHECKPOINT, GIGPO_QUALITY_MANIFEST_SHA256=MANIFEST,
        GIGPO_QUALITY_SELECTION=q['selection_path'], GIGPO_QUALITY_SELECTION_SHA256=q['selection_sha256'],
        GIGPO_QUALITY_REVIEW=q['selection_review_path'], GIGPO_QUALITY_REVIEW_SHA256=q['selection_review_sha256'])
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        directory = config_directory or LOCAL / 'work/gigpo-e1-quality-resume/verl/trainer/config'
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            config = compose(config_name='ppo_gigpo_instruct300_quality10_safe')
        config.actor_rollout_ref.actor.policy_safety.mode = 'calibration'
        config.trainer.experiment_name = 'gigpo-e1-policy-safety-calibration25'
        OmegaConf.resolve(config)
        return OmegaConf.to_container(config, resolve=True)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_archive(path, sources):
    with Path(path).open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name, source in sorted(sources.items()):
                payload = Path(source).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid, info.mode = len(payload), 0, 0, 0, 0o644
                archive.addfile(info, io.BytesIO(payload))


def bound_local(record):
    require(set(record) == {'path','bytes','sha256'}, 'Proof input must explicitly bind path/bytes/SHA')
    path=Path(record['path'])
    require(path.is_absolute() and path.resolve()==path and path.is_relative_to(OUT)
            and path.is_file() and not path.is_symlink(), 'Not a dedicated local original proof')
    require(type(record['bytes']) is int and record['bytes']>=0 and SHA_RE.fullmatch(record['sha256'])
            and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'Missing/changed proof original')
    return path


def validate_runtime_inputs(proofs):
    paths={role:bound_local(proofs[role]) for role in FACTORY_PROOF_ROLES}
    require(paths['runtime_manifest'].name == FINAL_RUNTIME_MANIFEST_NAME,
            'Only exact new v7 factory runtime manifest may gate this draft')
    manifest=json.loads(paths['runtime_manifest'].read_text())
    require(manifest.get('source_commit',manifest.get('code_commit'))==SOURCE
            and manifest['runtime_changed_filenames']==FACTORY_CHANGED
            and manifest['factory_change_scope']==FACTORY_SCOPE
            and manifest['base_v6_manifest_sha256']==BASE_RUNTIME_MANIFEST_SHA,
            'Not exact external initialization factory scope/base')
    base_path=OUT/'CALIBRATION-RUNTIME-IMPLEMENTATION-v6-v3.json'
    require(sha(base_path)==BASE_RUNTIME_MANIFEST_SHA,'Preserved v6-v3 runtime manifest changed')
    originals={Path(record['path']).name:record for record in json.loads(base_path.read_text())['files']}
    records=manifest['files'];new={Path(record['path']).name:record for record in records}
    require(len(originals)==13 and len(records)==len(new)==14
            and set(new)==set(originals)|{'metadata_colocation_v1.py'},'Not exact original thirteen plus one external metadata factory')
    changed=[]
    for name,record in new.items():
        path=RUNTIME/name
        require(Path(record['path'])==path and path.is_file() and not path.is_symlink()
                and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'], 'Runtime manifest differs from actual v7 bytes')
        if name not in originals:
            changed.append(name);continue
        old=OLD_RUNTIME/name
        require(old.stat().st_size==originals[name]['bytes'] and sha(old)==originals[name]['sha256'], 'Preserved v6 runtime changed')
        if path.read_bytes()!=old.read_bytes():changed.append(name)
    require(sorted(changed)==FACTORY_CHANGED and new['calibration_run_v1.py']['sha256']==
            '7aa80984aee6a5589ee44f9b72c4bed0d091e30c38e9c181fe2cc2728d1172c9',
            'Existing probability/worker/wrapper/guard/math bytes may not change')
    validate_runtime_review(proofs,json.loads(paths['runtime_review'].read_text()),sorted(changed))
    cpu=json.loads(paths['runtime_cpu'].read_text())
    require(cpu['status']=='PASS' and cpu['scope']=='ACTUAL_SOURCE_RAY_CPU_CONSTRUCTOR_AND_DISPATCH_ONLY'
            and cpu['node']=='h100-8' and cpu['source_commit']==SOURCE
            and cpu['candidate_sha256']==new['metadata_colocation_v1.py']['sha256']
            and cpu['torch_version']=='2.4.0+cu121' and cpu['ray_version']=='2.40.0'
            and cpu['torch_cuda_initialized'] is False and cpu['torch_process_group_initialized'] is False
            and cpu['ray_cluster_created'] is False and cpu['actual_8gpu_initialization_accepted'] is False
            and cpu['actual_restore'] is False and cpu['actual_sampling'] is False and cpu['actual_adam_writes']==0
            and all(cpu[key] is True for key in ('constructor_config_identity','environment_restored_success_and_inner_exception',
                'nested_unknown_factory_rejected','original_inner_disable_preserved','original_role_dispatch_attributes',
                'original_same_immediate_base_constraint_preserved','outer_worker_metadata','positional_and_keyword_arguments_preserved',
                'same_payload_identity','scoped_patch_restored_success_and_exception')),
            'Exact frozen-source CPU constructor/dispatch proof missing; cannot imply actual8GPU initialization')
    return manifest,paths


def validate_runtime_review(proofs,review,changed):
    require(review['status']=='PASS' and review.get('code_commit',review.get('source_commit'))==SOURCE
            and review['runtime_manifest_sha256']==proofs['runtime_manifest']['sha256']
            and review['runtime_diff_sha256']==proofs['runtime_diff']['sha256']
            and review['runtime_cpu_sha256']==proofs['runtime_cpu']['sha256'], 'Independent runtime review not exact manifest/diff/CPU')
    require(review['factory_change_scope']==FACTORY_SCOPE and review['runtime_changed_filenames']==changed
            and review['gpu_used'] is False and review['source_modified'] is False
            and review['production_thresholds_selected'] is False, 'Runtime static review exceeds external initialization scope')

def cpu_success_count(text):
    """Accept explicit successful pytest or unittest terminals, never a failure summary."""
    if re.search(r'\b(FAILED|FAILURES|ERRORS|ERROR|failed|error)\b',text):return 0
    pytest=re.search(r'(\d+) passed(?:[, ].*)?\s*$',text)
    unittest=re.search(r'Ran (\d+) tests? in [^\n]+\n\s*OK\s*$',text)
    match=pytest or unittest
    return int(match.group(1)) if match is not None else 0


def validate_failure_inputs(proofs, paths):
    payload=json.loads(paths['failure_payload'].read_text());review=json.loads(paths['failure_review'].read_text())
    diagnosis=json.loads(paths['failure_mro_diagnostic'].read_text())
    require(review['status']=='FAILED_CALIBRATION_STARTUP_NOT_ACCEPTED' and review['job_id']=='3276'
            and review['code_commit']==SOURCE and review['restore_manifest_sha256']==MANIFEST
            and all(type(review[key]) is int and review[key]==0 for key in
                ('actual_adam_writes','actual_gradient_micro_batches','actual_new_outer_updates','retained_training_updates_contributed'))
            and all(review[key] is False for key in ('actual_restore25_completed','actual_baseline_evaluation',
                'actual_sampling','probability_material_available','approval_generated')),
            'Not actual preserved failed3276 pre-restore/no-update evidence')
    require(review['consumed_payload']['path']==proofs['failure_payload']['path']
            and review['consumed_payload']['sha256']==proofs['failure_payload']['sha256']
            and review['diagnosis']['independent_diagnosis_path']==proofs['failure_mro_diagnostic']['path']
            and review['diagnosis']['independent_diagnosis_sha256']==proofs['failure_mro_diagnostic']['sha256']
            and review['diagnosis']['math_gradient_or_optimizer_failure'] is False,
            'Actual failed3276 independent raw/MRO diagnosis identity does not close')
    require(diagnosis['status']=='CONFIRMED_STARTUP_FACTORY_INCOMPATIBILITY' and diagnosis['job_id']=='3276'
            and diagnosis['code_commit']==SOURCE and diagnosis['binding_sha256']==review['binding_sha256']
            and diagnosis['scope']=='STATIC_SOURCE_AND_CPU_AST_FACTORY_REPRODUCTION_ONLY'
            and diagnosis['gpu_used'] is False and diagnosis['source_modified'] is False and diagnosis['submitted'] is False,
            'Not exact original MRO diagnosis scope')
    require(payload['job_id']=='3276' and payload['node']=='h100-8' and payload['run']==review['run'],
            'Failure raw job/node/run identity differs')
    resource=review['resource']
    require(resource['saved_cleanup_residual_states']=={} and resource['saved_final_compute']==[]
            and resource['postterminal_queue_empty'] is True and resource['postterminal_compute_empty'] is True
            and resource['cgroup_job_and_step_ENOENT_with_readable_slurm_ancestor'] is True,
            'Actual failed3276 exclusive allocation release was not accepted')
    originals=[];by={}
    for record in payload['originals']:
        raw=record['text'].encode('utf-8');path=Path(record['path'])
        require(path.is_absolute() and '..' not in path.parts
                and (path.parent==ROOT/'logs' or path.is_relative_to(payload['run']))
                and type(record['bytes']) is int and len(raw)==record['bytes']
                and hashlib.sha256(raw).hexdigest()==record['sha256'],'Failure original encoded bytes/SHA corrupt')
        item={key:record[key] for key in ('path','bytes','sha256')};originals.append(item);by[item['path']]=record
    require(len(by)==len(originals)==11 and review['consumed_originals']==originals,'Original failed3276 inventory differs')
    terminal=json.loads(by[review['terminal_path']]['text']);result=json.loads(by[review['result_path']]['text'])
    require(by[review['terminal_path']]['sha256']==review['terminal_sha256']
            and terminal['fields']['JobId']=='3276' and terminal['fields']['NodeList']=='h100-8'
            and terminal['fields']['JobState']=='FAILED' and terminal['fields']['ExitCode']=='1:0'
            and terminal['fields']['Requeue']==terminal['fields']['Restarts']=='0','Not original single failed3276 terminal')
    require(by[review['result_path']]['sha256']==review['result_sha256'] and result['status']=='FAILED'
            and result['job_id']=='3276' and result['source_commit']==SOURCE and result['binding_sha256']==review['binding_sha256']
            and result['restore_checkpoint']==CHECKPOINT and result['restore_manifest_sha256']==MANIFEST
            and result['retained_training_updates_contributed']==0
            and result['cleanup_residual_states']=={} and result['final_gpu']['processes']==[],
            'Original failed3276 result/source/restore/resource identity differs')
    log=by[str(Path(payload['run'])/'calibration.log')]['text']
    require("ActorRolloutRefWorker.__init__() missing 2 required positional arguments: 'config' and 'role'" in log
            and 'create_trainer' in log and 'trainer.init_workers()' in log,'Actual original constructor failure trace missing')
    return originals

def seal_installer(candidate, sources, archive):
    """Seal only our new local unsealed installer, after immutable draft/package exist."""
    path=HERE.with_name('safety-calibration-install-v7')/'install_calibration_v7.py'
    source=path.read_text()
    manifest_record=candidate['factory_control_proofs']['runtime_manifest']
    constants=dict(PACKAGE_SHA=sha(archive),CANDIDATE_SHA=sha(OUT/'CALIBRATION-CANDIDATE-52BEFE9-v7.json'),
        IMPLEMENTATION_SHA=manifest_record['sha256'],RUNTIME_MANIFEST_NAME=Path(manifest_record['path']).name,
        EVIDENCE_NAMES={name.split('/',1)[1] for name in sources if name.startswith('evidence/')},
        EVIDENCE_COUNT=len(candidate['evidence']),REQUIRED_EXISTING_COUNT=len(candidate['evidence'])-
            len([name for name in sources if name.startswith('evidence/')]))
    for name,value in constants.items():
        marker=name+' = None';require(source.count(marker)==1,'Installer was already sealed or code changed: '+name)
        source=source.replace(marker,name+' = '+repr(value),1)
    ast.parse(source)
    path.write_text(source)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),status='LOCAL_SEALED_NOT_DEPLOYED')


def main(proof_inputs_path, proof_inputs_sha):
    target=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v7.json';archive=HERE/'policy-safety-calibration-52befe9-v7-package.tar.gz'
    receipt_path=OUT/'CALIBRATION-PACKAGE-v7.json'
    require(not any(path.exists() for path in (target,archive,receipt_path)), 'Draft already/partly exists; inspect and preserve, never rebuild')
    spec_path=Path(proof_inputs_path)
    require(spec_path.is_absolute() and spec_path.resolve()==spec_path and spec_path.is_relative_to(OUT)
            and spec_path.is_file() and not spec_path.is_symlink()
            and sha(spec_path)==proof_inputs_sha,'Explicit dedicated proof input SHA/path changed')
    spec=json.loads(spec_path.read_text());proofs=spec['proofs']
    require(spec['schema']==1 and spec['code_commit']==SOURCE and spec['runtime_version']=='v7'
            and set(proofs)==set(FACTORY_PROOF_ROLES),'Wrong/missing final proof role specification')
    require(len({record['path'] for record in proofs.values()})==len(proofs),'Duplicate explicit proof role original')
    runtime,proof_paths=validate_runtime_inputs(proofs)
    failure_originals=validate_failure_inputs(proofs,proof_paths)
    native_path=OUT/'NATIVE-CANDIDATE-52BEFE9-v2.json';predecessor_path=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v6.json'
    require(sha(native_path)==NATIVE_SHA and sha(predecessor_path)==PREDECESSOR_SHA,'Frozen predecessor/native candidate changed')
    native=json.loads(native_path.read_text());predecessor=json.loads(predecessor_path.read_text())
    require(native['code_commit']==SOURCE and native['helpers']==str(NATIVE_HELPERS) and len(native['files'])==42327,
            'Wrong inherited native source/assets closure')
    candidate=copy.deepcopy(predecessor)
    candidate.update(helpers=str(HELPERS),controls=str(CONTROLS),source_native_acceptance=None,
        native_pending_job_acceptance=None,predecessor_calibration_candidate_sha256=PREDECESSOR_SHA,
        preparation_scope='External calibration initialization factory only; original v6-v3 startup and probability bytes preserved; actual native/physical/READY separate',
        startup_change_scope=STARTUP_SCOPE,startup_runtime_changed_filenames=['calibration_run_v1.py'],
        factory_change_scope=FACTORY_SCOPE,runtime_changed_filenames=FACTORY_CHANGED,
        factory_base_runtime_manifest_sha256=BASE_RUNTIME_MANIFEST_SHA,
        unchanged_runtime_wrapper_sha256='7aa80984aee6a5589ee44f9b72c4bed0d091e30c38e9c181fe2cc2728d1172c9',
        native_directory_modified=False)
    old={record['path']:record for record in predecessor['files']};files={}
    for record in native['files']:
        observed=old[record['path']]
        require(observed['sha256']==record['sha256'] and type(observed['bytes']) is int,'Inherited native path/SHA/bytes differ')
        files[record['path']]=copy.deepcopy(observed)
    sources={};runtime_records=[]
    for record in runtime['files']:
        local=Path(record['path']);remote=str(HELPERS/local.name)
        require(remote not in files,'Runtime targets inherited native path')
        entry=dict(path=remote,bytes=record['bytes'],sha256=record['sha256']);files[remote]=entry;runtime_records.append(entry)
        sources['runtime/'+local.name]=local
    require({path.name for path in HERE.glob('*.py') if not path.name.startswith('test_')}==CONTROL_NAMES,'Not exact five versioned controllers')
    for name in sorted(CONTROL_NAMES):
        local=HERE/name;remote=str(CONTROLS/name);files[remote]=dict(path=remote,bytes=local.stat().st_size,sha256=sha(local));sources['control/'+name]=local
    require(len(files)==42346,'Full file count differs from native42327 plus14+5')
    candidate['files']=list(files.values());candidate['runtime_helpers']=runtime_records
    native_evidence={record['path'] for record in native['evidence']}
    previous_failure_originals={record['path'] for record in predecessor.get('failure3272_originals',[])}
    for record in predecessor['evidence']:
        if record['path'] not in native_evidence|previous_failure_originals:
            local=OUT/Path(record['path']).name
            require(local.stat().st_size==record['bytes'] and sha(local)==record['sha256'],'Original calibration evidence changed')
            sources['evidence/'+local.name]=local
    evidence={record['path']:record for record in candidate['evidence']}
    def add_local(local):
        local=Path(local);record=dict(path=str(ROOT/'logs'/local.name),bytes=local.stat().st_size,sha256=sha(local))
        require(record['path'] not in evidence,'New proof would overwrite an old evidence path')
        evidence[record['path']]=record;sources['evidence/'+local.name]=local
        return record
    candidate['factory_control_proofs']={role:add_local(path) for role,path in proof_paths.items()}
    candidate['factory_proof_input_spec']=add_local(spec_path)
    for role in ('actual_cpu_stdout','actual_cpu_stderr','cpu_probe_script'):
        record=runtime['proofs'][role];path=Path(record['path'])
        require(path.is_file() and not path.is_symlink() and path.resolve()==path
                and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],
                'Exact original factory CPU supplementary proof changed')
        add_local(path)
    for local in (OUT/'CALIBRATION-DEPLOYMENT-v7.md',OUT/'CALIBRATION-DEPLOYMENT-CPU-v7.log'):add_local(local)
    for record in failure_originals:
        require(record['path'] not in evidence or evidence[record['path']]==record,'Conflicting preserved failure original')
        evidence[record['path']]=record
    candidate['failure3276_originals']=failure_originals;candidate['evidence']=list(evidence.values())
    preview=json.loads((OUT/'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json').read_text())
    require(preview_config(candidate)==preview['resolved'],'New startup/runtime paths altered reviewed optimization/config')
    validate_candidate(candidate)
    write_once(target,candidate);sources['candidate/'+target.name]=target;sources['cpu/test_deployment_v7.py']=HERE/'test_deployment_v7.py'
    make_archive(archive,sources);installer=seal_installer(candidate,sources,archive)
    receipt=dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',candidate=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)),
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),installer=installer,
        proof_inputs=dict(path=str(spec_path),sha256=proof_inputs_sha),predecessor_candidate_sha256=PREDECESSOR_SHA,
        native_candidate_sha256=NATIVE_SHA,native_acceptance=None,submitted_job=None,retained_training_updates_contributed=0,
        runtime_helpers_path=str(HELPERS),controls_path=str(CONTROLS),startup_change_scope=STARTUP_SCOPE,
        factory_change_scope=FACTORY_SCOPE,runtime_changed_filenames=FACTORY_CHANGED,unchanged_runtime_files=12,
        inherited_native_42327_paths_sha_unchanged=True,native_directory_modified=False,actual_gpu_executed=False,
        file_count=len(candidate['files']),evidence_count=len(candidate['evidence']),package_members=sorted(sources))
    write_once(receipt_path,receipt)
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),installer_sha256=installer['sha256'],
        files=len(files),evidence=len(evidence),ready=False,submitted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-inputs',required=True);parser.add_argument('--proof-inputs-sha',required=True)
    args=parser.parse_args();main(args.proof_inputs,args.proof_inputs_sha)
