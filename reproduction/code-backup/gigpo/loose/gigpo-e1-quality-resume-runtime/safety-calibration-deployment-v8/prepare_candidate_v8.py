"""Prepare v8 external audit-row dtype compatibility after exact runtime/failure closure; local only."""
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

from deployment_contract_v8 import (ROOT, SOURCE, HELPERS, NATIVE_HELPERS, CONTROLS,
    MANIFEST, CHECKPOINT, STARTUP_SCOPE, STARTUP_PROOF_ROLES, FACTORY_SCOPE, FACTORY_CHANGED, FACTORY_PROOF_ROLES, DATAROW_SCOPE, DATAROW_CHANGED, DATAROW_PROOF_ROLES, DATAROW_BASE_MANIFEST_SHA, require, sha, write_once, validate_candidate)

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parents[2]
OUT = LOCAL / 'outputs/gigpo-e1-policy-safety'
RUNTIME = HERE.with_name('safety-calibration-v8')
OLD_RUNTIME = HERE.with_name('safety-calibration-v7')
NATIVE_SHA = '7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
PREDECESSOR_SHA = 'b8f0436a2d1b9668664d6a6c8d14ba8c066d9f818c1b2c2a97eb2a420472ba4e'
BASE_RUNTIME_MANIFEST_SHA = DATAROW_BASE_MANIFEST_SHA
FINAL_RUNTIME_MANIFEST_NAME = 'CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json'
CONTROL_NAMES = {'deployment_contract_v8.py','freeze_ready_v8.py','prepare_candidate_v8.py','submit_once_v8.py','verify_physical_v8.py'}
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


def validate_runtime_review(proofs, review, changed):
    require(review['status'] == 'PASS' and review.get('code_commit', review.get('source_commit')) == SOURCE
            and review['runtime_manifest_sha256'] == proofs['runtime_manifest']['sha256']
            and review['runtime_diff_sha256'] == proofs['runtime_diff']['sha256']
            and review['runtime_cpu_sha256'] == proofs['runtime_cpu']['sha256'], 'Datarow review does not bind exact manifest/diff/CPU')
    require(review['datarow_change_scope'] == DATAROW_SCOPE and review['runtime_changed_filenames'] == changed
            and review['gpu_used'] is False and review['source_modified'] is False
            and review['production_thresholds_selected'] is False, 'Datarow review exceeds metadata dtype scope')


def validate_target_cpu_inputs(proofs, paths, driver_sha):
    target = json.loads(paths['runtime_target_cpu'].read_text())
    raw = json.loads(paths['runtime_target_cpu_raw'].read_text())
    review = json.loads(paths['runtime_target_cpu_review'].read_text())
    require(raw['transport_returncode'] == 0 and raw['actual']['returncode'] == 0, 'Target CPU raw command failed')
    receipt = raw['actual']['records']['target_cpu']
    require(receipt['path'] == str(ROOT/'logs/CALIBRATION-DATAROW-TARGET-CPU-v8.json')
            and receipt['bytes'] == proofs['runtime_target_cpu']['bytes']
            and receipt['sha256'] == proofs['runtime_target_cpu']['sha256']
            and receipt['data'] == target and json.loads(raw['actual']['stdout']) == target,
            'Target CPU original path/bytes/SHA/data does not close')
    require(target['status'] == 'PASS' and target['scope'] == 'ACTUAL_DATAPROTO_SELECT_ROWS_AND_ONE_OUTER_CPU_ONLY'
            and target['runtime_driver_sha256'] == driver_sha and target['torch_version'] == '2.4.0+cu121'
            and target['tensordict_version'] == '0.5.0' and target['tests_run'] == 9
            and type(target['actual_Adam_writes']) is int and target['actual_Adam_writes'] == 0
            and target['errors'] == target['failures'] == 0 and target['actual_CUDA_initialized'] is False
            and target['Ray_cluster_created'] is False and target['actual_GPU_acceptance'] is False
            and target['actual_source_protocol'] is True and target['actual_full_one_outer_function_body'] is True
            and target['actual_protocol_path'] == str(ROOT/'repo-policy-safety-52befe9/verl/protocol.py')
            and target['actual_select_rows_path'] == str(ROOT/'repo-policy-safety-52befe9/search_r1/baselines/rollout.py'),
            'Not actual frozen-source target Torch2.4 metadata/one_outer CPU scope')
    require(review['status'] == 'PASS' and review['source_commit'] == SOURCE and review['node'] == 'h100-8'
            and review['runtime_manifest_sha256'] == proofs['runtime_manifest']['sha256']
            and review['runtime_driver_sha256'] == driver_sha
            and review['runtime_target_cpu_sha256'] == proofs['runtime_target_cpu']['sha256']
            and review['runtime_target_cpu_raw_sha256'] == proofs['runtime_target_cpu_raw']['sha256']
            and review['datarow_change_scope'] == DATAROW_SCOPE and review['runtime_changed_filenames'] == DATAROW_CHANGED
            and review['gpu_used'] is False and review['source_modified'] is False
            and review['production_thresholds_selected'] is False, 'Independent target CPU proof identity/scope missing')


def validate_runtime_inputs(proofs):
    paths = {role: bound_local(proofs[role]) for role in DATAROW_PROOF_ROLES}
    require(paths['runtime_manifest'].name == FINAL_RUNTIME_MANIFEST_NAME, 'Not exact v8 runtime manifest')
    manifest = json.loads(paths['runtime_manifest'].read_text())
    require(manifest['source_commit'] == SOURCE and manifest['runtime_changed_filenames'] == DATAROW_CHANGED
            and manifest['datarow_change_scope'] == DATAROW_SCOPE
            and manifest['base_v7_manifest_sha256'] == BASE_RUNTIME_MANIFEST_SHA
            and manifest['added_filenames'] == [], 'Not exact preserved-v7 audit-row dtype scope')
    base_path = OUT / 'CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json'
    require(sha(base_path) == BASE_RUNTIME_MANIFEST_SHA, 'Preserved v7 factory manifest changed')
    original = {Path(r['path']).name: r for r in json.loads(base_path.read_text())['files']}
    records = manifest['files']; fresh = {Path(r['path']).name: r for r in records}
    require(len(records) == len(fresh) == len(original) == 14 and set(fresh) == set(original), 'Datarow fix cannot add/remove helpers')
    changed = []
    old_line = b"batch.non_tensor_batch['audit_trajectory_row'] = np.arange(2560, dtype=np.int64)"
    new_line = b"batch.non_tensor_batch['audit_trajectory_row'] = np.arange(2560, dtype=object)"
    for name, record in fresh.items():
        path = RUNTIME / name; old = OLD_RUNTIME / name
        require(Path(record['path']) == path and path.is_file() and not path.is_symlink()
                and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'Runtime manifest differs from actual v8 bytes')
        require(old.stat().st_size == original[name]['bytes'] and sha(old) == original[name]['sha256'], 'Preserved v7 runtime bytes changed')
        before = old.read_bytes(); after = path.read_bytes()
        if before != after: changed.append(name)
        if name == 'calibration_driver_v1.py':
            require(before.count(old_line) == 1 and after == before.replace(old_line, new_line, 1), 'Driver must change only audit-row metadata dtype literal')
        else: require(before == after, 'Probability/worker/factory/wrapper/guard bytes may not change')
    require(changed == DATAROW_CHANGED and fresh['calibration_run_v1.py']['sha256'] ==
            '7aa80984aee6a5589ee44f9b72c4bed0d091e30c38e9c181fe2cc2728d1172c9'
            and fresh['metadata_colocation_v1.py']['sha256'] ==
            '39e3226c82132524e57ee1adc964f607da6a82314a784c1081c505fd10ea6b9e', 'Frozen factory/cooldown identity differs')
    validate_runtime_review(proofs, json.loads(paths['runtime_review'].read_text()), changed)
    cpu = json.loads(paths['runtime_cpu'].read_text())
    require(cpu['status'] == 'PASS' and cpu['scope'] == 'ACTUAL_DATAPROTO_SELECT_ROWS_AND_ONE_OUTER_CPU_ONLY'
            and cpu['runtime_driver_sha256'] == fresh['calibration_driver_v1.py']['sha256']
            and type(cpu['tests_run']) is int and cpu['tests_run'] >= 9 and cpu['errors'] == cpu['failures'] == 0
            and cpu['actual_source_protocol'] is True and cpu['actual_full_one_outer_function_body'] is True
            and cpu['actual_CUDA_initialized'] is False and cpu['Ray_cluster_created'] is False
            and cpu['actual_GPU_acceptance'] is False and type(cpu['actual_Adam_writes']) is int
            and cpu['actual_Adam_writes'] == 0, 'Actual source DataProto/one_outer CPU fixture missing; not GPU acceptance')
    validate_target_cpu_inputs(proofs, paths, fresh['calibration_driver_v1.py']['sha256'])
    return manifest, paths


def cpu_success_count(text):
    """Accept explicit successful pytest or unittest terminals, never a failure summary."""
    if re.search(r'\b(FAILED|FAILURES|ERRORS|ERROR|failed|error)\b',text):return 0
    pytest=re.search(r'(\d+) passed(?:[, ].*)?\s*$',text)
    unittest=re.search(r'Ran (\d+) tests? in [^\n]+\n\s*OK\s*$',text)
    match=pytest or unittest
    return int(match.group(1)) if match is not None else 0


def validate_failure_inputs(proofs, paths):
    payload = json.loads(paths['failure_payload'].read_text()); review = json.loads(paths['failure_review'].read_text())
    run = str(ROOT / 'runs/gigpo-e1-policy-safety-calibration25-job3281-1791243032869507143')
    binding = '1314632c490217442a46187e3dbbee0d97742f486e8eb16112eac8b919ab7a86'
    require(review['status'] == 'PASS_FAILURE_PRESERVATION_AND_ZERO_UPDATE_SCOPE_ONLY'
            and review['scope'] == 'CALIBRATION_STARTUP_METADATA_DTYPE_FAILURE_NOT_PROBABILITY_ACCEPTANCE'
            and review['job_id'] == '3281' and review['run'] == run and review['code_commit'] == SOURCE
            and review['binding_sha256'] == binding and review['restore_manifest_sha256'] == MANIFEST
            and review['producer_status'] == review['terminal_state'] == 'FAILED' and review['terminal_exit_code'] == '1:0'
            and all(type(review[k]) is int and review[k] == 0 for k in ('actual_adam_writes', 'actual_gradient_updates',
                'actual_new_outer_updates', 'retained_training_updates_contributed'))
            and review['cgroup_removed'] is True and review['cleanup_residual_states'] == {}
            and review['final_gpu_compute_empty'] is True, 'Not actual failed3281 zero-update/resource acceptance')
    require(review['evidence'][0]['path'] == proofs['failure_payload']['path']
            and review['evidence'][0]['sha256'] == proofs['failure_payload']['sha256'], 'Failure independent review consumes a different original payload')
    require(payload['job_id'] == '3281' and payload['node'] == 'h100-8' and payload['run'] == run, 'Failure raw job/node/run differs')
    originals = []; by = {}
    for record in payload['originals']:
        data = record['text'].encode('utf-8'); p = Path(record['path'])
        require(p.is_absolute() and '..' not in p.parts and (p.parent == ROOT/'logs' or p.is_relative_to(run))
                and type(record['bytes']) is int and len(data) == record['bytes']
                and hashlib.sha256(data).hexdigest() == record['sha256'], 'Original failed3281 bytes/SHA corrupt')
        item = {k: record[k] for k in ('path', 'bytes', 'sha256')}; originals.append(item); by[item['path']] = record
    require(len(by) == len(originals) == 27 and review['evidence'][1:] == originals, 'Exact failed3281 original inventory differs')
    terminal_path = str(ROOT/'logs/policy-safety-calibration-terminal-3281-raw-v1.json')
    terminal = json.loads(by[terminal_path]['text']); result = json.loads(by[run+'/run-result.json']['text'])
    require(by[terminal_path]['sha256'] == review['terminal_sha256'] and terminal['fields']['JobId'] == '3281'
            and terminal['fields']['NodeList'] == 'h100-8' and terminal['fields']['JobState'] == 'FAILED'
            and terminal['fields']['ExitCode'] == '1:0' and str(terminal['fields']['Requeue']) == str(terminal['fields']['Restarts']) == '0', 'Not original failed3281 terminal')
    require(by[run+'/run-result.json']['sha256'] == review['result_sha256'] and result['job_id'] == '3281'
            and result['status'] == 'FAILED' and result['source_commit'] == SOURCE and result['binding_sha256'] == binding
            and result['restore_checkpoint'] == CHECKPOINT and result['restore_manifest_sha256'] == MANIFEST
            and result['retained_training_updates_contributed'] == 0
            and result['cleanup_residual_states'] == {} and result['final_gpu']['processes'] == [], 'Failure result/source/restore identity differs')
    q = payload['queries']
    require(q['queue']['returncode'] == q['compute']['returncode'] == 0 and not q['queue']['stdout'].strip()
            and not q['compute']['stdout'].strip(), 'Failure allocation/compute release not proven')
    expected_cgroups = {f'/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3281{suffix}' for suffix in ('', '/step_0')}
    require(len(payload['cgroup']) == 2 and {x['path'] for x in payload['cgroup']} == expected_cgroups
            and all(x['exists'] is False and x['errno'] == 2 and x['ancestor_readable'] is True for x in payload['cgroup']), 'Failed3281 cgroup release lacks exact original evidence')
    expected_audit = {run+'/audit/'+name for rank in range(8) for name in (f'{rank}.jsonl', f'numeric/micro-{rank}.jsonl', f'policy-safety/rank-{rank}.jsonl')}
    require(len(payload['audit_paths']) == 24 and {x['path'] for x in payload['audit_paths']} == expected_audit
            and all(x['status'] == 'ABSENT' and x['errno'] == 2 and x['ancestor_readable'] is True for x in payload['audit_paths']), 'Wrong/missing source-defined failed3281 audits')
    log = by[run+'/calibration.log']['text']
    require('batch = select_rows(batch, sources)' in log and 'val.dtype == object' in log
            and 'data in the non_tensor_batch must be a numpy.array with dtype=object' in log,
            'Not actual metadata dtype failure before probability/Adam work')
    require(by[run+'/restore25-before-baseline.json']['sha256'] == by[run+'/restore25-before-rollout.json']['sha256'], 'Two actual restore receipts differ')
    baseline = json.loads(by[run+'/baseline-evaluation.json']['text'])
    require(baseline['logical_step'] == 25 and baseline['retained_training_updates_contributed'] == 0
            and baseline['full512']['val/qa_count/all'] == 512 and baseline['stage64']['val/qa_count/all'] == 64, 'Actual restore baseline scope missing')
    sampling = json.loads(paths['failure_sampling_metadata'].read_text()); trace = json.loads(paths['failure_trace_metadata'].read_text())
    require(sampling['transport_returncode'] == trace['transport_returncode'] == 0, 'Raw metadata SSH query failed')
    raw = sampling['actual']; traces = trace['actual']
    require(raw['job_id'] == traces['job_id'] == '3281' and raw['run'] == traces['run'] == run
            and raw['binding_sha256'] == binding and raw['raw_pt_loaded_with_torch'] is False
            and raw['pt_transferred_to_mac'] is False and raw['gpu_used'] is False
            and raw['probability_approval_generated'] is False, 'Raw metadata identity/scope differs')
    require(raw['training_prefix_original']['turns'] == 4 and raw['training_prefix_original']['source_set_is_exactly_0_2559'] is True
            and raw['training_prefix_original']['source_step_pairs_unique'] is True
            and [x['real_active_rows'] for x in raw['per_turn_coverage']] == [2560,2556,776,245], 'Actual turn/source identity incomplete')
    receipts = raw['worker_receipt_diagnostics']; inventory = raw['original_probability_files']
    require(len(receipts) == 32 and {(x['serial'], x['rank']) for x in receipts} == {(s,rank) for s in range(4) for rank in range(8)}
            and len(inventory) == 64 and len({x['path'] for x in inventory}) == 64
            and all(x['raw_declared_sha_matches'] is True and x['source_commit'] == SOURCE and x['phase'] == 'train' for x in receipts), 'Actual eight-worker turn originals incomplete')
    trace_records = traces['trace_originals']
    require(len(trace_records) == 4 and all(x['rows'] == 2560 and x['trajectory_indices_exact_0_2559'] is True
            and x['question_response_group_mapping_valid'] is True and x['unique_question_groups'] == 512 for x in trace_records), 'Actual four-turn five-response group traces incomplete')
    sampling_originals = [{k:x[k] for k in ('path','bytes','sha256')} for x in inventory+[raw['training_prefix_original']]+trace_records]
    require(len(sampling_originals) == len({x['path'] for x in sampling_originals}) == 69
            and all(Path(x['path']).is_relative_to(run) for x in sampling_originals), 'Sampling originals not confined to actual failed job')
    return originals, sampling_originals


def seal_installer(candidate, sources, archive):
    """Seal only our new local unsealed installer, after immutable draft/package exist."""
    path=HERE.with_name('safety-calibration-install-v8')/'install_calibration_v8.py'
    source=path.read_text()
    manifest_record=candidate['datarow_control_proofs']['runtime_manifest']
    constants=dict(PACKAGE_SHA=sha(archive),CANDIDATE_SHA=sha(OUT/'CALIBRATION-CANDIDATE-52BEFE9-v8.json'),
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


def inherited_local_proof(record):
    """Resolve the sole predecessor work-file proof using its frozen original record."""
    local = OUT / Path(record['path']).name
    if local.exists(): return local
    base = OUT / 'CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json'
    require(sha(base) == BASE_RUNTIME_MANIFEST_SHA, 'Inherited factory manifest changed')
    probe = json.loads(base.read_text())['proofs']['cpu_probe_script']
    local = Path(probe['path'])
    require(Path(record['path']).name == local.name == 'test_metadata_colocation_v7_actual.py'
            and record['bytes'] == probe['bytes'] and record['sha256'] == probe['sha256']
            and local.resolve() == local and local.is_file() and not local.is_symlink(),
            'Inherited proof has no exact original local source')
    return local


def main(proof_inputs_path, proof_inputs_sha):
    target=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v8.json';archive=HERE/'policy-safety-calibration-52befe9-v8-package.tar.gz'
    receipt_path=OUT/'CALIBRATION-PACKAGE-v8.json'
    require(not any(path.exists() for path in (target,archive,receipt_path)), 'Draft already/partly exists; inspect and preserve, never rebuild')
    spec_path=Path(proof_inputs_path)
    require(spec_path.is_absolute() and spec_path.resolve()==spec_path and spec_path.is_relative_to(OUT)
            and spec_path.is_file() and not spec_path.is_symlink()
            and sha(spec_path)==proof_inputs_sha,'Explicit dedicated proof input SHA/path changed')
    spec=json.loads(spec_path.read_text());proofs=spec['proofs']
    require(spec['schema']==1 and spec['code_commit']==SOURCE and spec['runtime_version']=='v8'
            and set(proofs)==set(DATAROW_PROOF_ROLES),'Wrong/missing final proof role specification')
    require(len({record['path'] for record in proofs.values()})==len(proofs),'Duplicate explicit proof role original')
    runtime,proof_paths=validate_runtime_inputs(proofs)
    failure_originals,sampling_originals=validate_failure_inputs(proofs,proof_paths)
    native_path=OUT/'NATIVE-CANDIDATE-52BEFE9-v2.json';predecessor_path=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v7.json'
    require(sha(native_path)==NATIVE_SHA and sha(predecessor_path)==PREDECESSOR_SHA,'Frozen predecessor/native candidate changed')
    native=json.loads(native_path.read_text());predecessor=json.loads(predecessor_path.read_text())
    require(native['code_commit']==SOURCE and native['helpers']==str(NATIVE_HELPERS) and len(native['files'])==42327,
            'Wrong inherited native source/assets closure')
    candidate=copy.deepcopy(predecessor)
    candidate.update(helpers=str(HELPERS),controls=str(CONTROLS),source_native_acceptance=None,
        native_pending_job_acceptance=None,predecessor_calibration_candidate_sha256=PREDECESSOR_SHA,
        preparation_scope='External audit-row metadata object dtype only; all v7 factory, v6-v3 startup and probability bytes preserved; actual native/physical/READY separate',
        startup_change_scope=STARTUP_SCOPE,startup_runtime_changed_filenames=['calibration_run_v1.py'],
        factory_change_scope=FACTORY_SCOPE,factory_runtime_changed_filenames=FACTORY_CHANGED,
        datarow_change_scope=DATAROW_SCOPE,runtime_changed_filenames=DATAROW_CHANGED,
        datarow_base_runtime_manifest_sha256=BASE_RUNTIME_MANIFEST_SHA,
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
    previous_failure_originals={record['path'] for key in ('failure3272_originals','failure3276_originals') for record in predecessor.get(key,[])}
    for record in predecessor['evidence']:
        if record['path'] not in native_evidence|previous_failure_originals:
            local=inherited_local_proof(record)
            require(local.stat().st_size==record['bytes'] and sha(local)==record['sha256'],'Original calibration evidence changed')
            sources['evidence/'+local.name]=local
    evidence={record['path']:record for record in candidate['evidence']}
    def add_local(local):
        local=Path(local);record=dict(path=str(ROOT/'logs'/local.name),bytes=local.stat().st_size,sha256=sha(local))
        require(record['path'] not in evidence,'New proof would overwrite an old evidence path')
        evidence[record['path']]=record;sources['evidence/'+local.name]=local
        return record
    candidate['datarow_control_proofs']={role:add_local(path) for role,path in proof_paths.items()}
    candidate['datarow_proof_input_spec']=add_local(spec_path)
    for role in ('actual_cpu_stdout','actual_cpu_stderr','cpu_probe_script'):
        record=runtime['proofs'][role];path=Path(record['path'])
        require(path.is_file() and not path.is_symlink() and path.resolve()==path
                and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],
                'Exact original datarow CPU supplementary proof changed')
        add_local(path)
    for local in (OUT/'CALIBRATION-DEPLOYMENT-v8.md',OUT/'CALIBRATION-DEPLOYMENT-CPU-v8-v2.log'):add_local(local)
    for record in failure_originals+sampling_originals:
        require(record['path'] not in evidence or evidence[record['path']]==record,'Conflicting preserved failure original')
        evidence[record['path']]=record
    candidate['failure3281_originals']=failure_originals;candidate['failure3281_sampling_originals']=sampling_originals;candidate['evidence']=list(evidence.values())
    preview=json.loads((OUT/'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json').read_text())
    require(preview_config(candidate)==preview['resolved'],'New startup/runtime paths altered reviewed optimization/config')
    validate_candidate(candidate)
    write_once(target,candidate);sources['candidate/'+target.name]=target;sources['cpu/test_deployment_v8.py']=HERE/'test_deployment_v8.py'
    make_archive(archive,sources);installer=seal_installer(candidate,sources,archive)
    receipt=dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',candidate=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)),
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),installer=installer,
        proof_inputs=dict(path=str(spec_path),sha256=proof_inputs_sha),predecessor_candidate_sha256=PREDECESSOR_SHA,
        native_candidate_sha256=NATIVE_SHA,native_acceptance=None,submitted_job=None,retained_training_updates_contributed=0,
        runtime_helpers_path=str(HELPERS),controls_path=str(CONTROLS),startup_change_scope=STARTUP_SCOPE,
        factory_change_scope=FACTORY_SCOPE,datarow_change_scope=DATAROW_SCOPE,runtime_changed_filenames=DATAROW_CHANGED,unchanged_runtime_files=13,
        inherited_native_42327_paths_sha_unchanged=True,native_directory_modified=False,actual_gpu_executed=False,
        file_count=len(candidate['files']),evidence_count=len(candidate['evidence']),package_members=sorted(sources))
    write_once(receipt_path,receipt)
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),installer_sha256=installer['sha256'],
        files=len(files),evidence=len(evidence),ready=False,submitted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-inputs',required=True);parser.add_argument('--proof-inputs-sha',required=True)
    args=parser.parse_args();main(args.proof_inputs,args.proof_inputs_sha)
