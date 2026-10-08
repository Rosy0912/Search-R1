"""Prepare isolated v9 FLASH propagation/backend gate; local unlaunched only."""
import argparse,ast,copy,gzip,hashlib,io,json,os,re,tarfile
from pathlib import Path
from deployment_contract_v9 import *
HERE=Path(__file__).resolve().parent
LOCAL=HERE.parents[2]
OUT=LOCAL/'outputs/gigpo-e1-policy-safety'
RUNTIME=HERE.with_name('safety-calibration-v9')
OLD_RUNTIME=HERE.with_name('safety-calibration-v8')
BASE_RUNTIME_MANIFEST_SHA=DATAROW_BASE_MANIFEST_SHA # inherited factory original, not v9 backend base
FINAL_RUNTIME_MANIFEST_NAME='CALIBRATION-RUNTIME-IMPLEMENTATION-v9.json'
CONTROL_NAMES={'deployment_contract_v9.py','freeze_ready_v9.py','prepare_candidate_v9.py','submit_once_v9.py','verify_physical_v9.py'}
SHA_RE=re.compile(r'[0-9a-f]{64}')
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

def cpu_success_count(text):
    """Accept explicit successful pytest or unittest terminals, never a failure summary."""
    if re.search(r'\b(FAILED|FAILURES|ERRORS|ERROR|failed|error)\b',text):return 0
    pytest=re.search(r'(\d+) passed(?:, \d+ (?:warnings?|skipped|xfailed|xpassed))* in [0-9.]+s\s*$',text)
    unittest=re.search(r'Ran (\d+) tests? in [^\n]+\n\s*OK\s*$',text)
    match=pytest or unittest
    return int(match.group(1)) if match is not None else 0

def seal_installer(candidate, sources, archive):
    """Seal only our new local unsealed installer, after immutable draft/package exist."""
    path=HERE.with_name('safety-calibration-install-v9')/'install_calibration_v9.py'
    source=path.read_text()
    manifest_record=candidate['backend_control_proofs']['runtime_manifest']
    constants=dict(PACKAGE_SHA=sha(archive),CANDIDATE_SHA=sha(OUT/'CALIBRATION-CANDIDATE-52BEFE9-v9.json'),
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
    """Resolve only the two exact predecessor work-file proofs via frozen manifests."""
    local=OUT/Path(record['path']).name
    if local.exists():return local
    known={
        'test_metadata_colocation_v7_actual.py':('CALIBRATION-RUNTIME-IMPLEMENTATION-v7.json',DATAROW_BASE_MANIFEST_SHA),
        'test_calibration_datarow_v8_actual.py':('CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json',BACKEND_BASE_MANIFEST_SHA)}
    name=Path(record['path']).name;require(name in known,'Inherited proof has no exact original local source')
    manifest,expected=known[name];base=OUT/manifest
    require(sha(base)==expected,'Inherited runtime manifest changed')
    probe=json.loads(base.read_text())['proofs']['cpu_probe_script'];local=Path(probe['path'])
    require(name==local.name and record['bytes']==probe['bytes'] and record['sha256']==probe['sha256']
        and local.resolve()==local and local.is_file() and not local.is_symlink(),
        'Inherited proof differs from exact predecessor original')
    return local

FACTORY_RUNTIME=HERE.with_name('safety-calibration-v7')
def validate_inherited_runtime_review(proofs, review, changed):
    require(review['status'] == 'PASS' and review.get('code_commit', review.get('source_commit')) == SOURCE
            and review['runtime_manifest_sha256'] == proofs['runtime_manifest']['sha256']
            and review['runtime_diff_sha256'] == proofs['runtime_diff']['sha256']
            and review['runtime_cpu_sha256'] == proofs['runtime_cpu']['sha256'], 'Datarow review does not bind exact manifest/diff/CPU')
    require(review['datarow_change_scope'] == DATAROW_SCOPE and review['runtime_changed_filenames'] == changed
            and review['gpu_used'] is False and review['source_modified'] is False
            and review['production_thresholds_selected'] is False, 'Datarow review exceeds metadata dtype scope')

def validate_inherited_target_cpu_inputs(proofs, paths, driver_sha):
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

def validate_inherited_datarow_inputs(proofs):
    paths = {role: bound_local(proofs[role]) for role in DATAROW_PROOF_ROLES}
    require(paths['runtime_manifest'].name == 'CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json', 'Not exact v8 runtime manifest')
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
        path = OLD_RUNTIME / name; old = FACTORY_RUNTIME / name
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
    validate_inherited_runtime_review(proofs, json.loads(paths['runtime_review'].read_text()), changed)
    cpu = json.loads(paths['runtime_cpu'].read_text())
    require(cpu['status'] == 'PASS' and cpu['scope'] == 'ACTUAL_DATAPROTO_SELECT_ROWS_AND_ONE_OUTER_CPU_ONLY'
            and cpu['runtime_driver_sha256'] == fresh['calibration_driver_v1.py']['sha256']
            and type(cpu['tests_run']) is int and cpu['tests_run'] >= 9 and cpu['errors'] == cpu['failures'] == 0
            and cpu['actual_source_protocol'] is True and cpu['actual_full_one_outer_function_body'] is True
            and cpu['actual_CUDA_initialized'] is False and cpu['Ray_cluster_created'] is False
            and cpu['actual_GPU_acceptance'] is False and type(cpu['actual_Adam_writes']) is int
            and cpu['actual_Adam_writes'] == 0, 'Actual source DataProto/one_outer CPU fixture missing; not GPU acceptance')
    validate_inherited_target_cpu_inputs(proofs, paths, fresh['calibration_driver_v1.py']['sha256'])
    return manifest, paths

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


def one_outer_ast_sha(path):
    tree=ast.parse(Path(path).read_text())
    node=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='one_outer')
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


def validate_runtime_review(proofs,review,changed):
    require(review['status']=='PASS' and review.get('source_commit',review.get('code_commit'))==SOURCE
        and review['runtime_manifest_sha256']==proofs['runtime_manifest']['sha256']
        and review['runtime_diff_sha256']==proofs['runtime_diff']['sha256']
        and review['runtime_cpu_sha256']==proofs['runtime_cpu']['sha256']
        and review['backend_change_scope']==BACKEND_SCOPE and review['runtime_changed_filenames']==changed
        and review['gpu_used'] is False and review['source_modified'] is False
        and review['production_thresholds_selected'] is False,'New backend review does not close exact runtime/diff/CPU/static scope')


def validate_backend_runtime(proofs):
    require(set(proofs)==set(BACKEND_PROOF_ROLES),'Explicit backend proof roles missing/extra')
    paths={role:bound_local(proofs[role]) for role in BACKEND_PROOF_ROLES}
    require(paths['runtime_manifest'].name==FINAL_RUNTIME_MANIFEST_NAME,'Not the v9 runtime original')
    runtime=json.loads(paths['runtime_manifest'].read_text())
    require(runtime['source_commit']==SOURCE and runtime['base_v8_manifest_sha256']==BACKEND_BASE_MANIFEST_SHA
        and runtime['runtime_changed_filenames']==BACKEND_CHANGED and runtime['added_filenames']==[]
        and runtime['backend_change_scope']==BACKEND_SCOPE and runtime['factory_sha256']==
            '39e3226c82132524e57ee1adc964f607da6a82314a784c1081c505fd10ea6b9e'
        and runtime['preserved_v8_ready_binding']['sha256']==PREDECESSOR_READY_SHA,
        'Not exact source52/actualv8/three-file backend scope')
    base=OUT/'CALIBRATION-RUNTIME-IMPLEMENTATION-v8.json'
    require(sha(base)==BACKEND_BASE_MANIFEST_SHA,'Preserved v8 manifest changed')
    previous={Path(row['path']).name:row for row in json.loads(base.read_text())['files']}
    fresh={Path(row['path']).name:row for row in runtime['files']}
    require(len(runtime['files'])==len(fresh)==len(previous)==14 and set(fresh)==set(previous),
        'Backend fix cannot add/remove fourteen helpers')
    changed=[]
    for name,row in fresh.items():
        path=RUNTIME/name;old=OLD_RUNTIME/name
        require(Path(row['path'])==path and path.is_file() and not path.is_symlink()
            and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Actual new helper bytes differ')
        require(old.stat().st_size==previous[name]['bytes'] and sha(old)==previous[name]['sha256'],
            'Old runtime originals changed')
        if path.read_bytes()!=old.read_bytes():changed.append(name)
    require(sorted(changed)==BACKEND_CHANGED and len(runtime['unchanged_v8_filenames'])==11
        and set(runtime['unchanged_v8_filenames'])==set(previous)-set(BACKEND_CHANGED),
        'Backend changes exceed exact reviewed three files')
    require(one_outer_ast_sha(RUNTIME/'calibration_driver_v1.py')==one_outer_ast_sha(OLD_RUNTIME/'calibration_driver_v1.py')
        and runtime['one_outer_full_AST_equal_to_v8'] is True,
        'Old metadata scope cannot cover altered one_outer body')
    validate_runtime_review(proofs,json.loads(paths['runtime_review'].read_text()),BACKEND_CHANGED)
    declared=runtime['proofs']
    for external,internal in (('runtime_diff','runtime_diff'),('runtime_cpu','local_cpu_log'),
        ('runtime_target_cpu_import','target_cpu_import'),('installed_backend_evidence','installed_backend_evidence'),('diagnosis','diagnosis')):
        require(proofs[external]==declared[internal],'Backend proof original relabeled/substituted: '+external)
    require(cpu_success_count(paths['runtime_cpu'].read_text())>=47
        and runtime['gpu_used'] is False and runtime['source_modified'] is False
        and runtime['production_thresholds_selected'] is False and runtime['tolerance_selected'] is False
        and runtime['checks']['actual_GPU_acceptance'] is False,
        'No successful narrow backend CPU/static-only proof')
    validate_target_import(json.loads(paths['runtime_target_cpu_import'].read_text()),
        json.loads(paths['installed_backend_evidence'].read_text()))
    return runtime,paths


def validate_target_import(raw,evidence):
    actual=raw['actual']
    require(raw['returncode']==0 and json.loads(raw['stdout'])==actual
        and actual['scope']=='CPU_DYNAMIC_IMPORT_AND_READONLY_HARDWARE_METADATA_ONLY_NOT_GPU_EXECUTION_OR_PARITY'
        and actual['node']=='h100-8' and actual['torch_version']=='2.4.0+cu121'
        and actual['CUDA_initialized_before'] is False and actual['CUDA_initialized_after'] is False
        and actual['GPU_used'] is False and actual['dynamic_extension_import_success'] is True
        and actual['vllm_flash_attn_version']=='2.6.2' and 128 in actual['flash_supported_head_sizes']
        and actual['flash_class']=='vllm.attention.backends.flash_attn.FlashAttentionBackend'
        and actual['hardware_query_returncode']==0 and len(actual['hardware_query_stdout'].strip().splitlines())==8
        and actual['actual_8GPU_kernel_support'] is False and actual['probability_accepted'] is False
        and actual['tolerance_selected'] is False,'Target CPU import is not actual eight-GPU kernel/parity acceptance')
    require(evidence['source_commit']==SOURCE and evidence['requested_new_backend']=='FLASH_ATTN'
        and evidence['actual_8GPU_kernel_support'] is False and evidence['actual_8GPU_probability_parity'] is False
        and evidence['production_thresholds_selected'] is False and evidence['tolerance_selected'] is False,
        'Installed backend evidence claims unsupported actual acceptance')
    native=json.loads((OUT/'NATIVE-CANDIDATE-52BEFE9-v2.json').read_text())
    closure={row['path']:row for row in native['files']}
    records=actual['original_bound_records']
    require(len(records)==len({r['path'] for r in records})==5,'Target selector/backend closure incomplete/duplicate')
    for row in records:
        expected=closure[row['path']]
        require(expected['sha256']==row['sha256'],'Installed selector/backend differs from native bound bytes')
    paths=actual['imported_paths']
    require(paths==dict(flash=str(ROOT/'env/lib/python3.10/site-packages/vllm/vllm_flash_attn/__init__.py'),
        backend=str(ROOT/'env/lib/python3.10/site-packages/vllm/attention/backends/flash_attn.py'),
        selector=str(ROOT/'env/lib/python3.10/site-packages/vllm/attention/selector.py')),
        'Target imports are from another runtime')
    lines=actual['hardware_query_stdout'].strip().splitlines()
    require([line.split(',')[0].strip() for line in lines]==[str(i) for i in range(8)]
        and all(line.split(',')[-1].strip()=='9.0' for line in lines)
        and len({line.split(',')[1].strip() for line in lines})==8,'Actual hardware indices/UUID/SM observation differs')


def validate_current_target_cpu_inputs(proofs, paths, driver_sha, runtime_manifest_sha):
    target=json.loads(paths['runtime_target_cpu'].read_text())
    raw=json.loads(paths['runtime_target_cpu_raw'].read_text())
    review=json.loads(paths['runtime_target_cpu_review'].read_text())
    require(raw['transport_returncode']==0 and raw['actual']['returncode']==0,'Current target CPU raw command failed')
    receipt=raw['actual']['records']['target_cpu']
    require(receipt['path']==str(ROOT/'logs/CALIBRATION-DATAROW-TARGET-CPU-v9.json')
        and receipt['bytes']==proofs['runtime_target_cpu']['bytes']
        and receipt['sha256']==proofs['runtime_target_cpu']['sha256']
        and receipt['data']==target and json.loads(raw['actual']['stdout'])==target,
        'New target CPU original bytes/SHA/data does not close')
    import base64
    raw_original=json.loads(raw['raw_stdout']);encoded=raw_original.pop('cpu_base64')
    require(raw_original==raw['actual'],'Target CPU raw stdout differs from parsed actual')
    original=base64.b64decode(encoded,validate=True)
    require(len(original)==receipt['bytes'] and hashlib.sha256(original).hexdigest()==receipt['sha256']
        and json.loads(original)==target,'Current target CPU base64 original corrupt')
    require(target['status']=='PASS' and target['scope']=='ACTUAL_DATAPROTO_SELECT_ROWS_AND_ONE_OUTER_CPU_ONLY'
        and target['runtime_driver_sha256']==driver_sha and target['torch_version']=='2.4.0+cu121'
        and target['tensordict_version']=='0.5.0' and target['tests_run']==9
        and type(target['actual_Adam_writes']) is int and target['actual_Adam_writes']==0
        and target['errors']==target['failures']==0 and target['actual_CUDA_initialized'] is False
        and target['Ray_cluster_created'] is False and target['actual_GPU_acceptance'] is False
        and target['actual_source_protocol'] is True and target['actual_full_one_outer_function_body'] is True
        and target['actual_protocol_path']==str(ROOT/'repo-policy-safety-52befe9/verl/protocol.py')
        and target['actual_select_rows_path']==str(ROOT/'repo-policy-safety-52befe9/search_r1/baselines/rollout.py'),
        'Current frozen driver target CPU does not prove exact DataProto/one_outer CPU-only scope')
    require(review['status']=='PASS' and review['source_commit']==SOURCE and review['node']=='h100-8'
        and review['runtime_manifest_sha256']==runtime_manifest_sha and review['runtime_driver_sha256']==driver_sha
        and review['runtime_target_cpu_sha256']==proofs['runtime_target_cpu']['sha256']
        and review['runtime_target_cpu_raw_sha256']==proofs['runtime_target_cpu_raw']['sha256']
        and review['datarow_change_scope']==DATAROW_SCOPE and review['runtime_changed_filenames']==DATAROW_CHANGED
        and review['gpu_used'] is False and review['source_modified'] is False
        and review['production_thresholds_selected'] is False and review['actual_GPU_backend_accepted'] is False
        and review['actual_probability_accepted'] is False and type(review['actual_Adam_writes']) is int
        and review['actual_Adam_writes']==0,'New driver target CPU independent identity/scope missing')


def main(proof_inputs_path,proof_inputs_sha):
    target=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v9.json'
    archive=HERE/'policy-safety-calibration-52befe9-v9-package.tar.gz';receipt=OUT/'CALIBRATION-PACKAGE-v9.json'
    require(not any(path.exists() for path in (target,archive,receipt)),'Existing or partial v9 package must be preserved')
    specpath=Path(proof_inputs_path);require(sha(specpath)==proof_inputs_sha,'Proof specification bytes changed')
    spec=json.loads(specpath.read_text());proofs=spec['proofs']
    require(spec['schema']==1 and spec['code_commit']==SOURCE and spec['runtime_version']=='v9'
        and len({row['path'] for row in proofs.values()})==len(proofs),'Wrong/duplicate explicit new proof spec')
    runtime,paths=validate_backend_runtime(proofs)
    current_proofs=spec['current_driver_target_cpu_proofs']
    require(set(current_proofs)==set(CURRENT_DRIVER_CPU_ROLES),'Explicit new driver target CPU roles missing/extra')
    current_paths={role:bound_local(row) for role,row in current_proofs.items()}
    driver_sha=next(row['sha256'] for row in runtime['files'] if Path(row['path']).name=='calibration_driver_v1.py')
    validate_current_target_cpu_inputs(current_proofs,current_paths,driver_sha,proofs['runtime_manifest']['sha256'])
    parentpath=OUT/'CALIBRATION-CANDIDATE-52BEFE9-v8.json'
    readypath=OUT/'policy-safety-calibration-52befe9-reviewed-binding-v8.json'
    require(sha(parentpath)==PREDECESSOR_CANDIDATE_SHA and sha(readypath)==PREDECESSOR_READY_SHA,
        'Actual preserved parent candidate/READY changed')
    parent=json.loads(parentpath.read_text());parent_ready=json.loads(readypath.read_text())
    require(parent_ready['status']=='REVIEWED_FOR_NATIVE_ONLY'
        and parent_ready['readiness']=='READY_FOR_PROBABILITY_PROBE_THEN_WAIT_ONLY'
        and parent_ready['candidate_sha256']==PREDECESSOR_CANDIDATE_SHA
        and parent_ready['source_native_acceptance']['job_id']=='3267','Actual parent READY/native identity differs')
    oldspec=OUT/'CALIBRATION-DEPLOYMENT-PROOF-INPUTS-v8.json'
    validate_inherited_datarow_inputs(json.loads(oldspec.read_text())['proofs'])
    native=json.loads((OUT/'NATIVE-CANDIDATE-52BEFE9-v2.json').read_text())
    require(native['code_commit']==SOURCE and len(native['files'])==42327,'Inherited native closure changed')
    candidate=copy.deepcopy(parent);candidate.pop('unchanged_runtime_wrapper_sha256')
    candidate.update(helpers=str(HELPERS),controls=str(CONTROLS),source_native_acceptance=None,
        predecessor_calibration_candidate_sha256=PREDECESSOR_CANDIDATE_SHA,
        preserved_v8_ready_binding_sha256=PREDECESSOR_READY_SHA,
        historical_runtime_proof_versions=dict(startup='v6-v3',factory='v7',datarow='v8'),
        datarow_runtime_changed_filenames=DATAROW_CHANGED,runtime_changed_filenames=BACKEND_CHANGED,
        inherited_v8_wrapper_sha256=parent['unchanged_runtime_wrapper_sha256'],
        backend_change_scope=BACKEND_SCOPE,backend_base_runtime_manifest_sha256=BACKEND_BASE_MANIFEST_SHA,
        backend_unchanged_runtime_files=11,expected_rollout_attention_backend='FLASH_ATTN',
        actual_backend_gate_required_before_restore_baseline_adam=True,
        actual_v9_runtime_wrapper_sha256=next(row['sha256'] for row in runtime['files'] if Path(row['path']).name=='calibration_run_v1.py'),
        preserved_v8_one_outer_ast_sha256=one_outer_ast_sha(OLD_RUNTIME/'calibration_driver_v1.py'),
        preparation_scope='V9 explicit FLASH_ATTN propagation and all8 actual read-only backend gate before restore/baseline/Adam; native/source/math/state and one_outer AST unchanged')
    previous={row['path']:row for row in parent['files']};files={}
    for row in native['files']:
        entry=previous[row['path']];require(entry['sha256']==row['sha256'],'Native parent files changed')
        files[row['path']]=copy.deepcopy(entry)
    sources={};runtime_records=[]
    for row in runtime['files']:
        local=Path(row['path']);entry=dict(path=str(HELPERS/local.name),bytes=row['bytes'],sha256=row['sha256'])
        require(entry['path'] not in files,'Runtime cannot overwrite native/source path')
        runtime_records.append(entry);files[entry['path']]=entry;sources['runtime/'+local.name]=local
    for name in sorted(CONTROL_NAMES):
        local=HERE/name;entry=dict(path=str(CONTROLS/name),bytes=local.stat().st_size,sha256=sha(local))
        files[entry['path']]=entry;sources['control/'+name]=local
    require(len(files)==42346,'Native42327+14+5 closure differs')
    candidate['files']=list(files.values());candidate['runtime_helpers']=runtime_records
    evidence={row['path']:copy.deepcopy(row) for row in parent['evidence']}
    native_evidence={row['path'] for row in native['evidence']}
    previous_raw={row['path'] for key in ('failure3272_originals','failure3276_originals','failure3281_originals','failure3281_sampling_originals') for row in parent.get(key,[])}
    for row in parent['evidence']:
        if row['path'] not in native_evidence|previous_raw:
            local=inherited_local_proof(row)
            require(local.stat().st_size==row['bytes'] and sha(local)==row['sha256'],'Historical proof bytes changed')
            sources['evidence/'+local.name]=local
    def add_local(local):
        local=Path(local);entry=dict(path=str(ROOT/'logs'/local.name),bytes=local.stat().st_size,sha256=sha(local))
        require(entry['path'] not in evidence or evidence[entry['path']]==entry,'Proof overwrite/substitution forbidden')
        evidence[entry['path']]=entry;sources['evidence/'+local.name]=local;return entry
    candidate['backend_control_proofs']={role:add_local(path) for role,path in paths.items()}
    candidate['current_driver_target_cpu_proofs']={role:add_local(path) for role,path in current_paths.items()}
    for row in spec.get('supplemental_backend_proofs',[]):add_local(bound_local(row))
    candidate['predecessor_candidate_record']=add_local(parentpath);candidate['predecessor_ready_record']=add_local(readypath)
    candidate['backend_proof_input_spec']=add_local(specpath)
    for row in runtime['proofs'].values():
        local=bound_local(row);add_local(local)
    for row in json.loads(paths['installed_backend_evidence'].read_text())['records']+json.loads(paths['installed_backend_evidence'].read_text())['transport_receipts']:
        local=bound_local(row);add_local(local)
    for path in (OUT/'CALIBRATION-DEPLOYMENT-v9.md',OUT/'CALIBRATION-DEPLOYMENT-CPU-v9.log'):add_local(path)
    candidate['evidence']=list(evidence.values())
    require(preview_config(candidate)==json.loads((OUT/'CALIBRATION-RESOLVED-CONFIG-PREVIEW-v2.json').read_text())['resolved'],
        'Backend-only controls changed common optimization/model/data resolved config')
    validate_candidate(candidate)
    candidate_bytes=(json.dumps(candidate,indent=2,allow_nan=False)+'\n').encode()
    require(len(candidate_bytes)<=12*1024*1024
        and all(path.stat().st_size<=12*1024*1024 for path in sources.values())
        and sum(path.stat().st_size for path in sources.values())+len(candidate_bytes)
            +(HERE/'test_deployment_v9.py').stat().st_size<=64*1024*1024,
        'Exact immutable package exceeds reviewed member/total bounds before any write')
    write_once(target,candidate);sources['candidate/'+target.name]=target;sources['cpu/test_deployment_v9.py']=HERE/'test_deployment_v9.py'
    make_archive(archive,sources);installer=seal_installer(candidate,sources,archive)
    write_once(receipt,dict(status='IMMUTABLE_LOCAL_DRAFT_NOT_READY_NOT_DEPLOYED',
        candidate=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)),
        archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),installer=installer,
        proof_inputs=dict(path=str(specpath),sha256=proof_inputs_sha),
        predecessor_candidate_sha256=PREDECESSOR_CANDIDATE_SHA,preserved_v8_ready_binding_sha256=PREDECESSOR_READY_SHA,
        runtime_helpers_path=str(HELPERS),controls_path=str(CONTROLS),file_count=len(files),evidence_count=len(evidence),
        package_members=sorted(sources),backend_change_scope=BACKEND_SCOPE,runtime_changed_filenames=BACKEND_CHANGED,
        inherited_native_42327_paths_sha_unchanged=True,native_directory_modified=False,source_native_acceptance=None,
        submitted_job=None,production_thresholds_selected=False,retained_training_updates_contributed=0))
    print(json.dumps(dict(candidate_sha256=sha(target),package_sha256=sha(archive),installer_sha256=installer['sha256'],ready=False,submitted=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof-inputs',required=True);parser.add_argument('--proof-inputs-sha',required=True)
    args=parser.parse_args();main(args.proof_inputs,args.proof_inputs_sha)
