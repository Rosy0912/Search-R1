"""Raw action calibration deployment only. Actual native inputs remain required."""
import base64, gzip, hashlib, json, os, re
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
SOURCE='2262942142f9df8f1666ce3f1b9ebfe9c9239241'
REPO=ROOT/'repo-raw-action-2262942'
NATIVE_HELPERS=ROOT/'tools/action-prefix-native-2262942-v2'
NATIVE_CONTROLS=ROOT/'tools/action-prefix-native-2262942-control-v2'
NATIVE_BINDING=ROOT/'logs/ACTION-PREFIX-NATIVE-BINDING-2262942-v2.json'
HELPERS=ROOT/'tools/policy-safety-calibration-raw-2262942-v3'
CONTROLS=ROOT/'tools/policy-safety-calibration-raw-controls-2262942-v3'
SCOPE='GIGPO_POLICY_SAFETY_RAW_ACTION_HEALTHY25_CALIBRATION_V1'
NATIVE_SCOPE='RAW_ACTION_PREFIX_PACKING_OFFICIAL_3B_NATIVE_ONLY'
ADMISSION_SCOPE='GIGPO_POLICY_SAFETY_NATIVE_ONLY'
MANIFEST='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
DATA='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CHECKPOINT=str(ROOT/'runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25')
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
PARENT_CANDIDATE_SHA='14c97028111b9fa2b5e7ba784577067582bf3daf073be472684de24066939369'
PARENT_PHYSICAL_SHA='24b2fddf2fc3bf54ced1a4611e466f27cf572c2003ff3878de4afe54db9d2142'
RUNTIME_MANIFEST_SHA='b634544fca3ec2942658f48ce3c956099028bbce7571025c24900c74d6ad338b'
RUNTIME_REVIEW_SHA='287e902220cffba0b1f9e5516ac5de110db89cf2ea55f18525b08c27c5cda6fb'
NAMES=('deployment_contract_raw_v3.py','prepare_candidate_raw_v3.py','verify_physical_raw_v3.py','freeze_ready_raw_v3.py','submit_once_raw_v3.py')
PREFIX=ROOT/'logs/policy-safety-calibration-raw-2262942-v3'
CANDIDATE=Path(str(PREFIX)+'-candidate.json')
PHYSICAL=Path(str(PREFIX)+'-physical.json')
BINDING=Path(str(PREFIX)+'-reviewed-binding.json')
INTENT=Path(str(PREFIX)+'-submit-intent.json')
RECEIPT=Path(str(PREFIX)+'-submit-receipt.json')
PROOF_ROLES=('runtime_manifest','runtime_review','runtime_diff','runtime_cpu','raw_support_result','raw_support_review','config_preview')

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_bound(item, compressed=False):
    path = Path(item['path'])
    require(path.is_absolute() and '..' not in path.parts and path.resolve()==path and not path.is_symlink()
            and path.is_file() and sha(path) == item['sha256'], 'Missing/changed proof: ' + str(path))
    if 'bytes' in item:
        require(path.stat().st_size==item['bytes'], 'Bound proof bytes changed')
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if compressed else payload)

def write_once(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def validate_packing_native_result(result, commit):
    """Consume the real new packing-native schema without claiming restore25.

    This official-model fixture contributes no formal update and does not load
    the selected checkpoint. The calibration driver must subsequently execute
    both complete restore25 calls itself. No historical native result fields are
    inserted or inferred here.
    """
    native_scope = 'RAW_ACTION_PREFIX_PACKING_OFFICIAL_3B_NATIVE_ONLY'
    revision = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    require(result.get('status') == 'PASS' and result.get('code_commit') == commit
            and result.get('validation_scope') == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
            and result.get('native_scope') == native_scope
            and result.get('native_initialization') == 'OFFICIAL_MODEL_INITIALIZATION'
            and result.get('native_actual_restore') is False
            and type(result.get('formal_updates_contributed')) is int
            and result['formal_updates_contributed'] == 0,
            'Wrong source or official-initialization-only packing native scope')
    require(result.get('node') == 'h100-8' and str(result.get('job', '')).isdigit()
            and int(result['job']) > 0 and isinstance(result.get('binding_sha256'), str)
            and len(result['binding_sha256']) == 64
            and all(c in '0123456789abcdef' for c in result['binding_sha256']),
            'Missing actual packing native job/node/binding identity')
    require(result.get('fixtures_passed') is True and type(result.get('cpu_exit')) is int
            and result['cpu_exit'] == 0 and type(result.get('gpu_exit')) is int
            and result['gpu_exit'] == 0 and result.get('production_probability_acceptance') is False,
            'Packing native is incomplete or misrepresented as probability acceptance')
    rows = result.get('rank_records')
    require(isinstance(rows, list) and len(rows) == 8
            and {row.get('rank') for row in rows} == set(range(8)),
            'Missing actual eight-rank packing result')
    for row in rows:
        require(row.get('status') == 'PASS' and row.get('world') == 8
                and row.get('code_commit') == commit and row.get('binding_sha256') == result['binding_sha256']
                and row.get('native_scope') == native_scope
                and row.get('native_initialization') == 'OFFICIAL_MODEL_INITIALIZATION'
                and row.get('native_actual_restore') is False
                and type(row.get('formal_updates_contributed')) is int
                and row['formal_updates_contributed'] == 0
                and row.get('official_3b') is True and row.get('official_revision') == revision,
                'Packing rank/source/official initialization scope differs')
        require(all(row.get(key) is True for key in (
                    'actual_fsdp', 'flash_attention2', 'remove_padding', 'manual_offload',
                    'prefix_4096_action500_forward_backward', 'old_ref_rank4_action16_no_grad_maximum',
                    'mapping_formula_value_and_gradient_equal', 'state_save_restore_next_outer_equal',
                    'native_hf_semantic_reload_equal')),
                'Actual official3B packing/FSDP/replay evidence missing')
        require(row.get('real_sampler_executed') is False and row.get('real_fixed_reference_executed') is False
                and row.get('hf_engine_probability_acceptance') is False
                and row.get('production_thresholds_selected') is False,
                'Packing native cannot substitute real sampling/reference/calibration acceptance')
    require(result.get('cleanup_residual_states') == {} and isinstance(result.get('final_gpu'), dict)
            and result['final_gpu'].get('processes') == [],
            'Packing native terminal resources not released')
    return dict(native_scope=native_scope, native_initialization='OFFICIAL_MODEL_INITIALIZATION',
                native_actual_restore=False, formal_updates_contributed=0,
                healthy25_restore_proved=False, probability_identity_proved=False)

def validate_packing_native_review(review, result, acceptance, commit):
    """Bind an independent actual native review to its producer, never restore25."""
    scope = validate_packing_native_result(result, commit)
    require(acceptance.get('status') == 'PASS' and acceptance.get('code_commit') == commit
            and acceptance.get('native_scope') == scope['native_scope']
            and str(acceptance.get('job')) == str(result['job'])
            and acceptance.get('binding_sha256') == result['binding_sha256'],
            'Different actual packing-native acceptance identity')
    require(review.get('status') == 'PASS' and review.get('code_commit') == commit
            and review.get('native_scope') == scope['native_scope']
            and str(review.get('job')) == str(result['job'])
            and review.get('binding_sha256') == result['binding_sha256']
            and review.get('result_sha256') == acceptance['result_sha256']
            and review.get('native_initialization') == 'OFFICIAL_MODEL_INITIALIZATION'
            and review.get('native_actual_restore') is False
            and type(review.get('formal_updates_contributed')) is int
            and review['formal_updates_contributed'] == 0,
            'Independent review is not this actual official-initialization packing native')
    # The selected checkpoint/manifest are independently bound by validate_binding
    # and actually restored twice by calibration_driver.restore25. They are NOT
    # inferred from a fixture initialized from the official model.
    return scope

def validate_terminal_payload(acceptance, result, review, native_binding, terminal, physical):
    """Require saved native terminal originals, not another currently running job."""
    job = str(result['job'])
    run = str(ROOT / ('runs/gigpo-e1-action-prefix-native-job' + job))
    require(acceptance['result_path'] == str(Path(run) / 'result.json'), 'Native result not in its actual job directory')
    require(Path(acceptance['terminal_path']).parent == ROOT / 'logs'
            and '..' not in Path(acceptance['terminal_path']).parts, 'Terminal must be an isolated actual native original')
    fields = terminal['fields']
    require(fields['JobId'] == job and fields['NodeList'] == 'h100-8'
            and fields['JobState'] == 'COMPLETED' and fields['ExitCode'] == '0:0'
            and str(fields['Requeue']) == str(fields['Restarts']) == '0',
            'Native did not actually complete exactly once on h100-8 with exit0:0')
    require(acceptance['terminal_state'] == 'COMPLETED' and acceptance['terminal_exit_code'] == '0:0'
            and str(acceptance['terminal_requeue']) == str(acceptance['terminal_restarts']) == '0'
            and acceptance['cgroup_removed'] is True, 'Native acceptance specification contradicts actual clean terminal')
    require(review['terminal_path'] == acceptance['terminal_path']
            and review['terminal_sha256'] == acceptance['terminal_sha256']
            and review['terminal_state'] == 'COMPLETED' and review['terminal_exit_code'] == '0:0'
            and str(review['terminal_requeue']) == str(review['terminal_restarts']) == '0'
            and review['cgroup_removed'] is True
            and review['physical_payload_path'] == acceptance['physical_payload_path']
            and review['physical_payload_sha256'] == acceptance['physical_payload_sha256'],
            'Native independent terminal/physical/cgroup acceptance missing or contradictory')
    require(physical['status'] == 'ORIGINAL_NATIVE_TERMINAL_AND_PHYSICAL_EVIDENCE_UNREVIEWED'
            and physical['job'] == job and physical['run'] == run
            and physical['result'] == result and physical['terminal'] == terminal,
            'Actual original native physical/terminal payload identity differs')
    require(physical['own_cgroups'] == [] and physical['own_job_queue'].strip() == '',
            'Native cgroup/job queue not released')
    # An empty glob cannot distinguish unreadable ancestors from true absence.
    release = physical['cgroup_release']
    cgroup = '/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_' + job
    require('..' not in Path(result['cgroup']).parts
            and (result['cgroup'] == cgroup or result['cgroup'].startswith(cgroup + '/'))
            and release['result_cgroup'] == result['cgroup'] and release['job_id'] == job and release['path'] == cgroup
            and release['ancestor_path'] == '/sys/fs/cgroup/freezer/slurm_h100-8'
            and release['ancestor_readable'] is True and release['stat_outcome'] == 'ENOENT'
            and release['scope_stat_outcome'] == 'ENOENT' and release['removed'] is True,
            'Native actual readable-ancestor job and original scope disappearance not proved')
    require(physical['current_compute']['returncode'] == 0 and isinstance(physical['current_node_queue'], str),
            'Actual target GPU/queue observation unreadable')
    # Current other-job GPU processes are not historical native allocation residue.
    require(result['cleanup_residual_states'] == {} and result['final_gpu']['processes'] == [],
            'Saved native exclusive-allocation cleanup was incomplete')
    originals = physical['originals']
    require(len(originals) == len({r['path'] for r in originals}), 'Duplicate native original paths')
    required = {
        acceptance['result_path']: (acceptance['result_sha256'], result),
        acceptance['terminal_path']: (acceptance['terminal_sha256'], terminal),
        acceptance['native_binding_path']: (acceptance['native_binding_sha256'], native_binding),
    }
    actual = {}
    for record in originals:
        payload = base64.b64decode(record['content_base64'], validate=True)
        require(Path(record['path']).is_absolute() and len(payload) == record['bytes']
                and hashlib.sha256(payload).hexdigest() == record['sha256'], 'Native original bytes/SHA corrupt')
        actual[record['path']] = record
        if record['path'] in required:
            expected, value = required[record['path']]
            require(record['sha256'] == expected and json.loads(payload) == value,
                    'Native required original differs from separately read bound receipt')
    require(set(required).issubset(actual), 'Missing required actual native result/terminal/binding original')
    paths = physical['physical']
    require(len(paths) == len({r['path'] for r in paths}) and paths
            and all('..' not in Path(r['path']).parts and Path(r['path']).is_relative_to(run)
                    and isinstance(r['bytes'], int) and r['bytes'] >= 0
                    and isinstance(r['sha256'], str) and len(r['sha256']) == 64 for r in paths),
            'Incomplete or foreign native physical file inventory')
    result_record = [r for r in paths if r['path'] == acceptance['result_path']]
    require(result_record == [{k: actual[acceptance['result_path']][k] for k in ('path', 'bytes', 'sha256')}],
            'Native result actual physical identity does not match original')
    by_path = {r['path']: r for r in paths}
    for path, record in actual.items():
        if Path(path).is_relative_to(run):
            require(by_path.get(path) == {k: record[k] for k in ('path', 'bytes', 'sha256')},
                    'Native raw run artifact and physical inventory disagree')

def closure(records, label):
    require(isinstance(records, list) and records, 'Missing '+label)
    seen={}
    for r in records:
        require(isinstance(r,dict) and {'path','bytes','sha256'} <= set(r), 'Incomplete '+label)
        p=Path(r['path'])
        # The accepted native includes the preserved original env and Wiki/E5
        # assets outside the new experiment root. validate_inheritance binds
        # every one of these paths/bytes/SHA exactly, with only23 new root paths.
        require(p.is_absolute() and '..' not in p.parts, 'Invalid closure path')
        require(r['path'] not in seen and type(r['bytes']) is int and r['bytes']>=0
                and re.fullmatch('[0-9a-f]{64}',str(r['sha256'])), 'Duplicate/invalid '+label)
        seen[r['path']]={k:r[k] for k in ('path','bytes','sha256')}
    return seen


def match_closure(actual, wanted, label):
    require(closure(actual,label)==closure(wanted,label), 'Actual '+label+' path/bytes/SHA differs')


def validate_controls(value):
    require(value.get('code_commit')==SOURCE and value.get('repo')==str(REPO)
            and value.get('node')=='h100-8' and value.get('helpers')==str(HELPERS)
            and value.get('controls')==str(CONTROLS), 'Wrong frozen source or isolated directory/node')
    require(value.get('validation_scope')==ADMISSION_SCOPE and value.get('calibration_scope')==SCOPE
            and value.get('actor_token_identity')=='sampled_action_prefix_v1'
            and value.get('capture_actual_runner_inputs') is True
            and value.get('allowed_safety_modes')==['calibration'], 'Wrong raw diagnostic admission scope')
    require(value.get('calibration_phase')=='probability_probe_then_wait'
            and value.get('allowed_calibration_phases')==['probability_probe_then_wait'], 'Live first-Adam review cannot be bypassed')
    require(value.get('restore_step')==25 and value.get('restore_checkpoint')==CHECKPOINT
            and value.get('restore_manifest_sha256')==MANIFEST
            and value.get('common_manifest_sha256')==DATA, 'Wrong bound healthy25/common origin')
    require(value.get('calibration_actual_outer_budget')==1 and value.get('optimizer_steps_per_outer')==10
            and value.get('retained_training_updates_contributed')==0, 'Calibration budget/formal scope changed')
    require(value.get('training_submitted') is False and value.get('calibration_submitted') is False,
            'Prelaunch binding cannot claim submission')
    for k in ('target_old_policy_kl','negative_tail_limit','actor_recompute_abs_tolerance',
              'actor_vllm_abs_tolerance','actor_recompute_joint_abs_tolerance','actor_vllm_joint_abs_tolerance'):
        require(k in value and value[k] is None, 'Unestablished threshold/tolerance: '+k)
    require(value.get('probability_tolerance_evidence')==[] and value.get('production_thresholds_selected') is False
            and value.get('prefix_mismatch_limit')==value.get('action_token_mismatch_limit')==0,
            'Structural mismatch or probability calibration cannot be assumed')
    require(value.get('healthy_baseline_f1_floor')==.3231979640745351/2
            and type(value.get('probability_review_wait_seconds')) is int
            and 0<value['probability_review_wait_seconds']<=43200, 'Wrong health floor or bounded review wait')
    files=closure(value.get('files'),'files'); evidence=closure(value.get('evidence'),'evidence')
    runtime=closure(value.get('runtime_helpers'),'runtime')
    require(len(runtime)==18 and set(runtime)=={str(HELPERS/name) for name in RUNTIME_IDENTITIES},
            'All18 exact runtime helpers must be isolated')
    for name, identity in RUNTIME_IDENTITIES.items():
        path=str(HELPERS/name)
        require(runtime[path]==dict(path=path,**identity) and files.get(path)==runtime[path],
                'Frozen raw-v2 helper differs: '+name)
    require({p for p in files if p.startswith(str(CONTROLS)+'/')}=={str(CONTROLS/n) for n in NAMES}, 'Five controls not fully bound')
    require(len([p for p in files if p.startswith(str(REPO)+'/')])==480, 'Tracked source closure incomplete')
    require(value.get('native_parent_candidate',{}).get('sha256')==PARENT_CANDIDATE_SHA
            and value.get('inherited_physical',{}).get('sha256')==PARENT_PHYSICAL_SHA, 'Native physical lineage differs')
    for key in ('native_parent_candidate','inherited_physical'):
        r=value[key]; require(evidence.get(r['path'])==r, 'Inherited native origin missing from evidence')
    proofs=value.get('raw_runtime_proofs',{})
    require(set(proofs)==set(PROOF_ROLES) and len({r['path'] for r in proofs.values()})==len(proofs)
            and all(Path(r['path']).parent==ROOT/'logs/policy-safety-calibration-raw-proof-v3'
                    and evidence.get(r['path'])==r for r in proofs.values()), 'Raw proofs incomplete or substituted')
    require(proofs['runtime_manifest']['sha256']==RUNTIME_MANIFEST_SHA
            and proofs['runtime_review']['sha256']==RUNTIME_REVIEW_SHA
            and value.get('expected_config_preview')==proofs['config_preview'], 'Runtime/preview proof differs')
    for key,name,digest in [('resource_guard','resource_guard_v2.py','9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),
                            ('resource_ownership','resource_ownership_v2.py','7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed')]:
        r=value.get(key); p=str(NATIVE_HELPERS/name)
        require(r==dict(path=p,sha256=digest) and files.get(p,{}).get('sha256')==digest, 'Actual native guard/module differs')
    q=value.get('quality_selection_control',{})
    require(q==QUALITY_CONTROL and evidence.get(q['selection_path'],{}).get('sha256')==q['selection_sha256']
            and evidence.get(q['selection_review_path'],{}).get('sha256')==q['selection_review_sha256'], 'Healthy25 selection identity differs')
    return True


def validate_candidate(value):
    require(value.get('status')=='CANDIDATE_NOT_READY' and value.get('source_native_acceptance') is None,
            'Candidate cannot borrow actual native acceptance')
    return validate_controls(value)


def validate_inheritance(value, parent, physical):
    require(parent.get('status')=='CANDIDATE_NOT_READY' and parent.get('code_commit')==SOURCE
            and parent.get('native_scope')==NATIVE_SCOPE and parent.get('source_native_acceptance') is None
            and parent.get('native_actual_restore') is False and parent.get('formal_updates_contributed')==0,
            'Not exact initialized-official native candidate')
    require(physical.get('status')=='PHYSICAL_NATIVE_SOURCE_HELPER_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU'
            and physical.get('candidate_sha256')==PARENT_CANDIDATE_SHA and physical.get('code_commit')==SOURCE
            and physical.get('node')=='h100-8' and physical.get('cuda_initialized') is False,
            'Not actual new source CPU physical record')
    match_closure(physical['metadata'],parent['files'],'inherited native files')
    match_closure(physical['evidence'],parent['evidence'],'inherited native evidence')
    old=closure(parent['files'],'parent files'); fresh=closure(value['files'],'candidate files')
    require(all(fresh.get(path)==r for path,r in old.items()) and
            set(fresh)-set(old)=={str(HELPERS/n) for n in RUNTIME_IDENTITIES}|{str(CONTROLS/n) for n in NAMES},
            'Calibration can add only18runtime+5controls without altering native closure')
    oldev=closure(parent['evidence'],'parent evidence'); ev=closure(value['evidence'],'candidate evidence')
    require(all(ev.get(path)==r for path,r in oldev.items()), 'Historical native/source evidence changed')
    return True


def validate_native_payload(acceptance,result,review,binding,guard,terminal,physical):
    require(isinstance(acceptance,dict), 'New-source actual native acceptance remains null')
    validate_packing_native_review(review,result,acceptance,SOURCE)
    job=str(result['job'])
    require(job not in ('3243','3259','3267','3297')
            and acceptance.get('native_binding_path')==str(NATIVE_BINDING)
            and acceptance['native_binding_sha256']==result['binding_sha256']
            ==acceptance['binding_sha256'], 'Old native or conflicting actual binding SHA')
    require(result['guard_identity']==acceptance['guard_identity']==guard, 'Actual new native guard differs')
    require(binding.get('code_commit')==SOURCE and binding.get('helpers')==str(NATIVE_HELPERS)
            and binding.get('controls')==str(NATIVE_CONTROLS)
            and binding.get('native_scope')==NATIVE_SCOPE
            and binding.get('native_initialization')=='OFFICIAL_MODEL_INITIALIZATION'
            and binding.get('native_actual_restore') is False and binding.get('formal_updates_contributed')==0
            and binding.get('official_revision')==REVISION and binding.get('allowed_safety_modes')==['native_fixture']
            and binding.get('restore_step')==25 and binding.get('restore_checkpoint')==CHECKPOINT
            and binding.get('restore_manifest_sha256')==MANIFEST
            and {k:binding[k] for k in guard}==guard, 'Actual packing fixture binding differs')
    for row in result['rank_records']:
        require(row.get('scheduler_at_retained_boundary')==2 and row.get('adam_at_retained_boundary')==20,
                'Native same-worker saved state boundary missing')
        outer=row.get('outer_records',[])
        require(len(outer)==3 and all(r.get('optimizer_writes')==10 and r.get('gradient_micro_forwards')==160
            and r.get('before_monitor_forwards')==160 and r.get('after_monitor_forwards')==160
            and r.get('original_local_trajectory_rows')==320 and r.get('mini_denominators')==[32.]*10 for r in outer),
            'New packing native real optimizer/micro/row control missing')
        refusals=row.get('rank5_refusals',{})
        require(set(refusals)=={'temperature','output_shape'} and all(r.get('all8_refused') is True
                and r.get('extra_adam')==0 for r in refusals.values()), 'Native matching refusal/no-extraAdam missing')
    validate_terminal_payload(acceptance,result,review,binding,terminal,physical)
    return True


def read_native_acceptance(acceptance,value):
    require(isinstance(acceptance,dict), 'Actual new packing native acceptance null: launch refused')
    require(type(acceptance.get('physical_payload_compressed',True)) is bool, 'Native compression scope must be explicit bool')
    objects={}
    for key in ('result','review','native_binding','terminal','physical_payload'):
        require(isinstance(acceptance.get(key+'_path'),str) and re.fullmatch('[0-9a-f]{64}',str(acceptance.get(key+'_sha256'))),
                'Actual native proof missing: '+key)
        objects[key]=read_bound(dict(path=acceptance[key+'_path'],sha256=acceptance[key+'_sha256']),
            compressed=(key=='physical_payload' and acceptance.get('physical_payload_compressed',True)))
    guard={k:value[k] for k in ('resource_guard','resource_ownership')}
    validate_native_payload(acceptance,objects['result'],objects['review'],objects['native_binding'],guard,
                            objects['terminal'],objects['physical_payload'])
    return objects['result']


def validate_ready(value):
    validate_controls(value)
    require(value.get('status')=='REVIEWED_FOR_NATIVE_ONLY'
            and value.get('readiness')=='READY_FOR_RAW_PROBABILITY_PROBE_THEN_WAIT_ONLY', 'No independent raw calibration READY')
    acceptance=value.get('source_native_acceptance')
    require(isinstance(acceptance,dict), 'Actual new native acceptance missing')
    evidence=closure(value['evidence'],'READY evidence')
    for key in ('result','review','native_binding','terminal','physical_payload'):
        path=acceptance.get(key+'_path');digest=acceptance.get(key+'_sha256')
        require(path in evidence and evidence[path]['sha256']==digest, 'Native proof missing from full READY closure: '+key)
    read_native_acceptance(acceptance,value)
    require(type(value.get('pad_token_id')) is int and value['pad_token_id']>=0
            and type(value.get('eos_token_id')) is int and value['eos_token_id']>=0, 'Actual tokenizer PAD/EOS missing')
    return True

RUNTIME_IDENTITIES={'calibration_audit_v1.py': {'bytes': 2876, 'sha256': '9a187b94d3bbcee1886f47a438b50a573d0e3b1008b10e242ea0c23b26861d8e'}, 'calibration_contract_v1.py': {'bytes': 16749, 'sha256': 'f8cfe4d83cf9fb2ebb72dd6512cc4704987c38265aa428a7269e221797db3d29'}, 'calibration_driver_v1.py': {'bytes': 39243, 'sha256': '7e9f15887df3d86c1861df6f73025b09e1c99bd260db46de04c8fc18c1dbf2b9'}, 'calibration_run_v1.py': {'bytes': 29425, 'sha256': '5754045063c342c02f4969bd165fc2e1f892e6d772884fccb62d7dd13df5c897'}, 'calibration_state_v1.py': {'bytes': 4456, 'sha256': 'b75c96537b006a70d101a11c5f2210634f55dd7796db1b06264425eb2f738015'}, 'calibration_v1.sbatch': {'bytes': 832, 'sha256': '053d85391dd43fa0e2d191c1677ed70118943b58a3a056e9f028ca7f5b036daf'}, 'calibration_worker_v1.py': {'bytes': 18584, 'sha256': '6a26573258ca98888081f907676ae3a8a1e75677538a1fe13dce8a872a041dcd'}, 'metadata_colocation_v1.py': {'bytes': 3049, 'sha256': '39e3226c82132524e57ee1adc964f607da6a82314a784c1081c505fd10ea6b9e'}, 'owned_runtime_resume_v1.py': {'bytes': 5258, 'sha256': 'cdfbccea40f30289c0a8d90a171b7e6e3c988b05087e28f6fefe49dcda0c99c6'}, 'raw_alignment_v1.py': {'bytes': 13406, 'sha256': 'a43943129d95175db198fe43e8c93414dde9f70da4ebb2699fe11029607cea0f'}, 'raw_hf_observation_v1.py': {'bytes': 10761, 'sha256': '9f1322fb026ce8f5423abecb1acf8fbd8da2e641870738d7ff152e489ec62ff6'}, 'raw_runner_observation_v1.py': {'bytes': 11225, 'sha256': '47226bc1b77f189baa30c0c65f388205ef9cddbcbf1934189f41d4fe064c7f09'}, 'ray_audit_resume_v1.py': {'bytes': 2402, 'sha256': 'dbfcdffa710e92b7396c53df9bf7678cecc73fccdfc169ff62fe720b5b1cd4d6'}, 'safety_audit_v1.py': {'bytes': 5823, 'sha256': 'bf6f090f57080a831366c3193d0dc4b8921826c0615429dc83fa60f62f53eb51'}, 'test_calibration_audit_v1.py': {'bytes': 2581, 'sha256': 'b6944033556cebd04718e3345f3f27687e7b5f0f40782edd55bcbd11d66bdaaa'}, 'test_calibration_contract_v1.py': {'bytes': 13494, 'sha256': '5a93c329cf61bf62f15a7eef2cb72e2b25196e7267260af90eef315b990a8cb4'}, 'test_calibration_state_v1.py': {'bytes': 4902, 'sha256': 'bb618bf4e0f340b87e17e7204a3067b55adef93a6f5f25e856308176a90bf14a'}, 'test_raw_adapter_v1.py': {'bytes': 31002, 'sha256': '2755e2c3ff6721d469f81086f9cfbbef7da896c2e0a531813e5516c5e62ab5c0'}}
QUALITY_CONTROL={'enabled': True, 'selection_path': '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/logs/quality512-selection-job3232.json', 'selection_sha256': 'd1a1f6ca2b40d5ab827393003291d0fa85c0a2952308e356b58ebc4378fe0657', 'selection_review_path': '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/logs/QUALITY512-SELECTION-RESULT-INDEPENDENT-REVIEW-v1.json', 'selection_review_sha256': 'fa484387121e5f697fbc98dc2fb5f6426330668d0ff6a6fd499dba49c93adb1c'}
