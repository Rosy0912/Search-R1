"""Independent local-only review of captured actual pre-GPU originals; no remote/signing."""
import base64
import csv
import datetime
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess

H = pathlib.Path(__file__).resolve().parents[1]
O = H / 'job3302-pre-gpu-actual-originals'
REPO = H.parent / 'opsd-unclipped-multitraj200-nativehead'
SOURCE = 'b611864d3f6ad048189b11b9b9613854e789da5d'
RUN = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3302'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(raw):
    def pairs(rows):
        d = {}
        for k, v in rows:
            need(k not in d, 'Duplicate JSON key')
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read(name):
    p = O / name
    need(p.resolve() == p and p.is_file() and p.stat().st_uid == os.getuid(), 'Physical local original')
    return unique(p.read_bytes())


def main():
    raw = (O / 'transport.stdout.original').read_bytes()
    wire = unique(raw)
    need((O / 'transport.stderr.original').read_bytes() == b'', 'Transport stderr not empty')
    need(wire['job_id'] == '3302' and wire['run_root'] == RUN and wire['hostname'] == 'h100-10'
         and type(wire['uid']) is int and wire['uid'] == 0, 'Captured actual identity')
    pins = {}
    for name, rec in wire['files'].items():
        payload = base64.b64decode(rec['base64'], validate=True)
        need(rec['physical'] == RUN + '/' + name and type(rec['uid']) is int and rec['uid'] == 0,
             'Captured physical/UID differs: ' + name)
        need(sha(payload) == rec['sha256'] and (O / name).read_bytes() == payload,
             'Transport original bytes differ: ' + name)
        unique(payload)
        pins[name] = rec['sha256']
    need(len(pins) == 13, 'Exact 13 original files required')
    observation = read('actual-observation.json')
    need(observation == {k: v for k, v in wire.items() if k != 'files'}, 'Observation differs from raw wire')
    ready = read('approval-ready.json')
    need(pins['approval-ready.json'] == 'c8a0d7e30e2d9826bb69b643ea129c28af528e5a2e740654fc5c2679481c73a1', 'Ready pin')
    need(ready['source_commit'] == SOURCE and ready['job_id'] == '3302' and ready['run_root'] == RUN
         and type(ready['uid']) is int and ready['uid'] == 0 and type(ready['pid']) is int
         and ready['pid'] == 2475994 and type(ready['process_start_ticks']) is str
         and ready['process_start_ticks'] == '1834317113', 'Ready exact identity')
    binding = read('bindings/manifest.json')
    roles = {'allocation', 'source', 'import_closure', 'data_manifest', 'model_manifest', 'tool_manifest', 'environment'}
    need(type(binding['version']) is int and binding['version'] == 1
         and binding['source_commit'] == SOURCE and binding['run_root'] == RUN
         and set(binding['files']) == roles and pins['bindings/manifest.json'] == ready['binding_sha256'], 'Seven-role manifest')
    hashes = {}
    for role, rec in binding['files'].items():
        need(set(rec) == {'path', 'sha256'} and rec['path'] == RUN + '/bindings/' + role + '.json'
             and rec['sha256'] == pins['bindings/' + role + '.json'], 'Binding role exact bytes: ' + role)
        hashes[role] = rec['sha256']
    need(pins['bindings/public.json'] == ready['public_key_sha256']
         == 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c', 'Public-only key pin')
    public = read('bindings/public.json')
    need(set(public) == {'algorithm', 'e', 'n_hex'} and public['algorithm'] == 'RSA-SHA256'
         and type(public['e']) is int and public['e'] == 65537, 'Public key schema')
    source = read('bindings/source.json')
    reviewed = unique((H / 'INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes())
    bound = unique((H / 'runtime-tools-bound-b611864d3f6a-v3.json').read_bytes())
    need(source['clean'] is True and source['source_commit'] == SOURCE
         and source['source_files'] == reviewed['reviewed_source_files'] == bound['source_files']
         and len(source['source_files']) == 19
         and source['repo'] == bound['remote_repo']
         and source['formal_manifest_sha256'] == bound['formal']['manifest_sha256'], 'Reviewed source/formal binding')
    need(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
         and subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip() == '', 'Actual local committed source clean')
    for name, pin in source['source_files'].items():
        need(sha((REPO / name).read_bytes()) == pin and sha(subprocess.check_output(['git','-C',str(REPO),'show',SOURCE+':'+name])) == pin, 'Reviewed local source/tree changed: ' + name)
    closure = read('bindings/import_closure.json')
    need(set(closure) == {'modules'} and len(closure['modules']) == 211, 'Actual closure 211')
    project_count = 0
    for name, rec in closure['modules'].items():
        need(set(rec) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', rec['sha256']), 'Closure row schema')
        if name in ('search_r1', 'verl', 'reproduction') or name.startswith(('search_r1.', 'verl.', 'reproduction.')):
            rel = pathlib.PurePosixPath(rec['path']).relative_to(source['repo'])
            need(sha((REPO / str(rel)).read_bytes()) == rec['sha256'], 'Project closure vs committed bytes: ' + name)
            project_count += 1
    need(project_count == 207, 'Actual project closure count')
    proof = read('evidence/pre-gpu-import-check.json')
    need(proof['source_commit'] == SOURCE and proof['job_id'] == '3302' and proof['repo'] == source['repo']
         and type(proof['uid']) is int and proof['uid'] == 0 and type(proof['pid']) is int and proof['pid'] > 0
         and proof['closure_sha256'] == hashes['import_closure'] and proof['cuda_is_initialized'] is False
         and type(proof['module_count']) is int and proof['module_count'] == 84
         and type(proof['function_code_count']) is int and proof['function_code_count'] == 733
         and len(proof['actual_loaded_modules']['modules']) == 84, 'Actual CPU import proof')
    for name, rec in proof['actual_loaded_modules']['modules'].items():
        need(closure['modules'][name] == rec, 'Actual imported module vs closure: ' + name)
    allocation = read('bindings/allocation.json')
    need(allocation['hostname'] == 'h100-10' and allocation['job_id'] == '3302'
         and type(allocation['uid']) is int and allocation['uid'] == 0
         and allocation['selected_cuda_devices'] == [str(i) for i in range(8)]
         and allocation['wrapper_process'] == ready['process'], 'Actual allocation identity')
    for control in [allocation['scontrol'], wire['live']['scontrol']['stdout']]:
        fields = dict(x.split('=', 1) for x in control.split() if '=' in x)
        for k, v in {'JobId': '3302', 'JobState': 'RUNNING', 'UserId': 'root(0)', 'Restarts': '0',
                     'NodeList': 'h100-10', 'BatchHost': 'h100-10', 'NumNodes': '1', 'NumCPUs': '192',
                     'NumTasks': '1', 'OverSubscribe': 'NO', 'MinMemoryNode': '1T',
                     'TimeLimit': '7-00:00:00', 'TresPerNode': 'gres:gpu:h100:8', 'WorkDir': source['repo']}.items():
            need(fields.get(k) == v, 'Actual Slurm field: ' + k)
    need(len(allocation['nvml_double_guard']) == 2, 'Actual double idle guards')
    for guard in allocation['nvml_double_guard']:
        need(guard['applications'] == [] and len(guard['inventory']) == 8, 'Actual idle8 guards')
        for i, row in enumerate(guard['inventory']):
            need(type(row['index']) is int and row['index'] == i and row['uuid'] == ready['nvml_uuids'][i]
                 and row['name'] == 'NVIDIA H100 80GB HBM3' and row['total'] == 81559
                 and row['used'] == 0 and row['util'] == 0, 'Actual GPU guard')
    for k in ['scontrol', 'gpu_inventory', 'gpu_applications']:
        need(type(wire['live'][k]['returncode']) is int and wire['live'][k]['returncode'] == 0
             and wire['live'][k]['stderr'] == '', 'Actual live command success: ' + k)
    need(wire['live']['gpu_applications']['stdout'] == '', 'Actual third observation applications empty')
    rows = list(csv.reader(io.StringIO(wire['live']['gpu_inventory']['stdout'])))
    need(len(rows) == 8, 'Current8 NVML')
    for i, row in enumerate(rows):
        need([x.strip() for x in row] == [str(i), ready['nvml_uuids'][i], '81559', '0', '0'], 'Current NVML UUID/idle')
    proc = wire['live']['wrapper_proc']
    need(re.search(r'^Pid:\s+(\d+)', proc['status'], re.M).group(1) == str(ready['pid'])
         and re.search(r'^Uid:\s+(.*)', proc['status'], re.M).group(1).split() == ['0'] * 4
         and proc['stat'].rsplit(')', 1)[1].split()[19] == ready['process_start_ticks']
         and proc['cgroup'] == ready['process']['cgroup']
         and re.search(r'(?:^|/)job_3302/step_0(?:\n|$)', proc['cgroup'], re.M)
         and proc['cmdline'].split('\0') == ['python', str(pathlib.PurePosixPath(bound['remote_root']) / 'tools/formal-bbdb7feaf042/formal_supervisor.py'), '--prepare', ''], 'Current wrapper UID/PID/start/cgroup/argv')
    cfg = read('evidence/pre-gpu-profile.json')
    a = cfg['actor_rollout_ref']['actor']; r = cfg['actor_rollout_ref']['rollout']; d = a['distillation']
    need(cfg['baselines']['method'] == a['baseline_method'] == 'opsd'
         and cfg['baselines']['opsd_e1']['profile'] == 'multitraj200'
         and cfg['data']['shuffle_train_dataloader'] is True and 'shuffle' not in cfg['data']
         and cfg['data']['train_batch_size'] == cfg['data']['val_batch_size'] == 512
         and a['ppo_mini_batch_size'] == 256 and a['ppo_micro_batch_size'] == 32
         and r['log_prob_micro_batch_size'] == cfg['actor_rollout_ref']['ref']['log_prob_micro_batch_size'] == 64
         and a['ppo_epochs'] == 1 and a['grad_clip'] == 1
         and a['optim']['lr'] == 1e-6 and a['optim']['lr_warmup_steps'] == 57
         and a['optim']['warmup_style'] == 'constant' and a['entropy_coeff'] == 0
         and a['use_kl_loss'] is False and a['kl_loss_coef'] == cfg['algorithm']['kl_ctrl']['kl_coef'] == 0
         and d['beta'] == 0 and d['component_clip'] is None and d['temperature'] == 1
         and d['response_only_logits'] is False and d['teacher_forward_micro_batch'] == 2
         and d['teacher_prompt_mode'] == 'user_context' and d['privileged_answers'] is True
         and r['temperature'] == r['top_p'] == 1 and r['top_k'] == -1 and r['n'] == 1
         and r['n_agent'] == 5 and r['tensor_model_parallel_size'] == 1
         and cfg['trainer']['total_training_steps'] == 200 and cfg['trainer']['acceptance_step'] == 5
         and cfg['trainer']['n_gpus_per_node'] == 8 and cfg['trainer']['nnodes'] == 1,
         'Actual pure unclipped multitraj200 configuration')
    need(cfg['baselines']['opsd_e1']['run_root'] == RUN
         and cfg['baselines']['opsd_e1']['binding_manifest_sha256'] == ready['binding_sha256']
         and cfg['baselines']['opsd_e1']['public_key_sha256'] == ready['public_key_sha256']
         and cfg['actor_rollout_ref']['model']['path'] == d['teacher_path'] == cfg['baselines']['distillation']['teacher_path'], 'Actual runtime/source model paths')
    manifest = read('bindings/data_manifest.json'); model = read('bindings/model_manifest.json')
    need(hashes['data_manifest'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
         and {k: v['rows'] for k, v in manifest['splits'].items()} == {'train': 169028, 'validation': 512, 'stage': 64, 'final': 51713}, 'Common manifest identity/counts')
    need(model['repo'] == 'Qwen/Qwen2.5-3B-Instruct' and model['revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
         and len(model['files']) == 11, 'Official model manifest identity')
    tool = read('bindings/tool_manifest.json')
    need(tool['hostname'] == 'h100-10' and tool['uid'] == 0 and set(tool['results']) == {'corpus', 'e5', 'gzip_decompressed', 'index', 'wiki_gzip'}, 'Pinned same-node tool asset manifest')
    env = read('bindings/environment.json')
    need(env['values']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7', 'Actual wrapper CUDA selectors')
    packages_record = read('evidence/pre-gpu-python-environment.json')
    packages = unique(packages_record['packages'])
    expected_packages = {'click':'8.2.1','flash-attn':'2.7.4.post1','hydra-core':'1.3.2','ray':'2.40.0','torch':'2.4.0','transformers':'4.46.3','vllm':'0.6.3'}
    need(packages == expected_packages and env['python_version'].startswith('3.10.12 '), 'Actual native package versions')
    need(env['python'] == '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
         and env['values']['SEARCH_R1_REPO'] == source['repo']
         and env['values']['PYTHONPATH'] == source['repo']
         and env['values']['SEARCH_R1_BASELINE_RUN_ROOT'] == RUN
         and env['values']['RAY_ADDRESS'] is None and env['values']['RAY_TMPDIR'] is None
         and env['values']['OPSD_E1_RETRIEVER_URL'] == 'http://127.0.0.1:18084/retrieve'
         and cfg['baselines']['local_runtime_only'] is True
         and cfg['baselines']['local_runtime_host'] == 'h100-10'
         and cfg['data']['max_prompt_length'] == 4096 and cfg['data']['max_start_length'] == 2048
         and cfg['data']['max_response_length'] == cfg['data']['max_obs_length'] == 500
         and cfg['max_turns'] == 4
         and cfg['baselines']['distillation'] == d
         and cfg['baselines']['holdout']['input_mode'] == 'prebuilt_manifest'
         and cfg['baselines']['holdout']['manifest_sha256'] == hashes['data_manifest'], 'Actual paths/retrieval/lengths/teacher consistency')
    bound_review = unique((H/'INDEPENDENT_BOUND_TOOLS_REVIEW_V3.json').read_bytes())
    need(reviewed['status'] == 'PASS_INDEPENDENT_SOURCE_SOFTWARE_ONLY'
         and reviewed['source_commit'] == SOURCE
         and bound_review['status'] == 'PASS_INDEPENDENT_SOURCE_BOUND_SOFTWARE_ONLY'
         and bound_review['source_commit'] == SOURCE
         and bound_review['runtime_binding_sha256'] == sha((H/'runtime-tools-bound-b611864d3f6a-v3.json').read_bytes()), 'Existing exact source/bound software qualification')
    for bundle_role in ('formal','native','first5'):
        b = bound[bundle_role]; directory = pathlib.Path(b['local_directory'])
        need(sha((directory/b['manifest_name']).read_bytes()) == b['manifest_sha256'], 'Local actual bound manifest pin')
        for name, pin in b['files'].items():
            need(sha((directory/name).read_bytes()) == pin, 'Actual bound member pin: '+name)
    report = dict(version=1, status='ACTUAL_PRE_GPU_INDEPENDENT_REVIEW_PASS_INITIALIZATION_ONLY',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=SOURCE, job_id='3302',
                  wrapper_pid=ready['pid'], process_start_ticks=ready['process_start_ticks'], nonce=ready['nonce'],
                  ready_sha256=pins['approval-ready.json'], binding_sha256=ready['binding_sha256'],
                  public_key_sha256=ready['public_key_sha256'], uuids=ready['nvml_uuids'],
                  all_seven_binding_hashes=hashes, native_closure_sha256=hashes['import_closure'],
                  actual_profile_sha256=pins['evidence/pre-gpu-profile.json'],
                  actual_import_proof_sha256=pins['evidence/pre-gpu-import-check.json'],
                  observed_utc=wire['utc'], original_transport_sha256=sha(raw),
                  original_files_sha256=pins, native_closure_records=211, committed_project_closure_records=207,
                  native_actual_imported_modules=84, native_actual_loaded_function_codes=733,
                  actual_cpu_import_pid=proof['pid'], cuda_is_initialized=False,
                  native_python_version=env['python_version'], native_packages=packages,
                  runtime_binding_sha256=sha((H/'runtime-tools-bound-b611864d3f6a-v3.json').read_bytes()),
                  configuration_summary={'questions_per_outer': 512, 'responses_per_question': 5,
                    'trajectories_per_outer': 2560, 'optimizer_calls_expected_per_rank_per_outer': 10,
                    'actor_micro_local': 4, 'old_log_prob_micro_local': 8, 'component_clip': None,
                    'pg': False, 'reference_kl': False, 'response_only_logits': False, 'teacher_micro': 2, 'warmup_outer': 57,
                    'scheduler_initial_used_lr': 0, 'horizon': 200},
                  source_software_review_sha256=sha((H/'INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes()),
                  bound_v3_software_review_sha256=sha((H/'INDEPENDENT_BOUND_TOOLS_REVIEW_V3.json').read_bytes()),
                  actual_asset_validation_scope={
                    'model_manifest_files': 11, 'common_manifest_original_identity_verified': True,
                    'tool_manifest_completed_utc': tool['completed_utc'],
                    'fresh_validation_before_ready': 'Frozen formal prepare uses checked CPU bootstrap --mode validate --profile multitraj200 (official assets and actual prebuilt splits), then full verify_tool_assets before ready emission. Captured actual ready proves producer completed this path. Reviewer independently joined original manifests and source code, not remote large-asset contents.',
                    'not_proven_here': 'No independent local Arrow re-read, GPU model load, GPU E5/FAISS retrieval response, CUDA capacity/full-head teacher-row equivalence, optimizer step, training outer, diagnostic archive, first5 receipt, effect or final200.'},
                  scope='Actual initialization-only admission based on captured originals; root must repeat current live/full-pin guards before publishing. Existing fixture/software PASS is not substituted for actual GPU/fresh5 evidence.',
                  completed_training_outers_proven_by_this_review=0,
                  no_remote_gpu_job_signature_private_key_deployment_or_production_execution_performed=True)
    publisher = H/'publish_gpu_init-bound-b611864d3f6a-v3.py'
    namespace = {'__name__': 'independent_schema_only', '__file__': str(publisher)}
    exec(compile(publisher.read_bytes(), str(publisher), 'exec'), namespace)
    namespace['validate_review'](report, ready, pins['approval-ready.json'], '3302', pathlib.Path(RUN))
    report['frozen_publisher_validate_review_schema_pass'] = True
    report['frozen_publisher_sha256'] = sha(publisher.read_bytes())
    p = H/'ACTUAL_JOB3302_PRE_GPU_REVIEW.json'
    with p.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    md = H/'ACTUAL_JOB3302_PRE_GPU_REVIEW.md'
    with md.open('x') as f:
        f.write('# Actual job3302 pre-GPU independent review\n\nPASS: initialization only. Captured actual 13 original files match their transported full SHA and physical UID0 paths; exact seven bindings, clean sourceb611 and 211 closure/84 actual CPU imports/733 function codes join. CUDA was not initialized. Live capture shows the original PID2475994/start1834317113 in job3302 step0, exclusive h100-10 eight H100 GPUs, 192 CPUs/1T, zero restarts, and idle GPUs with no applications.\n\nActual configuration matches pure full-KL unclipped OPSD, 512 questions × 5 responses, mini256, local actor4/logprob8, warmup57 and horizon200. V3 tools retain repaired native77af sampler reader; full native student head is enabled (response_only_logits=False), teacher rows2. Native producer completes pinned model/common-data/tool asset validators before ready; the historical tool manifest is explicitly retained as historical and revalidated by that producer path. This local reviewer did not re-read remote large assets.\n\nThis admits the next actual GPU initialization/capacity checks only. It proves zero completed training outers and does not accept GPU retrieval, first5, model/state reload, effect improvement, or final200. Root must repeat the publisher live guard; no SSH, GPU, job control, signing, private-key access or deployment was performed by the reviewer.\n')
    print(json.dumps({'status': report['status'], 'json': str(p), 'json_sha256': sha(p.read_bytes()),
                      'md_sha256': sha(md.read_bytes()), 'profile': report['actual_profile_sha256'],
                      'import': report['actual_import_proof_sha256'], 'originals': len(pins)}))


if __name__ == '__main__':
    main()
