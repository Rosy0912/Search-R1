"""Independent local replay of immutable actual 3261 originals; no remote action."""
import copy
import datetime
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE / 'actual-snapshots/3261-20261005T203236002279Z'
FILES = SNAP / 'original-files'
RUN = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261')
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
MANIFEST_SHA = '8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3'
PIN_RE = re.compile(r'^[a-f0-9]{64}$')

def need(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def plain_digest(value):
    # Independent scalar/container implementation of the frozen state_digest.
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, dict):
            h.update(('DICT:'+str(len(v))+':').encode())
            for k in sorted(v, key=lambda k: (str(type(k)), repr(k))):
                add(k); add(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(str((type(v).__name__,len(v))).encode())
            for item in v:
                add(item)
        else:
            h.update(str((type(v).__name__,repr(v))).encode())
    add(value)
    return h.hexdigest()

manifest_path = FILES / 'full_state/global_step_5/manifest.json'
manifest = read(manifest_path)
need(sha(manifest_path) == MANIFEST_SHA, 'Manifest bytes changed')
need((manifest['source_commit'], manifest['job_id'], manifest['driver_pid'], manifest['actual_outer_updates'], manifest['next_global_step'], manifest['horizon'], manifest['warmup_steps']) == (SOURCE, '3261', 1533312, 5, 6, 200, 0), 'Actual first5 identity/budget')
summary = read(SNAP / 'observation-summary.json')
observed = read(SNAP / 'observed-remote-original.json')
spec_path = HERE / 'ACTUAL3261_FIRST5_FIVE_ROLE_EVIDENCE_SPEC.json'
spec = read(spec_path)
need(sha(spec_path) == 'bc49285b99d8894a9ee817c3d310c5e65b618891f32a9b4c928fac993c127801', 'Exact five-role spec')
need(set(spec) == {'resources','retrieval','service','rank_audit','hf_reload'}, 'Five roles only')
for role in ('retrieval','service','rank_audit','hf_reload'):
    relative = str(Path(spec[role]['path']).relative_to(RUN))
    need(sha(FILES / relative) == spec[role]['sha256'], 'Actual role bytes '+role)
    fingerprint = summary['files'][relative]
    need(fingerprint['sha256'] == spec[role]['sha256'] and fingerprint['same_inode'] and fingerprint['unchanged_during_read'] and fingerprint['before']['uid'] == fingerprint['after']['uid'] == 0, 'Unstable owned actual role '+role)

cpudir = FILES / 'evidence/first5-cpu-inspection-once-v1'
native = read(cpudir / 'native.json')
hf = read(cpudir / 'hf.json')
completed_path = cpudir / 'inspection-completed-review-required.json'
completed = read(completed_path)
need(completed['cuda_is_initialized'] is False and completed['source_commit'] == SOURCE and completed['job_id'] == '3261' and completed['manifest_sha256'] == MANIFEST_SHA, 'CPU completion identity')
for name,pin in completed['files'].items():
    need(sha(cpudir/name) == pin, 'Actual CPU original modified '+name)
exits = {}
for name in ('native-exit-original.json','hf-exit-original.json','srun-exit-original.json','controller-exit-original.json'):
    value = read(cpudir/name)
    need(type(value['returncode']) is int and value['returncode'] == 0, 'Actual waited CPU failure '+name)
    exits[name] = {'sha256':sha(cpudir/name), **value}
step = read(cpudir / 'cpu-step-original.json')
need(step['cuda_is_initialized'] is False and step['cuda_visible_devices'] == '' and step['hostname'] == 'h100-10' and step['job_id'] == '3261' and step['uid'] == 0 and '/job_3261/step_1' in step['cgroup'], 'Own-node no-CUDA inspector step')
need(exits['controller-exit-original.json']['waited_by_supervisor_pid'] == 1552987 and exits['srun-exit-original.json']['waited_by_controller_pid'] == exits['controller-exit-original.json']['controller_pid'], 'Detached actual wait chain')
for obj in (native,hf):
    need(obj['passed'] is True and obj['cuda_is_initialized'] is False and obj['version'] == 1 and obj['source_commit'] == SOURCE and obj['job_id'] == '3261' and obj['checkpoint_manifest_sha256'] == MANIFEST_SHA and obj['run_root'] == str(RUN), 'Native/HF scope identity')
need(native['scope'] == 'NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE' and hf['scope'] == 'ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA', 'Truthful CPU scope')
need(native['inspector_source_sha256'] == 'e949ebab98a49b12ac67fd6531089ef34513e276f169d9fd240c6aaccf623db4' and hf['inspector_source_sha256'] == 'dfc8ac3e4c9c539756c823ad4ba93c4de3ba7ca6ff8ef18dcd2772abdbcd1525', 'Bound actual inspectors')
need(native['hf_all_rank_slices_exact'] is True and len(native['ranks']) == 8, 'Full HF/native slice checks')
for key,rec in manifest['files'].items():
    need(native['all_files'].get(rec['path']) == rec['sha256'], 'Native full-read SHA did not close manifest '+key)
need(len(native['all_files']) == 54, 'Actual inventory count')
for path,pin in hf['base_all_files'].items():
    need(native['all_files'].get(path) == pin, 'Official-base file changed between independent inspectors')
for key,rec in hf['files'].items():
    need(manifest['files'][key] == rec and native['all_files'][rec['path']] == rec['sha256'], 'HF checkpoint file pin mismatch '+key)

global_config = read(FILES / 'full_state/global_step_5/resolved-config.json')
need(sha(FILES / 'full_state/global_step_5/resolved-config.json') == manifest['config_sha256'] == '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60', 'Resolved actual config changed')
actor = copy.deepcopy(global_config['actor_rollout_ref']['actor'])
need(actor['ppo_mini_batch_size'] == 32 and actor['ppo_micro_batch_size'] == 16, 'Global batches')
actor['ppo_mini_batch_size'] //= 8
actor['ppo_micro_batch_size'] //= 8
actor['use_remove_padding'] = global_config['actor_rollout_ref']['model']['use_remove_padding']
actor_config_digest = plain_digest(actor)
need(actor_config_digest == '27b2f92a83c0e1ddf1e0d42de82f5f232f638a9323038820b7dc5149bba59da6', 'Derived rank actual actor config')

hfmap = hf['hf_map']
need(len(hfmap) == 435 and set(hfmap) == set(read(FILES/'checkpoints/actor/global_step_5/model.safetensors.index.json')['weight_map']), 'All actual 435 HF state keys')
need(set(hf['base_map']) == set(hf['canonical_state_keys']) == set(hf['parameter_changes']) and len(hf['base_map']) == 434 and set(hfmap)-set(hf['base_map']) == {'lm_head.weight'}, 'Tied alias/canonical full parameter inventory')
need(hf['actual_unique_parameters'] == 3085938688 and hf['all_loaded_parameters_finite'] is True and hf['actual_cpu_forward_finite'] is True and hf['config_computation_equal'] is True and hf['tied_embeddings'] is True, 'Whole model CPU reload and finite forward')
need(len(hf['loading_info']) == 2 and all(x == dict(error_msgs=[],mismatched_keys=[],missing_keys=[],unexpected_keys=[]) for x in hf['loading_info']), 'Base/HF complete clean loading')
need(hf['node'] == 'h100-10' and hf['uid'] == 0 and hf['torch_version'] == native['cpu_runtime']['torch_version'] == '2.4.0+cu121', 'Actual native CPU runtime')
need(native['base_named_parameter_digest'] == hf['base_named_parameter_digest'] == 'af4d2590defbcfa8deef89518f09ed5e368aef244d043b3ee20ca28624ada159', 'Official fixed teacher full parameter digest')
need(hf['hf_named_parameter_digest'] != hf['base_named_parameter_digest'], 'No actual policy change')
need(sum(x['elements'] for x in hf['parameter_changes'].values()) == hf['actual_unique_parameters'] and sum(x['changed_elements'] for x in hf['parameter_changes'].values()) == hf['total_changed_elements'] == 3085777957, 'Whole HF change arithmetic')
for value in hf['parameter_changes'].values():
    need(type(value['changed_elements']) is int and 0 <= value['changed_elements'] <= value['elements'] and math.isfinite(value['max_abs_delta']) and math.isfinite(value['mean_abs_delta']), 'Invalid actual tensor change')

rank_results = []
saved_join = read(HERE / 'ACTUAL3261_FIRST5_SAVED_RECORD_RESOURCE_INDEPENDENT_JOIN.json')
for rank,r in enumerate(native['ranks']):
    report = read(FILES / ('checkpoints/actor/global_step_5/training_state/rank_'+str(rank)+'.json'))
    witness = saved_join['rank_joins'][rank]
    need(r['rank'] == rank and r['pid'] == report['pid'] == witness['pid'] and r['object_ids'] == report['object_ids'] == witness['object_ids'] and r['state_digest'] == report['state_digest'] and r['state_sha256'] == report['state_sha256'] == manifest['files']['actor/training_state/rank_'+str(rank)+'.pt']['sha256'], 'Native raw state/live same-object record join')
    need(r['actor_config_digest'] == actor_config_digest and r['model_dtensor_entries'] == 435 and r['model_hf_slice_exact'] is True and set(r['model_hf_slice_keys']) == set(hfmap) == set(r['model_layouts']), 'Actual rank config/HF content inventory')
    need(r['optimizer_step_values'] == [80] and r['optimizer_state_entries'] == 37 and len(r['optimizer_moments']) == 37 and r['scheduler_last_epoch'] == 5 and r['scheduler_last_lr'] == [5e-6] and r['rng_verified'] is True, 'Actual Adam/scheduler/RNG clock')
    need(len({x['parameter_id'] for x in r['optimizer_moments']}) == 37 and sum(math.prod(x['shape']) for x in r['optimizer_moments']) == 385742336, 'All unique local optimizer moments')
    for moment in r['optimizer_moments']:
        need(moment['step'] == 80 and moment['dtype'] == 'torch.float32' and len(moment['shape']) == 1 and all(PIN_RE.fullmatch(moment[k]) for k in ('exp_avg_sha256','exp_avg_sq_sha256')), 'Actual Adam moment metadata/content hashes')
    need(r['teacher_content_digest'] == report['teacher']['model_content_digest'] == hf['base_named_parameter_digest'] and r['teacher_buffer_digest'] == report['teacher']['buffer_content_digest'] == hf['base_named_buffers_digest'] and r['teacher_cuda_update_entries'] == r['teacher_counters']['cuda_update_entries'] == 5 and r['teacher_counters']['scored_tokens'] > 0, 'Actual frozen official teacher/counters')
    layouts = r['model_layouts']
    for key,layout in layouts.items():
        need(layout['mesh_device_type'] == 'cuda' and layout['mesh_ranks'] == list(range(8)) and layout['mesh_coordinate'] == [rank] and layout['mesh_shape'] == [8] and layout['placements'] == ['S(0)'] and layout['global_dtype'] == layout['local_dtype'] == 'torch.float32' and PIN_RE.fullmatch(layout['local_bytes_sha256']), 'Actual DTensor layout '+key)
        need(layout['global_shape'][0] == layout['local_shape'][0]*8 and layout['global_shape'][1:] == layout['local_shape'][1:], 'Actual DTensor shard partition '+key)
        need(math.prod(layout['local_shape']) >= r['changed_elements_vs_original_fp32'][key] >= 0, 'Actual rank tensor change arithmetic')
    need(layouts['lm_head.weight'] == layouts['model.embed_tokens.weight'], 'Actual tied HF alias slice')
    need(sum(math.prod(v['local_shape']) for k,v in layouts.items() if k != 'lm_head.weight') == r['unique_local_model_numel'] == 385742336, 'Exact unique full local model')
    changed = sum(v for k,v in r['changed_elements_vs_original_fp32'].items() if k != 'lm_head.weight')
    need(changed > 0, 'No actual parameter change on rank '+str(rank))
    rank_results.append({'rank':rank,'pid':r['pid'],'start_ticks':witness['start_ticks'],'gpu_uuid':witness['gpu_uuid'],'changed_unique_local_elements':changed,'optimizer_step_values':[80],'scheduler_last_epoch':5,'state_sha256':r['state_sha256'],'state_digest':r['state_digest'],'actor_config_digest':actor_config_digest,'all_435_HF_slices_exact':True,'same_live_object_ids':r['object_ids']})
need(sum(x['changed_unique_local_elements'] for x in rank_results) == hf['total_changed_elements'], 'Native rank changes do not reconstruct full HF change count')
embed = hf['parameter_changes']['model.embed_tokens.weight']
unchanged = embed['elements'] - embed['changed_elements']
local_embed_elements = math.prod(native['ranks'][0]['model_layouts']['model.embed_tokens.weight']['local_shape'])
need(unchanged == 115252 and local_embed_elements == 38895616 and unchanged < local_embed_elements, 'Independent global-change bound implies every rank embedding changed')
driver = native['driver']
need(driver['pid'] == 1533312 and driver['sampler_cursor'] == driver['consumed_cursor'] == 2560 and driver['size'] == 169028 and driver['epoch'] == 0 and driver['rng_verified'] is True and driver['state_sha256'] == manifest['files']['driver/driver.pt']['sha256'], 'Actual data cursor/RNG/full driver state')
driver_report = read(FILES/'full_state/global_step_5/driver.json')
need(driver_report['digest'] == driver['state_digest'] and driver_report['status'] == 'EXACT_DRIVER_STATE_RELOAD_OK', 'Actual driver same-process reload record')
need(observed['processes']['1533312']['start_ticks'] == '1829085316' and observed['processes']['1533312']['in_observed_job'], 'Actual original driver remains own process')

raw_prefix = b''.join(gzip.decompress((SNAP/'resources-prefix-original.ndjson.gz').read_bytes()).splitlines(keepends=True)[:631])
need(hashlib.sha256(raw_prefix).hexdigest() == spec['resources']['sha256'] and len(raw_prefix) == 52990694, 'Exact immutable 631 resource raw prefix')
rows = [json.loads(line) for line in raw_prefix.splitlines()]
known = set(range(1531212,1531220)) | {1531763} | {x['pid'] for x in rank_results}
seen = set()
gpu_uuids = {r['gpu_uuid'] for r in rank_results}
worker_starts = {r['pid']:r['start_ticks'] for r in rank_results}
worker_starts[1531763] = '1829074494'
for row in rows:
    need(row['job_id'] == '3261' and len(row['gpus']) == 8 and {g['uuid'] for g in row['gpus']} == gpu_uuids and not any('error' in k.lower() for k in row), 'Actual resource errors/GPU allocation changed')
    for app in row['gpu_processes']:
        need(app['pid'] in known and app['uid'] == '0' and app['in_slurm_job'] is True, 'Foreign actual GPU application')
        if app['pid'] in worker_starts:
            need(app['process_start_ticks'] == worker_starts[app['pid']], 'Reused actual GPU PID')
        seen.add(app['pid'])
need(seen == known, 'Not all actual hardware/teacher-policy/retriever applications witnessed')
times = [datetime.datetime.fromisoformat(r['time']) for r in rows]
gaps = [(b-a).total_seconds() for a,b in zip(times,times[1:])]
need(min(gaps) > 0, 'Actual resource time not monotonic')
resource = {'records':631,'raw_bytes':len(raw_prefix),'sha256':spec['resources']['sha256'],'start_utc':rows[0]['time'],'end_utc':rows[-1]['time'],'min_sample_gap_seconds':min(gaps),'max_sample_gap_seconds':max(gaps),'sampled_max_gpu_mib':max(g['used_mib'] for r in rows for g in r['gpus']),'final_sample_gpu_mib':[g['used_mib'] for g in rows[-1]['gpus']],'sampled_foreign_gpu_applications':0,'scope':'SAMPLED_PREFIX_NOT_CONTINUOUS_PEAK_OR_JOB_MEMORY_LIMIT_PROOF'}

support_files = ['ACTUAL3261_FIRST5_INDEPENDENT_LOG_REPLAY.json','ACTUAL3261_FIRST5_BEHAVIOR_INDEPENDENT_REVIEW.json','ACTUAL3261_FIRST5_SAVED_RECORD_RESOURCE_INDEPENDENT_JOIN.json','ACTUAL3261_FIRST5_WORKER_IMPORT_INDEPENDENT_JOIN.json','INDEPENDENT_SOFTWARE_REVIEW.json','INDEPENDENT_BOUND_TOOLS_REVIEW.json','ACTUAL3261_GPU_HARDWARE_INDEPENDENT_REVIEW.json','ACTUAL3261_LOCAL_GPU_RETRIEVAL_INDEPENDENT_REVIEW.json']
support = {name:sha(HERE/name) for name in support_files}
output = {'version':1,'role':'INDEPENDENT_REVIEWER','status':'PASS_ACTUAL_FIRST5_EVIDENCE','scope':'ACTUAL_FIRST5_IMPLEMENTATION_AND_STATE_ACCEPTANCE_ONLY_NOT_EFFECT_IMPROVEMENT_OR_200_COMPLETION','source_commit':SOURCE,'job_id':'3261','driver_pid':1533312,'driver_start_ticks':'1829085316','pause_nonce':manifest['pause_nonce'],'checkpoint_manifest_sha256':MANIFEST_SHA,'evidence':spec,'first5_implementation_state_acceptance':True,'effect_improvement_proven':False,'completed_outer_iterations':5,'final_200_completed':False,'new_process_restore_accepted':False,'actual_snapshot':str(SNAP),'actual_snapshot_sha256':sha(SNAP/'observed-remote-original.json'),'actual_completed_sha256':sha(completed_path),'actual_waited_exits':exits,'supporting_independent_reports':support,'full_native_manifest_files_read':len(manifest['files']),'full_native_all_files_read':len(native['all_files']),'actual_rank_results':rank_results,'actual_cpu_HF_reload':{'unique_parameters':hf['actual_unique_parameters'],'total_changed_elements':hf['total_changed_elements'],'all435_state_keys':True,'clean_base_and_HF_loading_info':True,'all_loaded_parameters_finite':True,'actual_forward_finite':True,'official_teacher_parameter_digest':hf['base_named_parameter_digest'],'HF_parameter_digest':hf['hf_named_parameter_digest'],'tokenizer_semantic_sha256':hf['tokenizer_semantic_sha256'],'per_rank_embedding_change_mathematical_bound':{'global_unchanged':unchanged,'local_elements':local_embed_elements,'every_rank_changed_lower_bound':local_embed_elements-unchanged}},'actual_driver':driver,'actual_resource_prefix':resource,'negative_effect_observations':{'train_reward_by_outer':[.158203125,.177734375,.05078125,.04296875,.052734375],'train_finished_ratio_by_outer':[.96875,.927734375,.787109375,.76171875,.80859375],'train_legal_action_ratio_by_outer':[.98974609375,.91357421875,.823893229167,.78271484375,.796223958333],'stage5_64':{'EM':.046875,'F1':.059375,'answer_tag_rate':.09375,'fully_legal_format_rate':0},'interpretation':'Early behavioral outputs are poor and worsen across distinct training batches; no paired efficacy gain is established. Initial512 and stage64 are different cohorts. Exactly two original student examples are retained, not full64 trajectories or teacher distributions.'},'limitations':['Independent native CPU inspection checks full stored contents; actual same-object roundtrip is joined to original live-worker reports and frozen executed source, not claimed as a new-process full-state restart.','Token/microbatch reduction equivalence is established by pinned source, independent math tests, and observed actual rank config/path; no separate full real-GPU gradient-equivalence experiment is claimed.','Resource proof covers exact sampled prefix; gaps prevent continuous peak or foreign-process absence claims. Cgroup memory hierarchy is not an isolated job-limit proof.','Loss is an authorized clipped OPSD surrogate; finite or negative loss does not imply true KL decreases or task performance improves.','No numeric behavioral stop threshold is introduced retrospectively, and old3007 stopping scope does not cancel new3261 authorization.','Final200 and51713 evaluation remain future work. No signature, approval publication, training release, source edit, or GPU action is performed by this reviewer.']}
path = HERE/'ACTUAL3261_FIRST5_COMPLETE_INDEPENDENT_REVIEW.json'
with path.open('x') as handle:
    json.dump(output,handle,indent=2,sort_keys=True);handle.write('\n')
print(json.dumps({'review_path':str(path),'review_sha256':sha(path),'status':output['status'],'resource':resource,'completed_sha256':sha(completed_path),'rank_changed_elements':[x['changed_unique_local_elements'] for x in rank_results]},sort_keys=True))
