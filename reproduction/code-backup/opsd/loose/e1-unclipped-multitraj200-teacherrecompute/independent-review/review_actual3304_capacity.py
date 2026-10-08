"""Local-only audit of transported actual GPU probe evidence; no GPU or control."""
import base64
import datetime
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

H = pathlib.Path(__file__).resolve().parents[1]
O = H / 'job3304-snapshot-3'
REPO = H.parent / 'opsd-unclipped-multitraj200-teacherrecompute'
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
RUN = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
FIELD = 'short_native_teacher_logits_recompute_equivalence'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(first, second):
    return math.isclose(first, second, rel_tol=2e-12, abs_tol=1e-10)


raw_wire = (O / 'transport.stdout.original').read_bytes()
wire = unique(raw_wire)
need((O / 'transport.stderr.original').read_bytes() == b'', 'Transport stderr')
need(wire['job_id'] == '3304' and wire['run_root'] == RUN, 'Actual job/run')
observation = unique((O / 'actual-observation.json').read_bytes())
need({key: observation[key] for key in wire if key != 'files'} ==
     {key: value for key, value in wire.items() if key != 'files'}, 'Original observation core')
need(observation['observed_complete_driver_outer_records'] == 0 and
     observation['last_driver_record'] is None, 'No completed RL outer observed')
pins = {}
for name, record in wire['files'].items():
    raw = base64.b64decode(record['base64'], validate=True)
    need(sha(raw) == record['sha256'] and (O / name).read_bytes() == raw, 'Exact original: ' + name)
    pins[name] = record['sha256']
need(len(pins) == 7, 'Exactly seven transported snapshot originals')
aggregate = unique((O / 'evidence/capacity-micro4/micro-4.json').read_bytes())
need(aggregate['passed'] is True and type(aggregate['local_micro']) is int and
     aggregate['local_micro'] == 4 and len(aggregate['ranks']) == 8, 'Actual aggregate pass')
rawlog = (O / 'evidence/capacity-micro4/torchrun.log').read_bytes()
decoder = json.JSONDecoder()
rows = {}
offset = 0
for line_number, rawline in enumerate(rawlog.splitlines(keepends=True), 1):
    line = rawline.decode('utf-8')
    if line.startswith('{'):
        position = 0
        end_of_line = len(line.rstrip('\r\n'))
        while position < end_of_line:
            parsed, end = decoder.raw_decode(line, position)
            span = line[position:end].encode('utf-8')
            record = unique(span)
            need(record == parsed and FIELD in record, 'Exact actual rank object')
            rank = record['rank']
            need(type(rank) is int and 0 <= rank < 8 and rank not in rows, 'Unique actual rank')
            rows[rank] = (line_number, offset + len(line[:position].encode('utf-8')), sha(span), record)
            position = end
            need(position == end_of_line or line[position] == '{', 'Unknown rank JSON suffix')
    offset += len(rawline)
need(set(rows) == set(range(8)), 'Eight raw rank records, no hidden-rank inference')
source_review = unique((H / 'INDEPENDENT_SOFTWARE_REVIEW.json').read_bytes())
bound_review = unique((H / 'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.json').read_bytes())
bound = unique((H / 'runtime-tools-bound-fcecbea5d939-v5.json').read_bytes())
need(source_review['source_commit'] == bound_review['source_commit'] == bound['source_commit'] == SOURCE,
     'Reviewed actual source and V5 tools')
need(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE and
     subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip() == '',
     'Frozen local source clean')
source_pins = {}
for name in ['reproduction/baselines/e1-multitraj200/microbatch_probe.py',
             'search_r1/baselines/distillation.py']:
    raw = (REPO / name).read_bytes()
    need(sha(raw) == source_review['reviewed_source_files'][name], 'Exact reviewed producer/math')
    source_pins[name] = sha(raw)
formal_directory = pathlib.Path(bound['formal']['local_directory'])
formal_path = formal_directory / 'formal_supervisor.py'
need(sha(formal_path.read_bytes()) == bound['formal']['files']['formal_supervisor.py'], 'Frozen formal consumer')
# Only define the exact already-reviewed module under a non-main name. Its
# pure capacity validator reads the actual dictionaries; no main is called.
sys.path.insert(0, str(formal_directory))
formal = {'__name__': 'independent_actual_capacity_validator', '__file__': str(formal_path)}
exec(compile(formal_path.read_bytes(), str(formal_path), 'exec'), formal)
sys.path.pop(0)

rank_records = []
squared_stored = []
squared_recomputed = []
squared_delta = []
names = None
elements = 0
for record in aggregate['ranks']:
    rank = record['rank']
    line_number, raw_offset, raw_pin, logged = rows[rank]
    need(all(logged[key] == value for key, value in record.items()) and
         set(logged) - set(record) == {'elapsed_seconds'} and
         number(logged['elapsed_seconds']) and logged['elapsed_seconds'] > 0,
         'Raw torchrun vs aggregate exact fields and only final elapsed-time suffix')
    formal['validate_capacity_rank'](record, '3304')
    need(record['source_commit'] == SOURCE and record['job_id'] == '3304' and
         record['official_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1' and
         record['hostname'] == record['allocation']['hostname'] == 'h100-10' and
         record['allocation']['job_id'] == '3304' and
         re.search(r'(?:^|/)job_3304/step_0(?:\n|$)', record['allocation']['own_job_cgroup'], re.M) and
         type(record['rank']) is int and record['rank'] == record['local_rank'] and
         record['world_size'] == 8 and record['input_width'] == 6144 and record['response_width'] == 4096 and
         record['local_micro'] == 4 and record['response_only_logits'] is False and
         type(record['teacher_forward_micro_batch']) is int and record['teacher_forward_micro_batch'] == 4 and
         record['teacher_logits_recompute'] is True and record['component_clip'] is None and record['passed'] is True,
         'Actual full-head/full-teacher4/recompute contract')
    numeric = record[FIELD]
    statistics = numeric['parameter_statistics']
    need(type(numeric['checked_gradient_shards']) is int and
         numeric['checked_gradient_shards'] == len(statistics) == 37, '37 actual gradient shards')
    current_names = [item['name'] for item in statistics]
    need(len(set(current_names)) == 37, 'Unique actual gradient names')
    if names is None:
        names = current_names
    need(current_names == names, 'All ranks same FSDP shard inventory')
    for item in statistics:
        need(type(item['shard_elements']) is int and item['shard_elements'] > 0, 'Actual shard elements')
        elements += item['shard_elements']
        need(all(number(item[key]) and item[key] >= 0 for key in
                 ['native_stored_squared_norm', 'native_recomputed_squared_norm',
                  'difference_squared_norm', 'maximum_abs_difference']), 'Finite shard statistics')
        need(item['native_stored_squared_norm'] == item['native_recomputed_squared_norm'] and
             item['difference_squared_norm'] == item['maximum_abs_difference'] == 0,
             'Every actual captured gradient shard has exact zero delta')
    first = math.fsum(item['native_stored_squared_norm'] for item in statistics)
    second = math.fsum(item['native_recomputed_squared_norm'] for item in statistics)
    delta = math.fsum(item['difference_squared_norm'] for item in statistics)
    need(close(first, numeric['local_native_stored_squared_gradient_norm']) and
         close(second, numeric['local_native_recomputed_squared_gradient_norm']), 'Local gradient sums')
    squared_stored.append(first)
    squared_recomputed.append(second)
    squared_delta.append(delta)
    need(numeric['native_stored_full_head_loss'] == numeric['native_recomputed_full_head_loss'] and
         all(numeric[key] == 0 for key in ['loss_absolute_difference', 'loss_relative_difference',
             'shard_gradient_relative_l2', 'shard_gradient_max_abs_difference',
             'teacher_FP32_logits_max_absolute_difference', 'teacher_q_mean_probability_l1_difference',
             'teacher_q_forward_mean_probability_l1_difference', 'teacher_q_backward_mean_probability_l1_difference']),
         'Actual losses and separately recorded forward/backward q differences are zero')
    features = numeric['teacher_projection_forward_feature_sha256']
    need(features == numeric['teacher_projection_backward_feature_sha256'] and len(features) == 2 and
         all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) for value in features) and
         numeric['teacher_projection_calls'] == dict(stored_forward=2, stored_backward=0,
             recomputed_forward=2, recomputed_backward=2), 'Actual feature identities/multiplicity and phase calls')
    need((numeric['maximum_relative_gradient_l2'], numeric['minimum_gradient_cosine'],
          numeric['maximum_relative_loss'], numeric['maximum_teacher_q_mean_l1_difference'],
          numeric['maximum_cosine_roundoff']) == (.02, .9999, .002, 1e-4, 1e-12), 'Unchanged numeric gates')
    need(numeric['shard_gradient_cosine'] == 1 and numeric['shard_gradient_cosine_roundoff_applied'] is True and
         0 < numeric['shard_gradient_cosine_raw'] - 1 <= 1e-12, 'Explicit preserved raw roundoff')
    intervals = numeric['teacher_cost_intervals']
    need(len(intervals) == 2 and intervals[0]['teacher_logits_recompute'] is False and
         intervals[1]['teacher_logits_recompute'] is True and
         intervals[0]['backward_teacher_host_wall_seconds'] == intervals[0]['backward_teacher_cpu_seconds'] == 0 and
         intervals[1]['backward_teacher_host_wall_seconds'] > 0 and intervals[1]['backward_teacher_cpu_seconds'] > 0,
         'Actual recomputation backward cost is measured, not hidden')
    lp = record['old_logprob_capacity']
    need(lp['passed'] is True and lp['local_micro'] == 8 and lp['global_micro'] == 64 and
         lp['input_width'] == 6144 and lp['response_width'] == 4096 and
         lp['logprob_shape'] == lp['entropy_shape'] == [8, 4096] and
         lp['logprob_finite'] is True and lp['entropy_finite'] is True and
         lp['teacher_storage_runtime']['parameter_devices'] == ['cpu'], 'Actual LP8 packed path pass')
    active = record['teacher_active_runtime']
    need(active['parameter_dtypes'] == ['torch.float32'] and
         active['parameter_devices'] == ['cuda:' + str(rank)] and active['active_device'] == 'cuda:' + str(rank) and
         active['attention'] == 'eager' and active['matmul_tf32'] is False and active['cudnn_tf32'] is False and
         all(value == 0 for value in record['teacher_audit'].values()) and
         record['teacher_returned_runtime']['parameter_devices'] == ['cpu'], 'Frozen native CUDA teacher4, no teacher gradient or optimizer')
    need(type(record['optimizer_steps']) is int and record['optimizer_steps'] == 2 and
         [item['step'] for item in record['optimizer_step_records']] == [1, 2], 'Two actual synthetic Adam calls')
    for item in record['optimizer_step_records']:
        need(all(number(item[key]) and item[key] > 0 for key in
                 ['seconds', 'loss', 'grad_norm', 'tracked_parameter_delta']), 'Actual finite synthetic updates with changed tracked coordinate')
    need(len(record['decoder_forward_pre_hook_counts']) == 36 and
         set(record['decoder_forward_pre_hook_counts'].values()) == {4} and
         record['student_checkpoint_training_runtime'] == dict(student_wrapper_training=True,
             student_base_training=True, student_decoder_container_training=True, student_gradient_checkpointing=True,
             decoder_count=36, all_decoder_training=True, checkpoint_use_reentrant=False) and
         record['student_master_parameter_dtypes'] == ['torch.float32'] and
         record['student_mixed_precision_parameter_dtype'] == 'torch.bfloat16', 'Actual decoder backward checkpoints/precision')
    need(record['synthetic_prompt_runtime']['prompt_width'] == 2048 and
         record['synthetic_prompt_runtime']['full_input_width'] == 6144 and
         record['synthetic_prompt_runtime']['response_width'] == 4096 and
         record['synthetic_prompt_runtime']['actual_user_context_adapter_pass'] is True, 'Actual synthetic native template boundary')
    phases = record['memory_phases']
    need(all(item['num_ooms'] == item['num_alloc_retries'] == 0 for item in phases) and
         all(label in [item['label'] for item in phases] for label in ['step1_adam_done', 'step2_adam_done']),
         'Actual both Adam phases, no allocator OOM or retry')
    total = record['gpu_total_bytes']
    estimate = max(record['external_baseline_used_bytes'] + record['peak_reserved_bytes'] + 1024**3,
                   lp['estimated_peak_device_used_bytes'])
    need(estimate == record['estimated_peak_device_used_bytes'] and estimate <= .85 * total and
         record['minimum_required_free_fraction'] == .15 and
         record['headroom_estimate_includes_nccl_slack_bytes'] == 1024**3, 'Original 15% headroom formula')
    rank_records.append(dict(rank=rank, raw_line_number=line_number, raw_byte_offset=raw_offset,
        raw_record_sha256=raw_pin, checked_gradient_shards=37, local_stored_squared_norm=first,
        local_recomputed_squared_norm=second, difference_squared_norm=delta,
        raw_cosine=numeric['shard_gradient_cosine_raw'], diagnostic_cosine=1., roundoff_applied=True,
        teacher_forward_backward_feature_sha256=features, teacher_projection_calls=numeric['teacher_projection_calls'],
        teacher_q_recorded_forward_l1=0., teacher_q_recorded_backward_l1=0.,
        loss=numeric['native_stored_full_head_loss'], loss_delta=0., lp8_pass=True,
        synthetic_adam_calls=2, optimizer_step_records=record['optimizer_step_records'],
        gpu_total_bytes=total, peak_allocated_bytes=record['peak_allocated_bytes'],
        peak_reserved_bytes=record['peak_reserved_bytes'], external_baseline_used_bytes=record['external_baseline_used_bytes'],
        estimated_peak_device_used_bytes=estimate, estimated_free_fraction=1 - estimate / total,
        margin_above_15percent_bytes=.85 * total - estimate, oom_observed=False))

first = math.fsum(squared_stored)
second = math.fsum(squared_recomputed)
delta = math.fsum(squared_delta)
cosine = (first + second - delta) / (2 * math.sqrt(first * second))
relative_l2 = math.sqrt(delta / first)
need(first > 0 and first == second and delta == 0 and cosine == 1 and relative_l2 == 0, 'Independent global exact-delta arithmetic')
for _, _, _, record in rows.values():
    numeric = record[FIELD]
    need(close(first, numeric['global_native_stored_squared_gradient_norm']) and
         close(second, numeric['global_native_recomputed_squared_gradient_norm']), 'Actual allreduce global norms')
need(unique((O / 'evidence/capacity-micro4/exit.json').read_bytes()) == {'returncode': 0, 'rl_outer_updates': 0} and
     unique((O / 'evidence/hardware-exit.json').read_bytes()) == {'returncode': 0}, 'Actual GPU probe exits, zero RL count')
fields = dict(item.split('=', 1) for item in wire['live']['scontrol']['stdout'].split() if '=' in item)
need(wire['live']['scontrol']['returncode'] == 0 and wire['live']['scontrol']['stderr'] == '' and
     all(fields.get(key) == value for key, value in {'JobId': '3304', 'JobState': 'RUNNING',
         'Restarts': '0', 'NodeList': 'h100-10', 'UserId': 'root(0)'}.items()), 'Literal live own job observation')
report = dict(version=1, status='PASS_ACTUAL_GPU_CAPACITY_AND_TARGET_RECOMPUTATION_PROBE_ONLY',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), job_id='3304', source_commit=SOURCE,
    captured_utc=wire['utc'], original_snapshot_path=str(O), original_transport_sha256=sha(raw_wire),
    original_files_sha256=pins, original_member_count=7, reviewed_source_files=source_pins,
    bound_v5_software_review_sha256=sha((H / 'INDEPENDENT_BOUND_TOOLS_REVIEW_V5.json').read_bytes()),
    actual_pre_gpu_review_sha256=sha((H / 'ACTUAL_JOB3304_PRE_GPU_REVIEW.json').read_bytes()),
    actual_rank_records=rank_records,
    independently_recomputed_global=dict(stored_squared_gradient_norm=first, recomputed_squared_gradient_norm=second,
        difference_squared_norm=delta, relative_l2=relative_l2, cosine_from_norm_identity=cosine,
        raw_gpu_cosine=rank_records[0]['raw_cosine'], diagnostic_cosine=1.,
        recorded_gradient_shards=296, recorded_gradient_elements=elements, recorded_loss=1.687548041343689,
        loss_difference=0., maximum_recorded_gradient_component_difference=0.),
    actual_numeric_phase_evidence=dict(stored_forward_head_calls=2, stored_backward_head_calls=0,
        recomputed_forward_head_calls=2, recomputed_backward_head_calls=2,
        all_eight_forward_backward_feature_hash_and_multiplicity_join=True,
        recorded_FP32_logits_and_separate_q_l1_differences_zero=True,
        projection_raw_tensors_locally_recomputed=False,
        scope='Exact feature hashes, phase counts and GPU-produced logits/q summary statistics from raw rank records joined; raw logits/q tensors are not included in this shallow capsule.'),
    actual_capacity=dict(all_eight_lp8_pass=True, all_eight_worst_actor4_pass=True,
        synthetic_adam_calls_per_rank=2, synthetic_adam_calls_summed_over_ranks=16, rl_outers_counted=0,
        minimum_required_free_fraction=.15, thresholds_unchanged=True,
        conservative_used_gib_min=min(item['estimated_peak_device_used_bytes'] for item in rank_records)/1024**3,
        conservative_used_gib_max=max(item['estimated_peak_device_used_bytes'] for item in rank_records)/1024**3,
        estimated_free_fraction_min=min(item['estimated_free_fraction'] for item in rank_records),
        estimated_free_fraction_max=max(item['estimated_free_fraction'] for item in rank_records),
        narrowest_margin_above_required_bytes=min(item['margin_above_15percent_bytes'] for item in rank_records),
        scope='PyTorch peak reservation + pre-model external baseline +1GiB slack, also bounded by actual LP8 peak; not a continuous NVML residency peak or full vLLM coexistence/long-training guarantee.'),
    observed_training_scope=dict(literal_job_state='RUNNING', completed_driver_outers_observed=0,
        initialization_or_capacity_not_counted_as_training=True),
    supplemental_physical_originals_pending=True,
    limits=['All seven transported original full SHA verified; every actual eight-rank raw torchrun record joins every aggregate field, with only a positive elapsed_seconds emitted in the producer finally block. The initial reviewer whole-object equality assumption rejected this legitimate timing suffix; that reviewer script/new local rerun failure are preserved, not a probe failure. Standalone rank files, hardware rank identity and full retrieval provenance will be joined separately when collected.',
        'Each positive tracked-coordinate delta proves that rank synthetic Adam updated that coordinate; not every model parameter.',
        'Independent global reconstruction uses recorded per-shard squared norms and zero delta/max statistics; local reviewer does not rerun CUDA or reconstruct every GPU tensor.',
        'The lowest margin is barely above fixed15%; real training, vLLM/retrieval coexistence, diagnostic generation and state saving still require actual resource/fresh5 validation.',
        'This is not first5 acceptance, model/state/archive reload, scientific improvement, final200, or terminal/resource-release proof. No SSH, signing, private key, GPU, job control, source edit or production entrypoint was performed.'])
p = H / 'ACTUAL3304_GPU_RECOMPUTE_CAPACITY_INDEPENDENT_REVIEW.json'
with p.open('x') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write('\n')
md = H / 'ACTUAL3304_GPU_RECOMPUTE_CAPACITY_INDEPENDENT_REVIEW.md'
with md.open('x') as handle:
    handle.write('# Actual3304 GPU target-recomputation capacity review\n\nEight actual raw rank records match the transported aggregate exactly. All296 recorded FSDP gradient shards have zero component/delta difference; independent global squared norms are109903.3310102152 for both paths, giving relativeL2=0 and cosine1. Recorded short losses are both1.687548041343689. Separate teacher forward/backward projection counts are2/0 versus2/2, with exact feature hashes and multiplicity; recorded FP32 logits and q differences are zero. Raw cosine1.0000000000000002 is retained, with the reviewed1e-12 diagnostic-only roundoff representation.\n\nAll eight ranks pass actual LP8 and full-width actor4/input6144/response4096 with two actual synthetic Adam calls. The original15% margin remains unchanged: conservative used memory66.6704–67.2856GiB of79.1804GiB, leaving15.0224–15.7994%. Rank0 is only about0.0224 percentage point above the limit, so this is not a long-training memory guarantee. No allocator OOM or retry is recorded. Hardware and capacity exits are0; the capacity record explicitly counts zero RL outer updates.\n\nThis seven-original evidence audit proves the probe only. Supplemental standalone rank/hardware/retrieval physical provenance remains separate. Formal first5, saved HF/full state/trajectory archive evidence, same-process continuation, learning effect, final200 and resource release are not accepted here. No remote, GPU, signature, private-key, control or source mutation occurred.\n')
print(json.dumps({'status': report['status'], 'json_sha256': sha(p.read_bytes()),
    'md_sha256': sha(md.read_bytes()), 'global': report['independently_recomputed_global'],
    'memory': report['actual_capacity']}))
