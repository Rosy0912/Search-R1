"""Exact local source-author replay: actual50, fixed-set validation, saved JSON only."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import types

HERE = Path(__file__).resolve().parent
BASE = HERE / 'audit_actual49_local.py'
need_pin = 'bb99edbd4f7a772c803f9d2848b9c39498a8c8daf1d33a94c94259d8a608207d'
base_raw = BASE.read_bytes()
if hashlib.sha256(base_raw).hexdigest() != need_pin:
    raise ValueError('Local immutable49 replay helper changed')
base = types.ModuleType('exact_local49_helpers')
base.__file__ = str(BASE)
exec(compile(base_raw, str(BASE), 'exec', dont_inherit=True), base.__dict__)


def inspect():
    shared = base.module(base.COMMON, '29592450920eb6bc4f556138f4d0a73757bf3fc05703f7860c3b2f69c02c0295')
    gate = base.module(base.GATE, 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb')
    snapshot = HERE / 'actual-snapshots/3261-20261005T225124138829Z'
    original = snapshot / 'original-files'
    summary_raw, summary_sha = base.raw(snapshot / 'observation-summary.json')
    summary = shared.loads(summary_raw)
    _, transport_sha = base.raw(snapshot / 'observed-remote-original.json', summary['transport_original_sha256'])
    base.need(summary['job_id'] == shared.JOB and summary['node'] == 'h100-10'
        and summary['expected_source'] == shared.SOURCE and summary['run_root'] == str(shared.RUN), 'Snapshot identity')
    files = {}
    for name, record in summary['files'].items():
        path = original / name
        base.need(path.is_relative_to(original), 'Local evidence role escape')
        value, digest = base.raw(path, record['sha256'])
        base.need(len(value) == record['bytes'], 'Local original byte length')
        files[name] = (value, digest)
    base.need(len(files) == 263, 'Snapshot original count')
    get = lambda name: shared.loads(files[name][0])
    cfg = get('full_state/global_step_5/resolved-config.json')
    base.need(files['full_state/global_step_5/resolved-config.json'][1] == shared.CONFIG_SHA
        and cfg['trainer']['total_training_steps'] == 200, 'Resolved config/horizon binding')
    actor = cfg['actor_rollout_ref']['actor']
    base.need(actor['baseline_method'] == cfg['baselines']['method'] == 'opsd'
        and actor['use_kl_loss'] is False and actor['kl_loss_coef'] == actor['entropy_coeff'] == 0
        and cfg['algorithm']['kl_ctrl']['kl_coef'] == 0
        and actor['optim']['lr'] == 5e-6 and actor['optim']['lr_warmup_steps'] == 0
        and actor['grad_clip'] == .1 and actor['ppo_mini_batch_size'] == 32
        and actor['ppo_micro_batch_size'] == 16 and actor['distillation']['component_clip'] == .05, 'Pure author200 profile')
    source = get('bindings/source.json')
    base.need(source['source_commit'] == shared.SOURCE and source['repo'] == str(shared.REPO)
        and source['clean'] is True, 'Source binding')
    commands = summary['commands']
    base.need(commands['source_head']['returncode'] == 0 and commands['source_head']['stdout'].strip() == shared.SOURCE
        and commands['source_porcelain']['returncode'] == 0 and commands['source_porcelain']['stdout'] == '', 'Observed source drift')
    receipt = get('acceptance-5-receipt.json')
    base.need(files['acceptance-5-receipt.json'][1] == shared.RECEIPT5_SHA
        and files['full_state/global_step_5/manifest.json'][1] == shared.MANIFEST5_SHA, 'Accepted5 pins')
    gate.verify_rsa_signature(receipt['payload'], receipt['signature_hex'], get('bindings/public.json'))
    previous_raw, previous_sha = base.raw(HERE / 'SOURCE_ACTUAL3261_FIRST5_LOG_SAVED_REPORT_JOIN.json',
        '2d5a4dad8067f463c0d1884de59eac2113e021f3fbd3870f9fa46f914b7213e1')
    previous = shared.loads(previous_raw)
    workers = []
    reports50 = []
    for old in previous['all_rank_saved_report_proof']:
        rank = old['rank'];name = f'checkpoints/actor/global_step_50/training_state/rank_{rank}.json'
        report = get(name);accepted = get(f'checkpoints/actor/global_step_5/training_state/rank_{rank}.json')
        base.need(report['pid'] == accepted['pid'] == old['pid']
            and report['object_ids'] == accepted['object_ids'] == old['saved_object_ids'], 'Same-live worker objects')
        for key, expected in dict(version=3, rank=rank, world_size=8, step=50, scheduler_last_epoch=50,
            model_dtensor_entries=435, optimizer_state_entries=37).items():
            base.need(type(report[key]) is int and report[key] == expected, 'Typed serialized rank50 header')
        base.need(report['status'] == 'FULL_STATE_SERIALIZED_OK' and report['algorithm'] == 'pure_opsd'
            and report['sha256'] == report['state_sha256'], 'Serialized rank50 scope')
        teacher = report['teacher'];anchor = accepted['teacher']
        for key in ['asset_directory', 'asset_files', 'model_content_digest', 'buffer_content_digest',
                    'model_config', 'caches', 'parameter_count', 'model_bytes', 'execution_device', 'storage_device']:
            base.need(shared.canonical(teacher[key]) == shared.canonical(anchor[key]), 'Teacher immutable saved metadata')
        base.need(teacher['counters']['cuda_update_entries'] == 50, 'Teacher saved outer50 clock')
        process = summary['processes'][str(old['pid'])]
        base.need(process['pid'] == old['pid'] and process['start_ticks'] == old['start_ticks']
            and process['uids'] == [0] * 4 and process['in_observed_job'] is True
            and '/job_3261/step_0' in process['cgroup'], 'Worker live identity')
        env = process['environment_allowlist']
        base.need(env['SEARCH_R1_REPO'] == str(shared.REPO) and env['SEARCH_R1_BASELINE_RUN_ROOT'] == str(shared.RUN)
            and env['OPSD_E1_RETRIEVER_URL'] == 'http://127.0.0.1:18084/retrieve', 'Worker same-node source/tool')
        workers.append(dict(rank=rank, pid=old['pid'], start_ticks=process['start_ticks'], object_ids=report['object_ids']))
        reports50.append(dict(rank=rank, report_sha256=files[name][1], original_report=report))
    driver_identity = summary['processes'][str(shared.DRIVER_PID)]
    base.need(driver_identity['pid'] == shared.DRIVER_PID and driver_identity['start_ticks'] == str(shared.DRIVER_START)
        and driver_identity['uids'] == [0]*4 and driver_identity['in_observed_job'] is True, 'Driver live identity')
    closure = get('bindings/import_closure.json')
    context = dict(gate=gate, workers=workers,
        checkpoint_warning_path=Path(closure['modules']['torch.utils._contextlib']['path']).parent / 'checkpoint.py')
    stream_names = ('ray-stdout-pid-1533312.out', 'train.log')
    streams = {};boundaries = {}
    for name in stream_names:
        rec = summary['files'][name]
        base.need(rec['source_offset'] == 0 and rec['complete_original_at_start'] is True
            and rec['unchanged_during_read'] is True, 'Stable whole captured log')
        captured = files[name][0];end = captured.rfind(b'\n') + 1
        streams[name] = shared.parse_records(captured[:end], context)
        boundaries[name] = dict(captured_sha256=files[name][1], complete_line_prefix_sha256=hashlib.sha256(captured[:end]).hexdigest(),
            captured_bytes=len(captured), complete_line_prefix_bytes=end, unaccepted_tail_bytes=len(captured)-end)
    driver = streams[stream_names[0]];aggregate = streams[stream_names[1]]
    join = shared.join_log_prefixes(driver, aggregate, context, 50)
    base.need(join['authoritative_last_outer'] == join['aggregate_last_outer'] == 50, 'Actual50 prefixes')
    evaluations = shared.check_evaluation_sequence(driver, 50)
    base.need([(x['kind'],x['step'],x['count']) for x in evaluations['events']]
        == [('dev',0,512),('stage',5,64),('dev',50,512),('stage',50,64)], 'Actual fixed evaluation schedule')
    base.need(shared.canonical(driver['evaluations']) == shared.canonical(aggregate['evaluations']), 'Evaluation streams differ')
    data = get('bindings/data_manifest.json')
    base.need(files['bindings/data_manifest.json'][1] == shared.DATA_SHA, 'Fixed common manifest')
    pairs = {}
    for kind, oldstep, newstep in [('dev',0,50),('stage',5,50)]:
        old = next(x for x in evaluations['events'] if x['kind'] == kind and x['step'] == oldstep)
        new = next(x for x in evaluations['events'] if x['kind'] == kind and x['step'] == newstep)
        base.need(old['source_counts'] == new['source_counts'] and old['count'] == new['count'], 'Fixed-set pair counts')
        pairs[kind] = dict(old_step=oldstep, new_step=newstep, count=old['count'], source_counts=old['source_counts'],
            metrics={k: dict(before=old['original_metrics'][k], after=new['original_metrics'][k],
                delta=new['original_metrics'][k]-old['original_metrics'][k]) for k in old['original_metrics']},
            same_fixed_bound_dataset_path=cfg['data']['val_files'] if kind=='dev' else cfg['baselines']['opsd_e1']['stage_files'])
    report_driver = get('full_state/global_step_50/driver.json')
    base.need(report_driver['status'] == 'DRIVER_STATE_SERIALIZED_OK' and type(report_driver['step']) is int
        and report_driver['step'] == 50 and report_driver['pid'] == shared.DRIVER_PID
        and report_driver['job_id'] == shared.JOB, 'Driver serialized50 scope')
    algorithm = get('checkpoints/algorithm/global_step_50/state.json')
    base.need(algorithm['global_step'] == 50 and algorithm['method'] == 'pure_opsd'
        and algorithm['distillation'] == actor['distillation'] and algorithm['outcome_rewards_train_policy'] is False,
        'Pure algorithm saved50')
    index = get('checkpoints/actor/global_step_50/model.safetensors.index.json')
    base.need(len(index['weight_map']) == 435 and len(set(index['weight_map'].values())) == 3, 'HF50 index metadata')
    observed_console = []
    for report in aggregate['native_console']:
        if report['step'] == 50:
            base.need(shared.canonical(report) == shared.canonical(reports50[report['rank']]['original_report']), 'Actual rank50 console/report differs')
            observed_console.append(report['rank'])
    rows = []
    for row in driver['outers']:
        m = row['metrics'];step = row['actual_outer_updates']
        base.need(m['rollout/trajectories'] == 512 and not any(k.startswith(('actor/pg_loss','actor/kl_loss','actor/ref_kl')) for k in m), 'Actual pure trajectories')
        ranks = []
        for rank in range(8):
            suf = '/rank_' + str(rank)
            base.need(m['teacher/update_device_is_cuda_per_rank'+suf] == 1, 'Teacher GPU update')
            ranks.append(dict(rank=rank, optimizer_calls=m['actor/optimizer_steps'+suf],samples=m['actor/samples'+suf],
                loss=m['actor/opsd_loss_mean'+suf],grad=m['actor/grad_norm_mean'+suf],teacher_cuda_outer=m['teacher/cuda_update_entries_cumulative_per_rank'+suf]))
        rows.append(dict(outer=step, reward=m['critic/rewards/mean'], finish=m['env/finish_ratio'], legal=m['env/ratio_of_valid_action'],
            loss=m['actor/opsd_loss'],lr=m['actor/lr'],timing_s=m['timing_s/step'],ranks=ranks))
    negative = []
    for label, key, value in [('wrong16Adam','actor/optimizer_steps/rank_7',15),
        ('nonfiniteRankLoss','actor/opsd_loss_mean/rank_3',float('nan'))]:
        altered = copy.deepcopy(driver['outers']);altered[-1]['metrics'][key] = value
        try:shared.check_outer_rows(altered,context)
        except (ValueError,RuntimeError):negative.append(label)
        else:raise ValueError('Malformed actual50 row accepted')
    bad_eval = copy.deepcopy(driver);bad_eval['evaluations'][2]['val/qa_count/all'] = 64
    try:shared.check_evaluation_sequence(bad_eval,50)
    except (ValueError,RuntimeError):negative.append('dev50AliasedToStage64')
    else:raise ValueError('Aliased dev/stage set accepted')
    gpu = [dict(index=int(x[0]), used_gib=int(x[3])/1024, utilization_percent=int(x[5])) for x in summary['gpu_inventory_csv_rows']]
    base.need([x['index'] for x in gpu] == list(range(8)), 'Actual8 NVML GPUs')
    apps = {int(x[1]) for x in summary['unfiltered_application_csv_rows']}
    base.need(apps == {1531763} | {w['pid'] for w in workers}, 'Actual unfiltered GPU membership')
    reward = [x['reward'] for x in rows]
    return dict(version=1, scope='SOURCE_AUTHOR_READ_ONLY_ACTUAL50_FIXED_SET_AND_SERIALIZED_METADATA_REVIEW_NOT_NATIVE_CONTENT_OR_FINAL200',
        inspected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot=str(snapshot),snapshot_sha256=summary_sha,
        transport_sha256=transport_sha,snapshot_finished_utc=summary['finished_utc'],source_commit=shared.SOURCE,
        config_sha256=shared.CONFIG_SHA,job_id=shared.JOB,parser_source_sha256=base.raw(Path(__file__))[1],
        local49_helper_sha256=need_pin,common_sha256=base.raw(base.COMMON)[1],gate_sha256=base.raw(base.GATE)[1],
        all263_original_hashes_matched=True,driver_identity=driver_identity,workers=workers,first5_identity_reference_sha256=previous_sha,
        log_join=join,stream_boundaries=boundaries,real_outer_count=50,
        cumulative_actual_adam_calls_each_rank=[sum(x['ranks'][rank]['optimizer_calls'] for x in rows) for rank in range(8)],
        all8rank1to50_16Adam64SamplesFiniteLossGradConstantLRTeacherCUDA=True,
        evaluations_console_json_and_outer50_exact_pair=True,evaluation_sequence=evaluations,matched_fixed_set_pairs=pairs,
        data_manifest_sha256=shared.DATA_SHA,data_manifest_original=data,driver50_original_report=report_driver,
        rank50_serialized_original_reports=reports50,algorithm50_original=algorithm,
        hf50_index_metadata=dict(weight_keys=435,shards=sorted(set(index['weight_map'].values())),sha256=files['checkpoints/actor/global_step_50/model.safetensors.index.json'][1]),
        actually_observed_rank50_console_ranks=observed_console,missing_rank_console_not_inferred_from_ray_repeat=True,
        native_full25_or50_content_accepted=False,current50_newprocess_restore_performed=False,
        latest=rows[-1],reward_summary=dict(first2_mean=statistics.mean(reward[:2]),last10_mean=statistics.mean(reward[-10:])),
        phase_gpu_sample=gpu,unfiltered_gpu_pids=sorted(apps),rows=rows,negative_regressions_rejected=negative,
        outcome='SAME512_VALIDATION_DETERIORATION_OBSERVED_WHILE_EXECUTION_CLOCKS_FINITE_AND_CONTINUOUS',
        limitations=['No per-question raw generation archive exists in ordinary validation; aggregate fixed-set/protocol comparison only',
            'Saved50 statuses stay SERIALIZED; neither raw current rankPT/Adam/HF tensor content nor new-process recovery inspected',
            'Original CPU25 failure remains retained and has not been upgraded by this replay',
            'Training reward batches differ; they are descriptive and not the same-question paired estimate',
            'GPU memory is an instantaneous phase sample, not a fixed training peak',
            'No source188/runtime/GPU/SSH/signing/training control/terminal200 actions'])


if __name__ == '__main__':
    result = inspect()
    out = HERE / 'SOURCE_ACTUAL3261_PROGRESS50_FIXED_SET_READ_ONLY_REVIEW.json'
    with out.open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(output=str(out),sha256=base.raw(out)[1],actual_outer=50,
        dev512={k:result['matched_fixed_set_pairs']['dev']['metrics'][k] for k in ('val/em/all','val/f1/all','val/format_rate/all')})))
