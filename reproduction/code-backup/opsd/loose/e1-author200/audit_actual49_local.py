"""Local read-only replay of the root-delivered actual49 snapshot; no remote access."""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import types

HERE = Path(__file__).resolve().parent
SNAP = HERE / 'actual-snapshots/3261-20261005T224639356135Z'
ORIG = SNAP / 'original-files'
GATE = HERE.parent / 'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
COMMON = HERE / 'final-acceptance-candidate/author200_observation.py'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def raw(path, pin=None):
    path = Path(path)
    need(path.resolve(strict=True) == path and path.is_file(), 'Physical local original required')
    before = path.stat()
    with path.open('rb') as f:
        opened = os.fstat(f.fileno())
        value = f.read()
        closed = os.fstat(f.fileno())
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(stamp(before) == stamp(opened) == stamp(closed) == stamp(after), 'Local original changed')
    digest = hashlib.sha256(value).hexdigest()
    need(pin is None or digest == pin, 'Local original SHA differs')
    return value, digest


def module(path, pin):
    value, _ = raw(path, pin)
    result = types.ModuleType('exact_local_' + path.stem)
    result.__file__ = str(path)
    exec(compile(value, str(path), 'exec', dont_inherit=True), result.__dict__)
    return result


def inspect():
    shared = module(COMMON, '29592450920eb6bc4f556138f4d0a73757bf3fc05703f7860c3b2f69c02c0295')
    gate = module(GATE, 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb')
    summary_raw, summary_sha = raw(SNAP / 'observation-summary.json')
    summary = shared.loads(summary_raw)
    _, transport_sha = raw(SNAP / 'observed-remote-original.json', summary['transport_original_sha256'])
    need(summary['job_id'] == '3261' and summary['node'] == 'h100-10'
         and summary['expected_source'] == shared.SOURCE and summary['run_root'] == str(shared.RUN), 'Snapshot binding')
    local_files = {}
    for name, rec in summary['files'].items():
        local_path = ORIG / name
        need(local_path.is_relative_to(ORIG), 'Local role escape')
        value, digest = raw(local_path, rec['sha256'])
        need(len(value) == rec['bytes'], 'Original byte length differs')
        local_files[name] = (value, digest)
    need(len(local_files) == 244, 'Delivered snapshot member count differs')
    get = lambda name: shared.loads(local_files[name][0])
    cfg = get('full_state/global_step_5/resolved-config.json')
    need(local_files['full_state/global_step_5/resolved-config.json'][1] == shared.CONFIG_SHA, 'Config pin')
    actor = cfg['actor_rollout_ref']['actor']
    need(cfg['baselines']['method'] == actor['baseline_method'] == 'opsd'
         and cfg['baselines']['opsd_e1']['profile'] == 'author200'
         and actor['use_kl_loss'] is False and actor['kl_loss_coef'] == actor['entropy_coeff'] == 0
         and cfg['algorithm']['kl_ctrl']['kl_coef'] == 0, 'Pure OPSD configuration')
    need(actor['optim']['lr'] == 5e-6 and actor['optim']['lr_warmup_steps'] == 0
         and actor['grad_clip'] == .1 and actor['ppo_mini_batch_size'] == 32
         and actor['ppo_micro_batch_size'] == 16 and actor['ppo_epochs'] == 1
         and actor['distillation']['component_clip'] == .05, 'Author200 profile')
    receipt = get('acceptance-5-receipt.json')
    need(local_files['acceptance-5-receipt.json'][1] == shared.RECEIPT5_SHA, 'Accepted receipt pin')
    gate.verify_rsa_signature(receipt['payload'], receipt['signature_hex'], get('bindings/public.json'))
    need(local_files['full_state/global_step_5/manifest.json'][1] == shared.MANIFEST5_SHA, 'First5 manifest pin')
    source = get('bindings/source.json')
    need(source['source_commit'] == shared.SOURCE and source['clean'] is True
         and source['repo'] == str(shared.REPO), 'Source binding')
    commands = summary['commands']
    need(commands['source_head']['returncode'] == 0 and commands['source_head']['stdout'].strip() == shared.SOURCE
         and commands['source_porcelain']['returncode'] == 0 and commands['source_porcelain']['stdout'] == '', 'Observed frozen source changed')
    previous_raw, previous_sha = raw(HERE / 'SOURCE_ACTUAL3261_FIRST5_LOG_SAVED_REPORT_JOIN.json',
        '2d5a4dad8067f463c0d1884de59eac2113e021f3fbd3870f9fa46f914b7213e1')
    previous = shared.loads(previous_raw)
    workers = []
    for old in previous['all_rank_saved_report_proof']:
        rank = old['rank']
        r5 = get(f'checkpoints/actor/global_step_5/training_state/rank_{rank}.json')
        r25 = get(f'checkpoints/actor/global_step_25/training_state/rank_{rank}.json')
        need(r5['pid'] == r25['pid'] == old['pid'] and r5['object_ids'] == r25['object_ids'] == old['saved_object_ids'], 'Worker object identity changed')
        need(r5['status'] == 'EXACT_FULL_STATE_RELOAD_OK' and r25['status'] == 'FULL_STATE_SERIALIZED_OK'
             and r25['scheduler_last_epoch'] == r25['step'] == 25 and r25['model_dtensor_entries'] == 435, 'Saved report clock/status')
        p = summary['processes'][str(old['pid'])]
        need(p['pid'] == old['pid'] and p['start_ticks'] == old['start_ticks']
             and p['uids'] == [0] * 4 and p['in_observed_job'] is True
             and '/job_3261/step_0' in p['cgroup'], 'Worker live identity changed')
        env = p['environment_allowlist']
        need(env['SEARCH_R1_REPO'] == str(shared.REPO) and env['SEARCH_R1_BASELINE_RUN_ROOT'] == str(shared.RUN)
             and env['OPSD_E1_RETRIEVER_URL'] == 'http://127.0.0.1:18084/retrieve', 'Worker local source/tool identity')
        workers.append(dict(rank=rank, pid=old['pid'], start_ticks=p['start_ticks'], object_ids=r25['object_ids']))
    dp = summary['processes'][str(shared.DRIVER_PID)]
    need(dp['pid'] == shared.DRIVER_PID and dp['start_ticks'] == str(shared.DRIVER_START)
         and dp['uids'] == [0] * 4 and dp['in_observed_job'] is True
         and dp['environment_allowlist']['SEARCH_R1_REPO'] == str(shared.REPO), 'Driver live identity changed')
    closure = get('bindings/import_closure.json')
    warning_path = Path(closure['modules']['torch.utils._contextlib']['path']).parent / 'checkpoint.py'
    context = dict(gate=gate, workers=workers, checkpoint_warning_path=warning_path)
    dname = 'ray-stdout-pid-1533312.out'
    for name in (dname, 'train.log'):
        rec = summary['files'][name]
        need(rec['source_offset'] == 0 and rec['complete_original_at_start'] is True
             and rec['unchanged_during_read'] is True, 'Complete stable captured log required')
    # The immutable capture may end between a JSON print and its newline.
    # Check the same complete-line boundary required by the production prefix
    # observer; retain the unaccepted tail verbatim in the original file.
    stream_boundaries = {}
    streams = {}
    for name in (dname, 'train.log'):
        captured = local_files[name][0]
        end = captured.rfind(b'\n') + 1
        need(end > 0, 'No complete observed log lines')
        prefix = captured[:end]
        tail = captured[end:]
        streams[name] = shared.parse_records(prefix, context)
        stream_boundaries[name] = dict(captured_bytes=len(captured), complete_line_prefix_bytes=end,
            prefix_sha256=hashlib.sha256(prefix).hexdigest(), unterminated_tail_bytes=len(tail),
            tail_sha256=hashlib.sha256(tail).hexdigest(), tail_used_as_completed_evidence=False)
    driver = streams[dname]
    aggregate = streams['train.log']
    join = shared.join_log_prefixes(driver, aggregate, context, 5)
    need(join['authoritative_last_outer'] == 49 and join['aggregate_last_outer'] == 48, 'Actual delivered outer count')
    evaluations = shared.check_evaluation_sequence(driver, 49)
    need(len(evaluations['events']) == 2, 'Unexpected completed50 evaluation')
    need(shared.canonical(aggregate['evaluations']) == shared.canonical(driver['evaluations']), 'Forwarded evaluation differs')
    negative_cases = []
    for name, mutation in [
        ('rank7_optimizer_wrong', lambda x: x[0]['metrics'].__setitem__('actor/optimizer_steps/rank_7', 15)),
        ('rank3_nonfinite_loss', lambda x: x[1]['metrics'].__setitem__('actor/opsd_loss_mean/rank_3', float('nan'))),
        ('teacher_outer_clock_drift', lambda x: x[2]['metrics'].__setitem__('teacher/cuda_update_entries_cumulative_per_rank/rank_2', 4)),
        ('duplicate_or_missing_outer', lambda x: x.pop(20)),
    ]:
        altered = copy.deepcopy(driver['outers']);mutation(altered)
        try:
            shared.check_outer_rows(altered, context)
        except (ValueError, RuntimeError):
            negative_cases.append(name)
        else:
            raise ValueError('Expected counterexample passed: ' + name)
    reward = [r['metrics']['critic/rewards/mean'] for r in driver['outers']]
    rows = []
    for step, row in enumerate(driver['outers'], 1):
        m = row['metrics']
        need(m['rollout/trajectories'] == 512, 'Actual trajectory count')
        need(not any(k.startswith(('actor/pg_loss', 'actor/kl_loss', 'actor/ref_kl')) for k in m), 'PG/reference KL metric appeared')
        ranks = []
        for rank in range(8):
            suffix = '/rank_' + str(rank)
            need(m['teacher/update_device_is_cuda_per_rank' + suffix] == 1, 'Teacher CPU update appeared')
            ranks.append(dict(rank=rank, optimizer_calls=m['actor/optimizer_steps' + suffix],
                samples=m['actor/samples' + suffix], loss=m['actor/opsd_loss_mean' + suffix],
                grad_norm=m['actor/grad_norm_mean' + suffix], teacher_cuda_entries=m['teacher/cuda_update_entries_cumulative_per_rank' + suffix]))
        rows.append(dict(outer=step, reward=m['critic/rewards/mean'], finish=m['env/finish_ratio'],
            legal=m['env/ratio_of_valid_action'], loss=m['actor/opsd_loss'], lr=m['actor/lr'],
            timing_s=m['timing_s/step'], ranks=ranks))
    gpu = []
    for row in summary['gpu_inventory_csv_rows']:
        need(len(row) == 6 and row[2].strip() == 'NVIDIA H100 80GB HBM3', 'Observed GPU inventory')
        gpu.append(dict(index=int(row[0]), uuid=row[1].strip(), memory_used_gib=int(row[3]) / 1024,
                        total_gib=int(row[4]) / 1024, utilization_percent=int(row[5])))
    need([g['index'] for g in gpu] == list(range(8)), 'Eight actual GPUs required')
    apps = {int(row[1]) for row in summary['unfiltered_application_csv_rows']}
    need(apps == {1531763} | {w['pid'] for w in workers}, 'Unexpected actual NVML process membership')
    scontrol = commands['scontrol']['stdout']
    need('JobState=RUNNING ' in scontrol and 'Restarts=0 ' in scontrol and 'NodeList=h100-10 ' in scontrol, 'Observed running job identity')
    return dict(version=1, scope='SOURCE_AUTHOR_READ_ONLY_ACTUAL49_REPLAY_NOT_FINAL_OR_EFFECT_ACCEPTANCE',
        inspected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), snapshot=str(SNAP),
        snapshot_finished_utc=summary['finished_utc'], snapshot_sha256=summary_sha, transport_sha256=transport_sha,
        parser_source_sha256=raw(Path(__file__))[1], gate_sha256=source['source_files']['search_r1/baselines/checkpoint_gate.py'],
        common_sha256=raw(COMMON)[1], source_commit=shared.SOURCE, config_sha256=shared.CONFIG_SHA, job_id=shared.JOB,
        driver_identity=dp, workers=workers, first5_identity_reference_sha256=previous_sha,
        all244_originals_hash_matched=True, log_join=join, stream_boundaries=stream_boundaries, real_outer_count=49,
        cumulative_optimizer_calls_each_rank=[sum(r['ranks'][rank]['optimizer_calls'] for r in rows) for rank in range(8)],
        all_rank_finite_loss_gradient_and16_updates=True, pure_opsd_source_configuration_and_metrics=True,
        observed_gpu=gpu, unfiltered_nvml_pid_membership=sorted(apps), fixed_evaluation_events=evaluations,
        checkpoint50_or_dev50_observed=False, saved_checkpoint_steps=[5,25],
        reward_summary=dict(first2_mean=statistics.mean(reward[:2]), outer3to49_mean=statistics.mean(reward[2:]),
            last10_mean=statistics.mean(reward[-10:]), latest=reward[-1], first=reward[0]),
        rows=rows, negative_parser_cases_rejected=negative_cases,
        observed_training_state='RUNNING', logging_forwarding_lag=1,
        original_logs={n: summary['files'][n] for n in (dname, 'train.log', 'retriever.log')},
        negative_findings=['Training reward remains below initial first two batches; no effect improvement demonstrated',
            'Stage5 uses64 and dev0 uses512; they are not a matched same-set effect comparison',
            'This snapshot contains no dev50/stage50 or saved50 metadata; no same512 comparison is yet possible',
            'Stored25 FULL_STATE_SERIALIZED_OK reports do not establish full native checkpoint acceptance; original CPU exit1 retained'],
        limitations=['Source author replay supports independent review but is not the final independent reviewer decision',
            'Original logs cover captured prefixes only; they do not prove future completion or terminal/resource release',
            'Per-question/raw re-scoring and current full8 checkpoint contents not inspected in this replay',
            'No SSH/GPU/signing/training control/frozen source changes'])


if __name__ == '__main__':
    result = inspect()
    out = HERE / 'SOURCE_ACTUAL3261_PROGRESS49_READ_ONLY_REVIEW.json'
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(output=str(out), sha256=raw(out)[1], actual_outer_count=result['real_outer_count'],
        log_join=result['log_join'], reward=result['reward_summary']), ensure_ascii=False))
