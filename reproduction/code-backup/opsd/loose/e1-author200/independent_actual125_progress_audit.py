"""Independent local actual125 progress, identity and captured originals audit."""
from pathlib import Path
from types import ModuleType
from datetime import datetime, timezone
import hashlib, json, os, stat, math
B = Path(__file__).resolve().parent
S = B / 'actual-snapshots/3261-20261006T015500251151Z'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def loads(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            assert k not in result, k
            result[k] = v
        return result
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    json.dumps(value, allow_nan=False)
    return value

def read(path, pin=None):
    assert path.resolve(strict=True) == path and path.is_file()
    with path.open('rb') as f:
        before = os.fstat(f.fileno())
        assert before.st_uid == os.getuid() and stat.S_ISREG(before.st_mode)
        raw = f.read()
        after = os.fstat(f.fileno())
        fresh = path.stat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_uid, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        assert stamp(before) == stamp(after) == stamp(fresh) and len(raw) == after.st_size
    assert pin is None or sha(raw) == pin
    return raw

summary_raw = read(S / 'observation-summary.json')
s = loads(summary_raw)
assert len(s['files']) == 443
checks = []
for name, meta in s['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    raw = read(S / 'original-files' / name, meta['sha256'])
    assert len(raw) == meta['bytes']
    checks.append(dict(relative=name, sha256=sha(raw), bytes=len(raw)))
assert s['errors'] == [{'path': '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261/runtime', 'reason': 'symlink_not_followed'}]
p = B / 'review_progress_typed.py'
caller_raw = read(p, 'd9c1e303b32e74c747bf9158aefdbc5bb173d22df49eae14ab44d12a69af2eca')
m = ModuleType('independent_actual125_typed')
m.__file__ = str(p)
exec(compile(caller_raw, str(p), 'exec', dont_inherit=True), m.__dict__)
r = m.review(S)
assert r['actual_complete_outer'] == r['aggregate_complete_outer'] == 125
assert r['same_process_adam_calls_per_rank'] == 2000
root_raw = read(B / 'actual3261-progress-20261006T0155-typed.json')
assert r == loads(root_raw)
typed_path = B / 'INDEPENDENT_ACTUAL125_TYPED_PROGRESS_ORIGINAL.json'
with typed_path.open('x') as f:
    json.dump(r, f, indent=2, allow_nan=False)
    f.write('\n')
g, c, tm, tmeta = m.runtime_helpers()
ctx, _ = m.source_context(S, s, g, c, tm, tmeta)
streams = {}
for role, name in [('driver', 'ray-stdout-pid-1533312.out'), ('aggregate', 'train.log')]:
    body = m.captured_member(S, s, name)[0]
    streams[role] = c.parse_records(body[:body.rfind(b'\n') + 1], ctx)
for outer, row in enumerate(streams['driver']['outers'], 1):
    for rank in range(8):
        metrics = row['metrics']
        assert metrics['actor/optimizer_steps/rank_' + str(rank)] == 16
        assert metrics['actor/samples/rank_' + str(rank)] == 64
        assert metrics['actor/opsd_loss_finite/rank_' + str(rank)] == metrics['actor/grad_norm_finite/rank_' + str(rank)] == 1
        assert metrics['teacher/cuda_update_entries_cumulative_per_rank/rank_' + str(rank)] == outer
    assert metrics['actor/lr'] == 5e-6
coverage = c.check_evaluation_sequence(streams['driver'], 125)
coverage_a = c.check_evaluation_sequence(streams['aggregate'], 125)
assert [(e['kind'], e['step'], e['count']) for e in coverage['events']] == [('dev', 0, 512), ('stage', 5, 64), ('dev', 50, 512), ('stage', 50, 64), ('dev', 100, 512), ('stage', 100, 64)]
omit_line = lambda es: [{k: v for k, v in e.items() if k != 'console_line'} for e in es]
assert omit_line(coverage['events']) == omit_line(coverage_a['events'])
prior_path = B / 'ACTUAL100_PROGRESS_AND_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json'
prior_raw = read(prior_path, '4558b1bcd9a66e4ea20d1d8d9531db1c22d5038f1a56c9c2693a301d9692ad29')
prior = loads(prior_raw)
assert omit_line(coverage['events']) == omit_line(prior['fixed_evaluation_coverage']['events'])
outdir = 'evidence/checkpoint-125-cpu-inspection-once-v1/'
pending_names = ['native.json', 'inspection-completed-review-required.json', 'observer-exit-original.json', 'native-exit-original.json', 'srun-exit-original.json', 'controller-exit-original.json']
pending = {name: outdir + name in s['files'] for name in pending_names}
assert not any(pending.values()), pending
output = dict(version=1, status='PASS_ACTUAL125_PROGRESS_IDENTITY_AND_COPIED_ORIGINALS_NOT_CHECKPOINT_CONTENT_OR_EFFECT',
    reviewer='author200_review', utc=datetime.now(timezone.utc).isoformat(), snapshot=str(S), snapshot_finished_utc=s['finished_utc'],
    snapshot_summary_sha256=sha(summary_raw), actual_originals_full_SHA_verified=443, originals=checks,
    typed_caller_sha256=sha(caller_raw), independent_typed_result_sha256=sha(read(typed_path)), root_typed_result_sha256=sha(root_raw), exact_root_typed_result_match=True,
    source_commit=m.SOURCE, config_sha256=m.CONFIG, training_job_id='3261', driver_pid=1533312, driver_start_ticks='1829085316',
    actual_complete_outer=125, aggregate_complete_outer=125, same_process_Adam_calls_per_rank=2000,
    all1to125_8rank16_calls_64samples_finite_LR5e6_teacher_clock_checked=True, latest_metrics=r['latest_metrics'],
    observed_gpu_gib_range=[min(r['observed_gpu_used_gib']), max(r['observed_gpu_used_gib'])],
    resource_scope='Instantaneous captured same-allocation8NVML/16ownapps; no new full-history statistic or terminal resource claim.',
    existing_fixed_set_events_unchanged=coverage, prior_fixed_set_negative_effect_review=dict(path=str(prior_path), sha256=sha(prior_raw)),
    no_new_dev_or_stage125=True, CPU125_native_completion_and_waited_exits_present_at_capture=pending,
    CPU125_content_accepted=False, final200_or_terminal_or_resource_release_accepted=False,
    preserved_FSDP_frame_sha256=r['log_join']['interleaved_frames']['aggregate'][0]['record_sha256'],
    limits=['No remote, GPU, signature, training/source edit, job control or repeated inspector invocation.',
        'PureOPSD loss is clipped objective, not trueKL; logged gradient norm is pre-clipping.',
        'Train reward differs by sampled question batch; no paired improvement claim.',
        'CPU125 was incomplete at this capture. Saved metadata alone is not nativePT/HF content acceptance, live restore or wholeHF125 load.',
        'Known frozen formal terminal re-scan failure remains unchanged; actual terminal must be independently recorded.'])
report = B / 'ACTUAL125_PROGRESS_INDEPENDENT_REVIEW.json'
with report.open('x') as f:
    json.dump(output, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(dict(status=output['status'], report_sha256=sha(read(report)), latest_metrics=r['latest_metrics'], GPU_GiB=output['observed_gpu_gib_range'])))
