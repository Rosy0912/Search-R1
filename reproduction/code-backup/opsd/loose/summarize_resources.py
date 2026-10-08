"""Stream owned job samples; report observed estimates, never exclusive RSS."""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('samples')
p.add_argument('output')
p.add_argument('--job', required=True)
p.add_argument('--expected-gpus', type=int, default=8)
a = p.parse_args()
previous = None
lifetimes = {}
gaps = []
gpu_seconds = 0.0
gpu_observed_interval_seconds = 0.0
max_rss_sum = 0
count = 0
query_failures = 0
unavailable_process_records = 0
first = last = None
with Path(a.samples).open() as f:
    for line in f:
        s = json.loads(line)
        assert str(s['job_id']) == a.job
        t = dt.datetime.fromisoformat(s['time']).timestamp()
        count += 1
        if first is None:
            first = t
        last = t
        g = {v['uuid']: float(v['util']) for v in s.get('gpus', [])}
        if g:
            assert len(g) == len(s['gpus']) == a.expected_gpus, 'GPU count or duplicate UUID'
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in g.values())
        if not g:
            query_failures += 1
        if previous is not None:
            old_t, old_g = previous
            elapsed = t - old_t
            assert elapsed >= 0
            if elapsed > 10:
                gaps.append({'after_unix_seconds': old_t, 'seconds': elapsed})
            if elapsed <= 10 and old_g and g:
                assert set(old_g) == set(g), 'GPU inventory changed'
                gpu_seconds += elapsed * sum((g[k] + old_g[k]) / 200 for k in g)
                gpu_observed_interval_seconds += elapsed
        previous = (t, g)
        unavailable_process_records += sum(bool(v.get('unavailable')) for v in s['job_processes'])
        owned = [v for v in s['job_processes'] if v.get('in_slurm_job') and not v.get('unavailable')]
        max_rss_sum = max(max_rss_sum, sum(v['rss_kib'] for v in owned))
        for v in owned:
            key = (v['pid'], v['process_start_ticks'])
            cpu = (v['cpu_user_ticks'] + v['cpu_system_ticks']) / v['clock_ticks_per_second']
            record = lifetimes.setdefault(key, {'first_cpu_seconds': cpu, 'last_cpu_seconds': cpu,
                                                'pid': v['pid'], 'start_ticks': v['process_start_ticks'],
                                                'name': v['name']})
            assert cpu >= record['last_cpu_seconds'], 'CPU ticks moved backwards'
            record['last_cpu_seconds'] = cpu
assert count > 1
for v in lifetimes.values():
    v['observed_cpu_seconds_lower_bound'] = v['last_cpu_seconds'] - v['first_cpu_seconds']
result = {
    'job_id': a.job, 'samples': count, 'first_sample_unix_seconds': first,
    'samples_missing_gpu_queries': query_failures, 'unavailable_process_records': unavailable_process_records,
    'last_sample_unix_seconds': last, 'sample_span_seconds': last - first,
    'sampled_gpu_utilization_integral_gpu_seconds_excluding_gaps': gpu_seconds,
    'gpu_integral_covered_interval_seconds': gpu_observed_interval_seconds,
    'gpu_integral_uncovered_interval_seconds': last - first - gpu_observed_interval_seconds,
    'gaps_over_10_seconds_excluded_from_integral': gaps,
    'observed_owned_process_cpu_seconds_lower_bound': sum(v['observed_cpu_seconds_lower_bound'] for v in lifetimes.values()),
    'max_observed_owned_process_rss_sum_kib_shared_pages_can_repeat': max_rss_sum,
    'process_lifetimes': list(lifetimes.values()),
    'scope': 'Whole job observations include preflight and diagnostic work. Worker CPU is not teacher-only. RSS sums can repeat shared pages; shared Slurmd cgroup memory is deliberately excluded. GPU utilization integral is a sampling estimate, not GPU reservation cost. Allocation timestamps must come from Slurm accounting separately.'
}
Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'process_lifetimes'}, indent=2))
