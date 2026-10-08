"""Read-only root review of actual driver prefixes; never controls training."""
from pathlib import Path
import base64, datetime, hashlib, json, math, re, types

H = Path(__file__).resolve().parent
D = H / 'job3304-release-live-originals-v2'
R = H / 'job3304-first5-complete-originals-v3/raw'
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
RECEIPT = 'd1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
STARTS = {2548974:'1834785562',2549227:'1834786108',2549228:'1834786110',
          2549229:'1834786111',2549231:'1834786113',2549232:'1834786114',
          2549230:'1834786112',2549233:'1834786115'}

def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError('Duplicate JSON key: '+key)
        out[key] = value
    return out

def read(path):
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def same_proc(pid, raw, status, cgroup, start):
    assert int(raw.split(' ', 1)[0]) == pid
    assert raw.rsplit(')', 1)[1].split()[19] == start
    assert re.search(r'^Uid:\s+0\s+0\s+0\s+0$', status, re.M)
    assert re.search(r'^State:\s+[RSDI]', status, re.M)
    assert '/job_3304/step_0\n' in cgroup

assert read(D/'exit.original.json') == {'returncode':0,'waited':True}
wire = read(D/'transport.stdout.original')
for name, rec in wire['files'].items():
    raw = base64.b64decode(rec['base64'], validate=True)
    assert raw == (D/name).read_bytes() and len(raw) == rec['bytes']
    assert hashlib.sha256(raw).hexdigest() == rec['sha256']
assert sha(D/'acceptance-5-receipt.json') == RECEIPT
same_proc(2548319, wire['driver_stat'], wire['driver_status'], wire['driver_cgroup'], '1834784786')
for pid, start in STARTS.items():
    v = wire['workers'][str(pid)]
    same_proc(pid, v['stat'], v['status'], v['cgroup'], start)
for name in ['source_head','source_status','slurm','gpu','gpu_apps']:
    assert wire[name]['returncode'] == 0
assert wire['source_head']['stdout'].strip() == SOURCE
assert not wire['source_status']['stdout'].strip()
for token in ['JobId=3304 ', 'JobState=RUNNING', 'Restarts=0', 'NodeList=h100-10', 'gres:gpu:h100:8']:
    assert token in wire['slurm']['stdout']
gp = R/'source/search_r1/baselines/checkpoint_gate.py'
raw = gp.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
g = types.ModuleType('actual_frozen_gate'); g.__file__ = str(gp)
exec(compile(raw, str(gp), 'exec', dont_inherit=True), g.__dict__)
warn = (R/'native-source/fully_sharded_data_parallel.py').read_bytes()
rows, releases = [], []
for line in (D/'driver-current-prefix.out').read_text().splitlines():
    for marker, target in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_RELEASED_JSON=',releases)]:
        if marker in line:
            target.append(g.parse_log_marker(line, marker, driver_pid=2548319, job_id='3304',
                worker_pids=tuple(STARTS),
                checkpoint_warning_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'),
                fsdp_warning_source_bytes=warn))
assert [x['actual_outer_updates'] for x in rows] == list(range(1,7))
assert releases == [dict(driver_pid=2548319,actual_outer_updates=5,next_global_step=6,receipt_sha256=RECEIPT)]
summary = []
for x in rows:
    assert x['source_commit'] == SOURCE and x['config_sha256'] == CONFIG
    assert x['job_id'] == '3304' and x['driver_pid'] == 2548319
    assert x['global_step'] == x['actual_outer_updates']
    g.validate_outer(x['metrics'], profile='multitraj200')
    step, m = x['actual_outer_updates'], x['metrics']
    assert math.isclose(m['actor/lr'], min(step/57,1)*1e-6,rel_tol=1e-12)
    assert math.isclose(m['actor/optimizer_lr_used'], min((step-1)/57,1)*1e-6,rel_tol=1e-12,abs_tol=1e-18)
    for rank in range(8):
        assert m[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}'] == step
    assert m['diagnostic/paired_archive_complete'] == 1
    assert m['diagnostic/student_archived_count'] == 2560
    assert m['diagnostic/teacher_archived_count'] == 512
    summary.append(dict(outer=step,reward=m['critic/score/mean'],loss_rank0=m['actor/opsd_loss'],
        loss_rank_mean=sum(m[f'actor/opsd_loss_mean/rank_{r}'] for r in range(8))/8,
        finish_ratio=m['env/finish_ratio'],outer_seconds=m['timing_s/step'],
        generation_seconds=m['timing_s/gen'],update_seconds=m['timing_s/update_actor'],
        optimizer_per_rank=[m[f'actor/optimizer_steps/rank_{r}'] for r in range(8)],
        used_lr=m['actor/optimizer_lr_used'],post_outer_lr=m['actor/lr']))
out = H/'ROOT_ACTUAL3304_COMPLETE6_CONTINUITY_REVIEW.json'
v = dict(version=1,status='PASS_ROOT_ACTUAL_COMPLETED6_SAME_PROCESS_LOG_SCOPE',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_utc=wire['utc'],
    job_id='3304',source_commit=SOURCE,config_sha256=CONFIG,signed_first5_receipt_sha256=RECEIPT,
    actual_completed_outer_iterations=6,strict_rows=summary,
    historical_first5_native_optimizer_clock_per_rank=50,
    actual_optimizer_calls_after6_per_rank=sum(x['optimizer_per_rank'][0] for x in summary),
    same_original_driver_and_all8_worker_starts=True,source_clean=True,
    original_files={p.name:sha(p) for p in D.iterdir() if p.is_file()},
    scope='Actual strict complete6 driver metrics and same original live identities only.',
    limits=['Archive completion/counts use actual writer metrics and observed paired-manifest existence here; no second full raw archive acceptance.',
            'Stored sampler cursor2560 at5 and same-process continuation are proven; this capture does not inspect a new stored sampler cursor at6.',
            'No new-process recovery, final200, terminal, resource release or efficacy improvement claim.'])
with out.open('x') as f:
    json.dump(v,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),latest=summary[-1]),sort_keys=True))
