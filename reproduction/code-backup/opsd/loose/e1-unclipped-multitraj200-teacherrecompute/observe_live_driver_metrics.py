"""Read-only progress observations from pinned original driver stdout."""
import argparse, base64, csv, datetime, hashlib, json, math, re, types
from pathlib import Path

H = Path(__file__).resolve().parent
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
CONFIG = '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990'
RECEIPT = 'd1db79938fd1a3b3520e4d1da673525dd260302d2402e063fd7734fcb11721fc'
STARTS = {2548974:'1834785562',2549227:'1834786108',2549228:'1834786110',2549229:'1834786111',2549231:'1834786113',2549232:'1834786114',2549230:'1834786112',2549233:'1834786115'}

def sha(raw): return hashlib.sha256(raw).hexdigest()

def strict(raw):
    def pairs(items):
        out = {}
        for key,value in items:
            if key in out: raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def proc(pid,raw,status,cgroup,start):
    assert int(raw.split(' ',1)[0]) == pid and raw.rsplit(')',1)[1].split()[19] == start
    assert re.search(r'^Uid:\s+0\s+0\s+0\s+0$',status,re.M)
    assert re.search(r'^State:\s+[RSDI]',status,re.M)
    assert '/job_3304/step_0\n' in cgroup

def observe(directory):
    exit_record = strict((directory/'exit.original.json').read_bytes())
    assert set(exit_record) == {'returncode','waited'}
    assert type(exit_record['returncode']) is int and exit_record['returncode'] == 0
    assert exit_record['waited'] is True
    wire = strict((directory/'transport.stdout.original').read_bytes())
    assert type(wire['driver_pid']) is int and wire['driver_pid'] == 2548319
    driver_path = wire['files']['driver-current-prefix.out']['physical']
    assert driver_path.startswith('/tmp/o4e-3304/ray/session_') and driver_path.endswith('-2548319.out')
    assert Path(driver_path).is_absolute() and '..' not in Path(driver_path).parts
    for name,rec in wire['files'].items():
        raw = base64.b64decode(rec['base64'],validate=True)
        assert raw == (directory/name).read_bytes() and len(raw) == rec['bytes'] and sha(raw) == rec['sha256']
    assert sha((directory/'acceptance-5-receipt.json').read_bytes()) == RECEIPT
    proc(2548319,wire['driver_stat'],wire['driver_status'],wire['driver_cgroup'],'1834784786')
    for pid,start in STARTS.items():
        v=wire['workers'][str(pid)];proc(pid,v['stat'],v['status'],v['cgroup'],start)
    for name in ['source_head','source_status','slurm','gpu','gpu_apps']:
        assert type(wire[name]['returncode']) is int and wire[name]['returncode'] == 0
    assert wire['source_head']['stdout'].strip() == SOURCE and not wire['source_status']['stdout'].strip()
    for token in ['JobId=3304 ','JobState=RUNNING','Restarts=0','NodeList=h100-10','gres:gpu:h100:8']:
        assert token in wire['slurm']['stdout']
    rr=list(csv.reader(wire['gpu']['stdout'].splitlines()));assert len(rr)==8
    gpus=[dict(index=int(r[0]),uuid=r[1].strip(),used_MiB=int(r[2]),total_MiB=int(r[3]),utilization_percent=int(r[4])) for r in rr]
    assert [r['index'] for r in gpus]==list(range(8))
    prior_bytes=(H/'ACTUAL3304_SIGNED_FIRST5_PUBLICATION_INDEPENDENT_REVIEW.json').read_bytes()
    assert sha(prior_bytes) == 'e094d37b1d2d7adf4f8ca95339d4c629b961c85e5daf3985af699a517f1f82d3'
    prior=strict(prior_bytes)
    assert {r['index']:r['uuid'] for r in gpus} == {int(k):v for k,v in prior['uuid_map'].items()}
    apps=list(csv.reader(wire['gpu_apps']['stdout'].splitlines()))
    expected={(gpus[r]['uuid'],pid) for r,pid in enumerate(STARTS)} | {(g['uuid'],2543106) for g in gpus}
    assert len(apps)==16 and {(r[0].strip(),int(r[1])) for r in apps}==expected
    source=H/'job3304-first5-complete-originals-v3/raw/source/search_r1/baselines/checkpoint_gate.py'
    code=source.read_bytes();assert sha(code)=='09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
    gate=types.ModuleType('current_pinned_gate');gate.__file__=str(source)
    exec(compile(code,str(source),'exec',dont_inherit=True),gate.__dict__)
    warnings=(H/'job3304-first5-complete-originals-v3/raw/native-source/fully_sharded_data_parallel.py').read_bytes()
    assert sha(warnings) == 'ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
    raw=(directory/'driver-current-prefix.out').read_bytes();records=[];partial=[];releases=[]
    for line in raw.splitlines(keepends=True):
        for marker,target in [('OPSD_E1_OUTER_JSON=',records),('OPSD_E1_STEP5_RELEASED_JSON=',releases)]:
            if marker.encode() not in line:continue
            try:
                x=gate.parse_log_marker(line.decode(),marker,driver_pid=2548319,job_id='3304',worker_pids=tuple(STARTS),
                    checkpoint_warning_path=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'),
                    fsdp_warning_source_bytes=warnings)
            except json.JSONDecodeError:
                assert not line.endswith(b'\n') and raw.endswith(line)
                partial.append(dict(marker=marker,bytes=len(line),sha256=sha(line)));continue
            target.append(x)
    assert releases==[dict(driver_pid=2548319,actual_outer_updates=5,next_global_step=6,receipt_sha256=RECEIPT)]
    assert 5<=len(records)<=200 and [r['actual_outer_updates'] for r in records]==list(range(1,len(records)+1))
    rows=[]
    for rec in records:
        step,m=rec['actual_outer_updates'],rec['metrics']
        assert type(step) is int and type(rec['global_step']) is int and type(rec['driver_pid']) is int
        assert rec['source_commit']==SOURCE and rec['config_sha256']==CONFIG
        assert rec['driver_pid']==2548319 and rec['job_id']=='3304' and rec['global_step']==step
        gate.validate_outer(m,profile='multitraj200')
        assert math.isclose(m['actor/lr'],min(step/57,1)*1e-6,rel_tol=1e-12)
        assert math.isclose(m['actor/optimizer_lr_used'],min((step-1)/57,1)*1e-6,rel_tol=1e-12,abs_tol=1e-18)
        assert m['diagnostic/paired_archive_complete']==1 and m['diagnostic/student_archived_count']==2560 and m['diagnostic/teacher_archived_count']==512
        for rank in range(8):assert m[f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}']==step
        legal={k:m[k] for k in ['env/ratio_of_valid_action','env/number_of_valid_action','env/number_of_valid_search']}
        rows.append(dict(outer=step,reward=m['critic/score/mean'],loss_rank0=m['actor/opsd_loss'],loss_rank_mean=sum(m[f'actor/opsd_loss_mean/rank_{r}'] for r in range(8))/8,
            finish_ratio=m['env/finish_ratio'],legal_metrics=legal,outer_seconds=m['timing_s/step'],generation_seconds=m['timing_s/gen'],update_seconds=m['timing_s/update_actor'],
            optimizer_per_rank=[m[f'actor/optimizer_steps/rank_{r}'] for r in range(8)],used_lr=m['actor/optimizer_lr_used'],post_outer_lr=m['actor/lr']))
    return dict(version=1,status='PASS_ROOT_PINNED_ACTUAL_LIVE_PROGRESS_LOG_SCOPE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_utc=wire['utc'],
        source_commit=SOURCE,config_sha256=CONFIG,job_id='3304',observer_source_sha256=sha(Path(__file__).read_bytes()),prior_signed_publication_review_sha256=sha(prior_bytes),same_original_driver_and_all8_worker_starts=True,source_clean=True,slurm_state='RUNNING',slurm_restarts=0,
        actual_completed_outer_iterations=len(rows),actual_optimizer_calls_per_rank=sum(r['optimizer_per_rank'][0] for r in rows),rows=rows,gpu_snapshot=gpus,
        incomplete_eof_records=partial,original_files={p.name:sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()},
        limits=['Root actual metrics observation; separate independent first5 and completed6 reviews remain their own scopes.',
                'Archive writer metrics are actual counts; this read does not independently replay all later archived tokens.',
                'Instantaneous NVML memory is a phase snapshot, not peak/lifetime memory.',
                'No stored checkpoint reload after5, efficacy improvement, final200, no201 lifetime or terminal/resource release claimed.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.directory.resolve(strict=True)==a.directory and a.directory.parent==H
    v=observe(a.directory)
    with a.output.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output.read_bytes()),actual_completed=v['actual_completed_outer_iterations'],latest=v['rows'][-1],gpu_snapshot=v['gpu_snapshot']),sort_keys=True))
