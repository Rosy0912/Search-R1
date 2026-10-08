"""Root read-only actual first5 byte/state/log audit; never signs or releases."""
import datetime, hashlib, json, pathlib, types

H = pathlib.Path(__file__).resolve().parent
O = H / 'job3304-first5-complete-originals-v3'
R = O / 'raw'

def read(p):
    def pairs(items):
        d = {}
        for k, v in items:
            assert k not in d, ('duplicate JSON', k)
            d[k] = v
        return d
    return json.loads(p.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    inventory = read(O / 'inventory.json')
    assert inventory['source_commit'] == 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
    assert inventory['job_id'] == '3304'
    assert read(O / 'transport-exit.original.json')['returncode'] == 0
    assert read(H / 'job3304-collector-v3-execution-original/exit.original.json') == {
        'returncode': 0, 'waited': True, 'utc': '2026-10-06T13:28:49.592922+00:00'}
    for rel, rec in inventory['files'].items():
        p = R / rel
        assert p.is_file() and p.stat().st_size == rec['bytes']
        assert sha(p) == rec['sha256'], rel
    mp = R / 'full_state/global_step_5/manifest.json'
    manifest = read(mp)
    assert sha(mp) == '30b38d6d96abd56ff956baf8a4ba954319c62049aff8263e0c9700037e2c8577'
    assert manifest['driver_pid'] == 2548319 and manifest['job_id'] == '3304'
    assert manifest['pause_nonce'] == '20bf4e0dddc9ac6eed8b0d1a9da656ad60d99e8d8bbf07f93ef8356abdfd5764'
    run = pathlib.PurePosixPath(manifest['run_root'])
    reports = [read(R / pathlib.PurePosixPath(manifest['files'][
        'actor/training_state/rank_'+str(n)+'.json']['path']).relative_to(run)) for n in range(8)]
    native = read(R / 'evidence/first5-cpu-inspection-once-v1/native.json')
    hf = read(R / 'evidence/first5-cpu-inspection-once-v1/hf.json')
    producer = H / 'first5-1b16c2227caf/receipt_payload.py'
    assert sha(producer) == '7042f5a93158541e9ff0044bf4c21849d8de16eed159c9787fdd7ae8406c564e'
    module = types.ModuleType('root_local_readonly_receipt_checks')
    exec(compile(producer.read_bytes(), str(producer), 'exec'), module.__dict__)
    module.check_cpu_evidence(native, hf, manifest, sha(mp), reports)
    gatep = R / 'source/search_r1/baselines/checkpoint_gate.py'
    assert sha(gatep) == '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
    gate = types.ModuleType('root_local_actual_gate')
    exec(compile(gatep.read_bytes(), str(gatep), 'exec'), gate.__dict__)
    # manifest_worker_pids reads physical report paths. Supply the already
    # verified local exact original mapping, not a guessed PID list.
    workers = {report['pid'] for report in reports}
    rows, pauses, released, evals = [], [], [], []
    fsdp_source = (R / 'native-source/fully_sharded_data_parallel.py').read_bytes()
    assert hashlib.sha256(fsdp_source).hexdigest() == 'ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
    for line in (R / 'train.log').read_text().splitlines():
        for marker, dest in [('OPSD_E1_OUTER_JSON=', rows),
                             ('OPSD_E1_STEP5_PAUSED_JSON=', pauses),
                             ('OPSD_E1_STEP5_RELEASED_JSON=', released),
                             ('BASELINE_EVALUATION_METRICS_JSON=', evals)]:
            if marker in line:
                dest.append(gate.parse_log_marker(line, marker, driver_pid=manifest['driver_pid'],
                            job_id=manifest['job_id'], worker_pids=workers,
                            checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py',
                            fsdp_warning_source_bytes=fsdp_source))
    assert [r['actual_outer_updates'] for r in rows] == [1,2,3,4,5]
    assert len(pauses) == 1 and not released and len(evals) == 2
    assert pauses[0]['manifest_sha256'] == sha(mp) and pauses[0]['pause_nonce'] == manifest['pause_nonce']
    gate.validate_evaluation(evals[0], 512)
    gate.validate_evaluation(evals[1], 64)
    for row in rows:
        assert all(row[k] == manifest[k] for k in ('driver_pid','job_id','source_commit','config_sha256'))
        gate.validate_outer(row['metrics'], profile='multitraj200')
        assert row['metrics']['diagnostic/student_archived_count'] == 2560
        assert row['metrics']['diagnostic/teacher_archived_count'] == 512
        for n in range(8):
            assert row['metrics']['actor/optimizer_steps/rank_'+str(n)] == 10
    changed = [r['changed_elements_vs_original_fp32'] for r in native['ranks']]
    assert all(type(v) is dict and len(v) == 435 and
               all(type(n) is int and n >= 0 for n in v.values()) and
               sum(v.values()) > 0 for v in changed)
    assert hf['total_changed_elements'] == 2937629149
    assert native['trajectory_archive']['student_count'] == 12800
    assert native['trajectory_archive']['teacher_count'] == 2560
    assert native['trajectory_archive']['official_tokenizer_replayed'] is True
    assert native['driver']['rng_verified'] is True
    assert native['driver']['trajectory_archive_state']['completed_outer'] == 5
    for name in ('native-exit-original.json','hf-exit-original.json','srun-exit.original.json','controller-exit.original.json'):
        d = read(R / 'evidence/first5-cpu-inspection-once-v1' / name)
        assert d['returncode'] == 0, (name, d)
    result = dict(version=1, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='PASS_ROOT_ACTUAL_BYTES_NATIVE_HF_LOG_SCOPE_NOT_TOTAL_ACCEPTANCE',
        source_commit=manifest['source_commit'], job_id=manifest['job_id'], driver_pid=manifest['driver_pid'],
        manifest_sha256=sha(mp), inventory_sha256=sha(O/'inventory.json'),
        whole_raw_files=len(inventory['files']), whole_archive_members=inventory['full_archive_members'],
        full_raw_bytes=inventory['raw_bytes'], actual_outer_updates=5,
        optimizer_calls_per_rank=50, scheduler_outer_steps=5,
        all8_original_model_changed_elements=[sum(v.values()) for v in changed],
        native_sha256=sha(R/'evidence/first5-cpu-inspection-once-v1/native.json'),
        hf_sha256=sha(R/'evidence/first5-cpu-inspection-once-v1/hf.json'),
        scope='Root rehash all collected originals, exact native/HF content join, strict observed 5x8 console and true CPU waited exits; independent raw semantic/config/resource acceptance separately required. Tensor originals not downloaded; native actual content inspector evidence supplies full reads.')
    out = H/'ROOT_ACTUAL3304_FIRST5_BYTES_NATIVE_HF_LOG_AUDIT_V1.json'
    with out.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps({'path':str(out),'sha256':sha(out),'status':result['status']}))

if __name__ == '__main__':
    main()
