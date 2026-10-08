"""Local read-only progress evidence join. Never polls, controls or rewrites training."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import types

HERE = Path(__file__).resolve().parent
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
RUN = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261')
WORKERS = [1533910, 1534229, 1534230, 1534231, 1534232, 1534233, 1534234, 1534235]
STARTS = ['1829086072', '1829086582', '1829086583', '1829086584', '1829086585',
          '1829086586', '1829086587', '1829086588']
RECEIPT = '812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b'
MANIFEST5 = '8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3'
PUBLIC = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
CLOSURE = '3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
GATE = HERE.parent / 'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
GATE_SHA = 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb'
TOOLS = HERE / 'final-acceptance-90c2695a7f46'
TOOLS_SHA = '90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
COMMON_SHA = '9360f3e80b8525a7291c649fe60f38a4e283b087db4baa482276447915055b5e'
FSDP_PACKET = HERE / 'actual-fsdp-warning-source-20261005T235824197037Z'
FSDP_SHA = 'ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3'
FSDP_SUMMARY = '45dc1263d876c9614bd8015742d6a97322601fd48aa663330b7687f05e717ddc'
FSDP_REVIEW = '59b4526812a5d56fd43f9b57b70971e8d6a8bbba47b2186bd6edd9f13232de52'
FSDP_TRANSPORT = 'f1afcf575570f1253eb0889e6d15cde7f415f9eca05e594fc827f0c6ff872a83'
FSDP_READER = '18ca88675a4e26a4fb36fe52bbe6cc7cc280290a79ade1985cc3ef7762b9be57'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_original(path, expected=None, limit=64 * 1024 * 1024):
    """Parse only the bytes whose opened-FD digest was verified, without pyc use."""
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'Physical original required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        def identity(s):
            return (s.st_dev, s.st_ino, s.st_uid, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
             and before.st_size <= limit and identity(before) == identity(path.stat()),
             'Original owner/type/size/identity differs')
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b''.join(chunks)
        after = os.fstat(fd)
        need(identity(before) == identity(after) == identity(path.stat())
             and path.resolve(strict=True) == path and len(raw) == after.st_size,
             'Original changed or aliased while reading')
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None:
            need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected)
                 and digest == expected, 'Pinned original SHA differs: ' + str(path))
        return raw, dict(local_physical_path=str(path), sha256=digest, bytes=len(raw),
                         device=after.st_dev, inode=after.st_ino, uid=after.st_uid)
    finally:
        os.close(fd)


def module_original(path, pin, name):
    raw, _ = read_original(path, pin)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def runtime_helpers():
    gate = module_original(GATE, GATE_SHA, 'actual_progress_source_gate')
    common = module_original(TOOLS / 'author200_observation.py', COMMON_SHA, 'actual_progress_typed_common')
    raw, manifest_rec = read_original(TOOLS / 'manifest.json', TOOLS_SHA)
    manifest = common.loads(raw)
    need(manifest['source_commit'] == SOURCE
         and manifest['tools']['author200_observation.py'] == COMMON_SHA,
         'Typed parser manifest/source differs')
    for name, pin in manifest['tools'].items():
        need(Path(name).name == name and name.endswith('.py'), 'Unsafe tool member')
        read_original(TOOLS / name, pin)
    need(manifest['review_records']['ACTUAL75_FSDP_WARNING_FRAMING_PROVENANCE_INDEPENDENT_REVIEW.json']
         == FSDP_REVIEW, 'Actual framing provenance not pinned by tools')
    read_original(TOOLS / 'ACTUAL75_FSDP_WARNING_FRAMING_PROVENANCE_INDEPENDENT_REVIEW.json', FSDP_REVIEW)
    return gate, common, manifest, manifest_rec


def captured_member(snapshot, summary, relative, expected=None):
    relative = str(relative)
    remote = summary['files'][relative]
    raw, local = read_original(snapshot / 'original-files' / relative, expected or remote['sha256'])
    origin = ('/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs/'
              'worker-8350026ae4bc8387c3d3b6880792734dd21e46ad5a9c09e08669a24e-02000000-1533312.out'
              if relative == 'ray-stdout-pid-1533312.out' else str(RUN / relative))
    need(remote['bytes'] == len(raw) and remote['physical_path'] == origin
         and remote['source_offset'] == 0 and remote['complete_original_at_start'] is True
         and remote['same_inode'] is True, 'Snapshot member/transport binding differs')
    return raw, dict(local, physical_origin=remote['physical_path'], captured_member=remote)


def source_context(snapshot, summary, gate, common, tool_manifest, manifest_record):
    anchors = {}
    def captured_json(relative, pin):
        raw, record = captured_member(snapshot, summary, relative, pin)
        value = common.loads(raw)
        need(type(value) is dict, 'Captured object required')
        anchors[str(relative)] = record
        return value
    manifest = captured_json('full_state/global_step_5/manifest.json', MANIFEST5)
    receipt = captured_json('acceptance-5-receipt.json', RECEIPT)
    public = captured_json('bindings/public.json', PUBLIC)
    need(set(receipt) == {'payload', 'signature_hex'}, 'Signed first5 envelope differs')
    payload = receipt['payload']
    gate.verify_rsa_signature(payload, receipt['signature_hex'], public)
    identity = ('method', 'run_root', 'source_commit', 'config_sha256', 'global_step',
                'actual_outer_updates', 'next_global_step', 'horizon', 'warmup_steps',
                'world_size', 'driver_pid', 'uid', 'job_id', 'pause_nonce')
    need(payload['status'] == 'PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
         and payload['checkpoint_manifest_sha256'] == MANIFEST5
         and all(payload.get(k) == manifest.get(k) for k in identity)
         and payload['checkpoint_files'] == manifest['files'], 'Accepted5 signed identity differs')
    need((manifest['method'], manifest['run_root'], manifest['source_commit'], manifest['config_sha256'],
          manifest['global_step'], manifest['actual_outer_updates'], manifest['next_global_step'],
          manifest['horizon'], manifest['warmup_steps'], manifest['world_size'], manifest['driver_pid'],
          manifest['job_id'], manifest['uid']) ==
         ('pure_opsd', str(RUN), SOURCE, CONFIG, 5, 5, 6, 200, 0, 8, 1533312, '3261', 0),
         'Accepted5 is not this actual author200 run')
    def anchor(role):
        record = manifest['files'][role]
        need(set(record) == {'path', 'sha256'}, 'Signed anchor schema differs')
        return captured_json(str(Path(record['path']).relative_to(RUN)), record['sha256'])
    closure = anchor('import_closure')
    need(manifest['files']['import_closure']['sha256'] == CLOSURE and set(closure) == {'modules'}
         and len(closure['modules']) == 210, 'Signed native210 changed')
    need(manifest['files']['resolved_config']['sha256'] == CONFIG, 'Accepted config pin differs')
    config = anchor('resolved_config')
    need(config['baselines']['opsd_e1']['profile'] == 'author200'
         and config['trainer']['total_training_steps'] == 200
         and config['actor_rollout_ref']['actor']['optim']['lr'] == 5e-6
         and config['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 32,
         'Resolved author200 profile differs')
    workers = []
    for rank in range(8):
        record = manifest['files'][f'actor/training_state/rank_{rank}.json']
        row = anchor(f'actor/training_state/rank_{rank}.json')
        pt = manifest['files'][f'actor/training_state/rank_{rank}.pt']
        need((row['rank'], row['world_size'], row['step'], row['status'], row['pid'])
             == (rank, 8, 5, 'EXACT_FULL_STATE_RELOAD_OK', WORKERS[rank])
             and row['path'] == pt['path'] and row['sha256'] == row['state_sha256'] == pt['sha256'],
             'Accepted worker identity/PT reference differs')
        workers.append(dict(rank=rank, pid=row['pid'], object_ids=row['object_ids'],
                            report_path=record['path'], report_sha256=record['sha256']))
    raw, packet_rec = read_original(FSDP_PACKET / 'summary.json', FSDP_SUMMARY)
    packet = common.loads(raw)
    raw, review_rec = read_original(HERE / 'ACTUAL75_FSDP_WARNING_FRAMING_PROVENANCE_INDEPENDENT_REVIEW.json', FSDP_REVIEW)
    peer = common.loads(raw)
    read_original(FSDP_PACKET / 'transport-original.json', FSDP_TRANSPORT)
    read_original(FSDP_PACKET / 'reader-source.py', FSDP_READER)
    torch = closure['modules']['torch.utils._contextlib']
    source_path = str(Path(torch['path']).parent.parent / 'distributed/fsdp/fully_sharded_data_parallel.py')
    source = packet['files']['fully_sharded_data_parallel.py']
    source_raw, captured_source = read_original(FSDP_PACKET / 'fully_sharded_data_parallel.py', FSDP_SHA)
    need(packet['native210_sha256'] == CLOSURE and packet['pinned_native210_torch_helper'] == torch
         and source['physical_path'] == source_path and source['sha256'] == FSDP_SHA
         and source['bytes'] == len(source_raw) == 99345
         and source['before'] == source['after'] and source['same_opened_inode'] is True
         and tool_manifest['dependencies'][source_path] == FSDP_SHA
         and peer['status'] == 'PASS_ACTUAL_RAW_FSDP_FRAME_PROVENANCE_NOT_NEW_PARSER_OR_CHECKPOINT_ACCEPTANCE'
         and peer['native210_closure_sha256'] == CLOSURE
         and peer['warning']['source_sha256'] == FSDP_SHA
         and peer['warning']['source_path'] == source_path, 'Actual FSDP source provenance differs')
    proc = packet['rank0_process']
    need(proc['pid'] == WORKERS[0] and proc['start_ticks'] == STARTS[0] and proc['uids'] == [0]*4
         and proc['present'] is True and '/job_3261/step_0' in proc['cgroup_original'],
         'Source packet warning owner differs')
    stderr_raw, stderr_rec = read_original(FSDP_PACKET / 'rank0-original.err',
                                         packet['files']['rank0-original.err']['sha256'])
    need(len(stderr_raw) == packet['files']['rank0-original.err']['bytes']
         and packet['rank0_stderr_actual_fd2'] == packet['files']['rank0-original.err']['physical_path']
         and peer['warning']['actual_stderr_sha256'] == stderr_rec['sha256'], 'Owned stderr provenance differs')
    calls = [n for n in ast.walk(ast.parse(source_raw)) if isinstance(n, ast.Call) and n.lineno == 689
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'warnings' and n.func.attr == 'warn']
    need(len(calls) == 1 and len(calls[0].args) == 2 and not calls[0].keywords
         and isinstance(calls[0].args[0], ast.Constant) and calls[0].args[0].value == common.FSDP_WARNING_TEXT
         and isinstance(calls[0].args[1], ast.Name) and calls[0].args[1].id == 'FutureWarning',
         'Actual pinned source warning AST differs')
    binding = dict(kind='EXTRA_ACTUAL_FSDP_SOURCE_NOT_SIGNED_NATIVE210_MODULE',
        native_closure_sha256=CLOSURE, torch_anchor=torch,
        source_record={k: source[k] for k in ('physical_path', 'sha256', 'bytes')},
        warning_line=689, warning_source_line=source_raw.decode().splitlines()[688],
        external_tools_manifest=manifest_record, local_captured_source_record=captured_source,
        original_capture_summary=packet_rec, original_source_provenance_review=review_rec,
        scope='VERIFIED_LOCAL_COPY_OF_INDEPENDENTLY_CAPTURED_PHYSICAL_SOURCE_NOT_A_NEW_REMOTE_READ')
    return dict(gate=gate, workers=workers, checkpoint_warning_path=Path(torch['path']).parent/'checkpoint.py',
                fsdp_warning_source=binding), anchors


def review_logs(driver_raw, aggregate_raw, context, common):
    records, originals = {}, {}
    for label, raw in [('driver', driver_raw), ('aggregate', aggregate_raw)]:
        length = raw.rfind(b'\n') + 1
        need(length > 0, 'No complete original LF prefix')
        prefix = raw[:length]
        records[label] = common.parse_records(prefix, context)
        releases = []
        for line in prefix.decode().splitlines():
            if 'OPSD_E1_STEP5_RELEASED_JSON=' in line:
                releases.append(context['gate'].parse_log_marker(line, 'OPSD_E1_STEP5_RELEASED_JSON=',
                    driver_pid=1533312, job_id='3261', worker_pids=WORKERS,
                    checkpoint_warning_path=context['checkpoint_warning_path']))
        need(releases == [{'actual_outer_updates': 5, 'driver_pid': 1533312, 'next_global_step': 6,
                           'receipt_sha256': RECEIPT}], 'Actual signed release differs')
        originals[label] = dict(whole_bytes=len(raw), whole_sha256=hashlib.sha256(raw).hexdigest(),
            complete_prefix_bytes=length, complete_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
            excluded_unclosed_tail_bytes=len(raw)-length)
    rows = records['driver']['outers']
    need(5 <= len(rows) <= 200, 'Missing/excess actual outer')
    join = common.join_log_prefixes(records['driver'], records['aggregate'], context, 5)
    keys = ['critic/rewards/mean', 'actor/opsd_loss', 'actor/lr', 'actor/grad_norm',
            'env/finish_ratio', 'env/ratio_of_valid_action', 'timing_s/step']
    return dict(actual_complete_outer=len(rows), aggregate_complete_outer=len(records['aggregate']['outers']),
        same_process_adam_calls_per_rank=16*len(rows), latest_metrics={k: rows[-1]['metrics'][k] for k in keys},
        reward_history=[r['metrics']['critic/rewards/mean'] for r in rows], log_join=join,
        original_log_prefixes=originals, driver_original_sha256=originals['driver']['whole_sha256'],
        aggregate_original_sha256=originals['aggregate']['whole_sha256'])


def review(snapshot):
    gate, common, tool_manifest, manifest_record = runtime_helpers()
    raw, summary_record = read_original(snapshot / 'observation-summary.json')
    summary = common.loads(raw)
    commands = summary['commands']
    need(summary['job_id'] == '3261' and summary['node'] == 'h100-10', 'Wrong snapshot')
    need(commands['source_head']['returncode'] == 0 and commands['source_head']['stdout'].strip() == SOURCE
         and commands['source_porcelain']['returncode'] == 0 and not commands['source_porcelain']['stdout'].strip(),
         'Source drift')
    fields = dict(x.split('=', 1) for x in commands['scontrol']['stdout'].split() if '=' in x)
    need(tuple(fields.get(k) for k in ['JobId','JobState','Restarts','NodeList','TresPerNode'])
         == ('3261','RUNNING','0','h100-10','gres:gpu:h100:8'), 'Not the same running job')
    for pid in WORKERS + [1533312,1531763]:
        p = summary['processes'][str(pid)]
        need(p['in_observed_job'] and p['uids'] == [0]*4 and p['state'] in ('R','S','D','I')
             and p['pid'] == pid and '/job_3261/step_0' in p['cgroup'], 'Process identity changed')
    need(summary['processes']['1533312']['start_ticks'] == '1829085316'
         and summary['processes']['1531763']['start_ticks'] == '1829074494'
         and all(summary['processes'][str(pid)]['start_ticks'] == start for pid,start in zip(WORKERS,STARTS)),
         'Process restart')
    apps = summary['unfiltered_application_csv_rows']
    need(len(apps) == 16 and set(int(a[1]) for a in apps) == set(WORKERS+[1531763]),
         'Unmatched GPU application')
    context, anchors = source_context(snapshot, summary, gate, common, tool_manifest, manifest_record)
    logs = [captured_member(snapshot, summary, name)[0]
            for name in ['ray-stdout-pid-1533312.out','train.log']]
    result = review_logs(*logs, context, common)
    return dict(result, status='PASS_CURRENT_LOGGED_OUTERS_NOT_FINAL200', job_id='3261',
        source_commit=SOURCE, config_sha256=CONFIG, driver_pid=1533312, driver_start_ticks='1829085316',
        snapshot=str(snapshot), snapshot_finished_utc=summary['finished_utc'], snapshot_summary=summary_record,
        typed_parser_sha256=COMMON_SHA, typed_tools_manifest_sha256=TOOLS_SHA,
        accepted5_cryptographic_anchor_records=anchors,
        observed_gpu_used_gib=[float(a[3])/1024 for a in summary['gpu_inventory_csv_rows']],
        slurm_fields={k:fields[k] for k in ['JobState','Restarts','RunTime','NodeList','TresPerNode']},
        scope='Captured complete original log rows and observed process/resource identity only; '
              'not later checkpoint, final evaluation, whole terminal or effect acceptance',
        limits=['Typed FSDP source pin is an independently captured local original, not a fresh remote read.',
                'Signed native210 remains unchanged; the additional FSDP pin is separately verified.',
                'Old frozen formal/c738 terminal re-scan remains unchanged and may fail on the preserved raw frame.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = review(args.snapshot.resolve(strict=True))
    with args.output.open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({k:result[k] for k in ['actual_complete_outer','aggregate_complete_outer',
        'latest_metrics','observed_gpu_used_gib']},ensure_ascii=False))
