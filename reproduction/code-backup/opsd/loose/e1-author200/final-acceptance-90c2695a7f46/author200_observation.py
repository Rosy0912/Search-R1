"""Read-only author200 evidence consumers. No signing, CUDA, restore or job control."""
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import types

SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
REPO = ROOT / 'repo-188f4e5e7b63'
ASSETS = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004')
JOB = '3261'
CONFIG_SHA = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
MANIFEST5_SHA = '8c6e17b2af0026b1267a2422c1437aa00461da46e0a1911caea3633bfef473d3'
RECEIPT5_SHA = '812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b'
PUBLIC_SHA = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
CLOSURE_SHA = '3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'
FORMAL_SHA = 'bb5f4872142f34a330abab14b5ae3869bc212c2e1eb7e247cf895f5aa3319d7c'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL_SHA = '7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
GATE_SHA = 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb'
DRIVER_PID = 1533312
DRIVER_START = 1829085316
LIVE_DRIVER_LOG = Path('/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs/worker-8350026ae4bc8387c3d3b6880792734dd21e46ad5a9c09e08669a24e-02000000-1533312.out')
STEPS = (25, 50, 75, 100, 125, 150, 175, 200)
KIND = 'OBSERVED_AUTHOR200_CHECKPOINT_INVENTORY_NOT_TRAINING_MANIFEST'
FINAL_COUNTS = {'nq': 3610, 'triviaqa': 11313, 'popqa': 14267,
                'hotpotqa': 7405, '2wikimultihopqa': 12576,
                'musique': 2417, 'bamboogle': 125}
DEV_COUNTS = {'nq': 256, 'hotpotqa': 256}
STAGE_COUNTS = {'nq': 32, 'hotpotqa': 32}
QA_KEYS = ('em', 'f1', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls')
FSDP_WARNING_TEXT = (
    'FSDP.state_dict_type() and FSDP.set_state_dict_type() are being '
    'deprecated. Please use APIs, get_state_dict() and set_state_dict(), '
    'which can support different parallelisms, FSDP1, FSDP2, DDP. '
    'API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html'
    '#torch.distributed.checkpoint.state_dict.get_state_dict .'
    'Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def number(raw):
        result = float(raw)
        need(math.isfinite(result), 'Nonfinite JSON float')
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=lambda _: need(False, 'Nonfinite JSON constant'))


def physical(path, root=None):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path,
         'Absolute physical path required')
    if root is not None:
        path.relative_to(Path(root))
    need(stat.S_ISREG(path.stat().st_mode) and path.stat().st_uid == os.getuid(),
         'Owned regular file required')
    return path


def stat_identity(value):
    return value.st_dev, value.st_ino, value.st_uid, value.st_mode


class MemberRecord(dict):
    """Exact public six-field schema, plus a transient publication fingerprint."""
    def __init__(self, values, fingerprint):
        super().__init__(values)
        self.fingerprint = fingerprint


def fingerprint(value):
    return stat_identity(value), value.st_size, value.st_mtime_ns, value.st_ctime_ns


def recheck_member(record, root):
    path = physical(record['physical_path'], root)
    need(isinstance(record, MemberRecord) and fingerprint(path.stat()) == record.fingerprint,
         'Checkpoint member changed after streamed SHA')


def stable_file(path, root=None, expected=None, read_limit=None):
    """Stream SHA; compare open FD and physical member before/after every read."""
    path = physical(path, root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat_identity(before) == stat_identity(path.stat()), 'Member replaced before read')
        need(before.st_size > 0 or expected == hashlib.sha256(b'').hexdigest(),
             'Empty member without an explicit legitimate empty-source pin')
        if read_limit is not None:
            need(before.st_size <= read_limit, 'Metadata member too large')
        digest = hashlib.sha256()
        data = bytearray() if read_limit is not None else None
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if data is not None:
                data.extend(block)
        after = os.fstat(fd)
        need((stat_identity(before), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (stat_identity(after), after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             'Checkpoint member changed while reading')
        need(stat_identity(after) == stat_identity(physical(path, root).stat()),
             'Checkpoint member replaced after read')
        pin = digest.hexdigest()
        if expected is not None:
            need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected)
                 and pin == expected, 'Member SHA differs: ' + str(path))
        record = MemberRecord(dict(physical_path=str(path), sha256=pin, bytes=after.st_size,
                              device=after.st_dev, inode=after.st_ino, uid=after.st_uid), fingerprint(after))
        return record, bytes(data) if data is not None else None
    finally:
        os.close(fd)


def json_file(path, root=None, expected=None):
    record, raw = stable_file(path, root, expected, 32 * 1024 * 1024)
    result = loads(raw)
    need(isinstance(result, dict), 'JSON object required')
    return result, record


def prefix_file(path, root, length, expected):
    """A fixed complete-line prefix may coexist with a growing live log."""
    need(type(length) is int and 0 < length <= 1024 ** 3, 'Explicit positive prefix bytes required')
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected), 'Explicit prefix SHA required')
    path = physical(path, root)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(before.st_size >= length, 'Log shorter than pinned prefix')
        need(stat_identity(before) == stat_identity(path.stat()), 'Log replaced before prefix read')
        remaining = length
        blocks = []
        while remaining:
            block = os.read(fd, min(8 * 1024 * 1024, remaining))
            need(bool(block), 'Log prefix truncated')
            blocks.append(block)
            remaining -= len(block)
        raw = b''.join(blocks)
        need(raw.endswith(b'\n') and hashlib.sha256(raw).hexdigest() == expected,
             'Incomplete terminal line or changed prefix SHA')
        # A second read checks pinned bytes, not the size/mtime of append-only tails.
        os.lseek(fd, 0, os.SEEK_SET)
        h = hashlib.sha256()
        remaining = length
        while remaining:
            block = os.read(fd, min(8 * 1024 * 1024, remaining))
            need(bool(block), 'Log prefix truncated during verification')
            h.update(block)
            remaining -= len(block)
        after = os.fstat(fd)
        need(h.hexdigest() == expected and after.st_size >= length
             and stat_identity(before) == stat_identity(after)
             == stat_identity(physical(path, root).stat()), 'Pinned log prefix/identity changed')
        raw.decode('utf-8')
        return raw, dict(physical_path=str(path), bytes=length, sha256=expected,
                         device=after.st_dev, inode=after.st_ino, uid=after.st_uid,
                         observed_file_size_at_open=before.st_size,
                         observed_file_size_at_close=after.st_size,
                         complete_lines=raw.count(b'\n'))
    finally:
        os.close(fd)


def exclusive_json(path, value):
    path = Path(path)
    need(path.is_absolute() and path.parent.resolve(strict=True) == path.parent
         and not path.exists() and not path.is_symlink(), 'New physical output required')
    with path.open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    return stable_file(path)[0]


def frozen_gate(repo):
    path = Path(repo) / 'search_r1/baselines/checkpoint_gate.py'
    _, raw = stable_file(path, expected=GATE_SHA, read_limit=2 * 1024 * 1024)
    module = types.ModuleType('author200_exact_pinned_gate')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def require_contract(args):
    need(Path(args.run_root) == RUN and str(args.job_id) == JOB,
         'Only current physical author200 run/job allowed')
    need(Path(args.source_root) == REPO and args.source_commit == SOURCE,
         'Wrong frozen author200 source')
    need(args.accepted5_manifest_sha256 == MANIFEST5_SHA
         and args.accepted5_receipt_sha256 == RECEIPT5_SHA
         and args.public_key_sha256 == PUBLIC_SHA
         and args.native_import_closure_sha256 == CLOSURE_SHA, 'Actual accepted5 pins required')


def load_accepted_context(args):
    require_contract(args)
    run = Path(args.run_root)
    need(run.resolve(strict=True) == run and run.stat().st_uid == os.getuid(), 'Physical owned run required')
    gate = frozen_gate(Path(args.source_root))
    manifest, mrec = json_file(args.accepted5_manifest, run, MANIFEST5_SHA)
    receipt, rrec = json_file(args.accepted5_receipt, run, RECEIPT5_SHA)
    public, prec = json_file(args.public_key, run, PUBLIC_SHA)
    need(set(receipt) == {'payload', 'signature_hex'}, 'Actual RSA envelope required')
    payload = receipt['payload']
    gate.verify_rsa_signature(payload, receipt['signature_hex'], public)
    need(payload.get('status') == 'PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW'
         and payload.get('checkpoint_manifest_sha256') == MANIFEST5_SHA,
         'Wrong signed first5 authorization')
    identity = ('method', 'run_root', 'source_commit', 'config_sha256', 'global_step',
                'actual_outer_updates', 'next_global_step', 'horizon', 'warmup_steps',
                'world_size', 'driver_pid', 'uid', 'job_id', 'pause_nonce')
    need(all(payload.get(k) == manifest.get(k) for k in identity)
         and payload.get('checkpoint_files') == manifest['files'], 'Signed first5 identities/file map differ')
    need(manifest['version'] == 1 and manifest['method'] == 'pure_opsd'
         and manifest['run_root'] == str(run) and manifest['source_commit'] == SOURCE
         and manifest['config_sha256'] == CONFIG_SHA and manifest['job_id'] == JOB
         and manifest['uid'] == os.getuid() and manifest['driver_pid'] == DRIVER_PID
         and (manifest['global_step'], manifest['actual_outer_updates'], manifest['next_global_step'],
              manifest['horizon'], manifest['warmup_steps'], manifest['world_size']) == (5, 5, 6, 200, 0, 8),
         'First5 does not belong to actual author200')
    references = {}
    for role, rec in manifest['files'].items():
        need(set(rec) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', rec['sha256']),
             'Invalid signed file reference')
        Path(rec['path']).relative_to(run)
        references[role] = dict(physical_path=rec['path'], sha256=rec['sha256'])
    # Repeated accepted anchors are legitimate. Large accepted HF/PT are referenced
    # by the already-signed map; this observer does not re-audit all of first5.
    def anchor(role):
        rec = references[role]
        return json_file(rec['physical_path'], run, rec['sha256'])[0]
    need(references['review_public_key'] == dict(physical_path=str(prec['physical_path']), sha256=PUBLIC_SHA),
         'Public key is not the signed binding')
    need(references['import_closure']['sha256'] == CLOSURE_SHA
         and references['resolved_config']['sha256'] == CONFIG_SHA
         and references['data_manifest']['sha256'] == DATA_SHA
         and references['model_manifest']['sha256'] == MODEL_SHA, 'Common/source anchor pins differ')
    need(str(args.native_import_closure) == references['import_closure']['physical_path'],
         'Native closure path is not signed original')
    config = anchor('resolved_config')
    need(config['baselines']['opsd_e1']['profile'] == 'author200'
         and config['trainer']['total_training_steps'] == 200
         and config['actor_rollout_ref']['actor']['optim']['lr'] == 5e-6
         and config['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == 32,
         'Resolved author200 profile differs')
    binding = anchor('launch_binding')
    need(binding['run_root'] == str(run) and binding['source_commit'] == SOURCE,
         'Launch binding identity differs')
    for role, rec in binding['files'].items():
        need(references[role] == dict(physical_path=rec['path'], sha256=rec['sha256']),
             'Launch and signed binding disagree')
        anchor(role)
    source = anchor('source')
    need(source['source_commit'] == SOURCE and source['repo'] == str(REPO)
         and source['clean'] is True and source['formal_manifest_sha256'] == FORMAL_SHA,
         'Actual source binding differs')
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    clean = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain'],
                           capture_output=True, text=True, check=True).stdout
    need(commit == SOURCE and clean == '', 'Frozen tracked source changed')
    formal_path = ROOT / 'tools/formal-bb5f4872142f/manifest.json'
    formal, _ = json_file(formal_path, ROOT, FORMAL_SHA)
    need(formal['source_commit'] == SOURCE and formal['source_files'] == source['source_files'],
         'Formal/source explicit file inventory differs')
    for name, pin in formal['source_files'].items():
        stable_file(REPO / name, REPO, pin)
    for name, pin in formal['tools'].items():
        stable_file(formal_path.parent / name, formal_path.parent, pin)
    closure = anchor('import_closure')
    need(set(closure) == {'modules'} and closure['modules'], 'Full native closure required')
    for rec in closure['modules'].values():
        need(set(rec) == {'path', 'sha256'}, 'Native closure file schema differs')
        stable_file(rec['path'], expected=rec['sha256'])
    workers = []
    for rank in range(8):
        report = anchor(f'actor/training_state/rank_{rank}.json')
        need((report['rank'], report['world_size'], report['step']) == (rank, 8, 5)
             and report['status'] == 'EXACT_FULL_STATE_RELOAD_OK', 'Wrong accepted rank report')
        pt = references[f'actor/training_state/rank_{rank}.pt']
        need(report['path'] == pt['physical_path'] and report['state_sha256'] == report['sha256'] == pt['sha256'],
             'Accepted rank/PT reference differs')
        workers.append(dict(rank=rank, pid=report['pid'], object_ids=report['object_ids'],
                            report_path=references[f'actor/training_state/rank_{rank}.json']['physical_path'],
                            report_sha256=references[f'actor/training_state/rank_{rank}.json']['sha256']))
    need(len({r['pid'] for r in workers}) == 8, 'Accepted workers alias')
    anchor('driver/driver.json')
    anchor('algorithm')
    context = dict(gate=gate, run=run, manifest=manifest, config=config,
                   workers=workers, anchor_files=references, common_manifest=anchor('data_manifest'),
                   source_files=formal['source_files'], formal_path=str(formal_path),
                   closure_path=references['import_closure']['physical_path'],
                   accepted5=dict(manifest_path=mrec['physical_path'], manifest_sha256=MANIFEST5_SHA,
                                  signed_receipt_path=rrec['physical_path'], signed_receipt_sha256=RECEIPT5_SHA,
                                  public_key_path=prec['physical_path'], public_key_sha256=PUBLIC_SHA,
                                  driver_pid=DRIVER_PID, driver_start_ticks=DRIVER_START, workers=workers))
    # The actual 210-file closure pins this Torch Python helper, not necessarily
    # the package __init__ module. Its installed directory binds warning syntax.
    torch_rec = closure['modules'].get('torch.utils._contextlib')
    need(torch_rec is not None, 'Actual native Torch helper path missing')
    context['checkpoint_warning_path'] = Path(torch_rec['path']).parent / 'checkpoint.py'
    context['fsdp_warning_source'] = bind_fsdp_warning_source(closure)
    return context


def bind_fsdp_warning_source(closure):
    """Bind the separately captured FSDP file, without changing signed native210."""
    torch_rec = closure['modules'].get('torch.utils._contextlib')
    need(type(torch_rec) is dict and set(torch_rec) == {'path', 'sha256'},
         'Actual native Torch helper binding required')
    stable_file(torch_rec['path'], expected=torch_rec['sha256'])
    torch_root = Path(torch_rec['path']).parent.parent
    source_path = torch_root / 'distributed/fsdp/fully_sharded_data_parallel.py'
    tools = Path(__file__).resolve().parent
    manifest, manifest_record = json_file(tools / 'manifest.json', tools)
    self_record, _ = stable_file(Path(__file__).resolve(), tools)
    need(manifest.get('source_commit') == SOURCE
         and manifest.get('tools', {}).get('author200_observation.py') == self_record['sha256'],
         'Current externally verified tools/common binding differs')
    pin = manifest.get('dependencies', {}).get(str(source_path))
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin),
         'FSDP warning source is absent from the new frozen dependency inventory')
    source_record, raw = stable_file(source_path, expected=pin, read_limit=32 * 1024 * 1024)
    source_lines = raw.decode('utf-8').splitlines()
    need(len(source_lines) >= 689 and source_lines[688].strip() == 'warnings.warn(',
         'Bound FSDP warning line689 no longer denotes warnings.warn')
    calls = [node for node in ast.walk(ast.parse(raw)) if isinstance(node, ast.Call)
             and node.lineno == 689 and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'warnings'
             and node.func.attr == 'warn']
    need(len(calls) == 1 and len(calls[0].args) == 2 and not calls[0].keywords
         and isinstance(calls[0].args[0], ast.Constant)
         and calls[0].args[0].value == FSDP_WARNING_TEXT
         and isinstance(calls[0].args[1], ast.Name) and calls[0].args[1].id == 'FutureWarning',
         'Pinned FSDP689 source warning literal/category differs')
    return dict(kind='EXTRA_ACTUAL_FSDP_SOURCE_NOT_SIGNED_NATIVE210_MODULE',
                native_closure_sha256=CLOSURE_SHA, torch_anchor=dict(torch_rec),
                source_record=dict(source_record), warning_line=689,
                warning_source_line=source_lines[688],
                external_tools_manifest=dict(manifest_record))


def assert_metric(value, bounded=False):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
         'Invalid nonfinite/negative evaluation metric')
    if bounded:
        need(value <= 1, 'Evaluation rate out of range')


def check_evaluation(metrics, counts):
    need(isinstance(metrics, dict), 'Evaluation object required')
    total = sum(counts.values())
    need(type(metrics.get('val/qa_count/all')) is int and metrics['val/qa_count/all'] == total,
         'Wrong QA total')
    for prefix in ('val/qa_count/', 'val/evaluated_count/'):
        observed = {k[len(prefix):]: v for k, v in metrics.items() if k.startswith(prefix)}
        expected = dict(counts, all=total) if prefix == 'val/qa_count/' else counts
        need(observed == expected and all(type(v) is int for v in observed.values()),
             'Wrong fixed evaluation source/cardinality')
    for source in counts:
        assert_metric(metrics.get('val/test_score/' + source), True)
    need({k.removeprefix('val/test_score/') for k in metrics if k.startswith('val/test_score/')} == set(counts),
         'Unexpected reward source')
    for key in QA_KEYS:
        observed_sources = {k[len('val/' + key + '/'):]
                            for k in metrics if k.startswith('val/' + key + '/')}
        need(observed_sources == set(counts) | {'all'}, 'Missing/extra QA metric sources')
        for source in set(counts) | {'all'}:
            assert_metric(metrics.get(f'val/{key}/{source}'), key not in ('generated_tokens', 'search_calls'))
        average = sum(metrics[f'val/{key}/{s}'] * n for s, n in counts.items()) / total
        need(math.isclose(metrics[f'val/{key}/all'], average, rel_tol=1e-9, abs_tol=1e-12),
             'Overall evaluation is not source-count weighted')
    return dict(count=total, source_counts=dict(counts), original_metrics=metrics,
                scope='AGGREGATE_FIXED_DATA_EVALUATION_NOT_PER_QUESTION_RESCORING')


def owned_fsdp_frame(line, physical_line, marker, role, lineno, offset, context, original_error):
    """Accept only one observed complete FSDP source-bound warning frame."""
    gate = context['gate']
    need(line.count(marker) == 1 and not any(other != marker and other in line
         for other in gate.REVIEW_LOG_MARKERS), 'Mixed/duplicate evidence markers')
    prefix, body = line.split(marker, 1)
    body_start = len(prefix) + len(marker) + len(body) - len(body.lstrip())
    body = body.lstrip()
    _, end = json.JSONDecoder().raw_decode(body)
    raw_json = body[:end]
    # Boundary decoding alone is never authority for body integrity.
    need(isinstance(loads(raw_json), dict), 'FSDP frame must follow one strict JSON object')
    suffix = body[end:]
    binding = context.get('fsdp_warning_source')
    need(type(binding) is dict and binding.get('kind') == 'EXTRA_ACTUAL_FSDP_SOURCE_NOT_SIGNED_NATIVE210_MODULE'
         and binding.get('native_closure_sha256') == CLOSURE_SHA
         and binding.get('warning_line') == 689, 'Actual extra FSDP source binding required')
    source = binding['source_record']
    need(type(source.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}', source['sha256'])
         and type(source.get('bytes')) is int and source['bytes'] > 0,
         'Actual FSDP source SHA/bytes required')
    source_path = Path(source['physical_path'])
    anchor = binding['torch_anchor']
    need(source_path.is_absolute() and source_path == Path(anchor['path']).parent.parent
         / 'distributed/fsdp/fully_sharded_data_parallel.py', 'FSDP source is outside pinned Torch root')
    pattern = (r'\x1b\[36m\(WorkerDict pid=(?P<pid>[1-9][0-9]*)\)\x1b\[0m '
               + re.escape(str(source_path) + ':689: FutureWarning: ' + FSDP_WARNING_TEXT))
    match = re.fullmatch(pattern, suffix)
    need(match is not None, 'Not the exact single owned FSDP warning frame')
    workers = context['workers']
    need(len(workers) == 8 and {r['rank'] for r in workers} == set(range(8))
         and len({r['pid'] for r in workers}) == 8
         and all(type(r['pid']) is int and r['pid'] > 0 for r in workers),
         'Original eight accepted native worker identities required')
    matches = [r for r in workers if r['pid'] == int(match.group('pid'))]
    need(len(matches) == 1, 'FSDP warning owner is outside accepted worker set')
    clean = line[:body_start] + raw_json
    value = gate.parse_log_marker(clean, marker, driver_pid=DRIVER_PID, job_id=JOB,
                                  worker_pids={r['pid'] for r in workers},
                                  checkpoint_warning_path=context.get('checkpoint_warning_path'))
    json_start = len(line[:body_start].encode('utf-8'))
    json_end = json_start + len(raw_json.encode('utf-8'))
    payload = physical_line.encode('utf-8')
    frame = dict(kind='OWNED_FSDP_FUTUREWARNING_AFTER_EVIDENCE_JSON', evidence_role=role,
        evidence_marker=marker, line=lineno, source_offset=offset,
        record_bytes=len(payload), record_sha256=hashlib.sha256(payload).hexdigest(),
        raw=physical_line, raw_json=raw_json, json_sha256=hashlib.sha256(raw_json.encode()).hexdigest(),
        raw_prefix=line[:body_start], raw_suffix=suffix,
        suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
        byte_boundaries=dict(json_start=json_start, json_end=json_end,
                             suffix_start=json_end, suffix_end=len(line.encode('utf-8'))),
        raw_line_ending=physical_line[len(line):], worker_evidence=dict(matches[0]),
        source_binding=binding, original_gate_sha256=GATE_SHA,
        original_gate_error=str(original_error), clean_gate_sha256=GATE_SHA,
        scope='ONE_RAW_TRANSPORT_FRAME_NOT_AN_ADDITIONAL_NATIVE_RANK_REPORT')
    return value, frame


def parse_records(raw, context):
    gate = context['gate']
    workers = {v['pid'] for v in context['workers']}
    result = dict(outers=[], outer_positions=[], evaluations=[], evaluation_positions=[],
                  final=[], final_positions=[], native_console=[], interleaved_frames=[])
    markers = {'OPSD_E1_OUTER_JSON=': 'outers', 'BASELINE_EVALUATION_METRICS_JSON=': 'evaluations',
               'SEARCH_R1_FINAL_METRICS_JSON=': 'final', 'FULL_STATE_RANK_JSON=': 'native_console'}
    offset = 0
    for lineno, physical_line in enumerate(raw.decode('utf-8').splitlines(keepends=True), 1):
        # Preserve physical UTF-8 bytes including the actual original line ending.
        line = physical_line.rstrip('\r\n')
        for marker, role in markers.items():
            if marker not in line:
                continue
            repeat = None
            frame = None
            arguments = dict(driver_pid=DRIVER_PID, job_id=JOB, worker_pids=workers,
                             checkpoint_warning_path=context.get('checkpoint_warning_path'))
            try:
                value = gate.parse_log_marker(line, marker, **arguments)
            except ValueError as original_error:
                tag_line = None
                if role == 'native_console':
                    prefix, body = line.split(marker, 1)
                    body = body.lstrip()
                    _, end = json.JSONDecoder().raw_decode(body)
                    suffix = body[end:]
                    tag = re.match(r'\s*\x1b\[32m \[repeated ([1-7])x across cluster\]\x1b\[0m', suffix)
                    if tag:
                        repeat = int(tag.group(1))
                        tag_line = prefix + marker + body[:end] + suffix[tag.end():]
                if tag_line is not None:
                    # Only the existing native AMP/tag corner uses this unchanged gate path.
                    value = gate.parse_log_marker(tag_line, marker, **arguments)
                else:
                    value, frame = owned_fsdp_frame(line, physical_line, marker, role,
                                                    lineno, offset, context, original_error)
            result[role].append(value)
            if frame is not None:
                result['interleaved_frames'].append(frame)
                # Existing observer persists this field into its transport_records.
                # A typed outer-warning record never increments native_console count.
                result.setdefault('native_transport', []).append(frame)
            if role == 'native_console':
                result.setdefault('native_transport', []).append(dict(line=lineno, raw=line,
                                                                      observed_repeat_tag=repeat))
            else:
                result[{'outers': 'outer_positions', 'evaluations': 'evaluation_positions',
                        'final': 'final_positions'}[role]].append(lineno)
        offset += len(physical_line.encode('utf-8'))
    return result


def check_outer_rows(rows, context):
    need(rows and len(rows) <= 200 and [r.get('actual_outer_updates') for r in rows]
         == list(range(1, len(rows) + 1)), 'Noncontiguous/duplicate/extra actual outer clock')
    for step, row in enumerate(rows, 1):
        need(row.get('global_step') == step and type(row.get('global_step')) is int
             and type(row.get('actual_outer_updates')) is int
             and row.get('driver_pid') == DRIVER_PID and row.get('job_id') == JOB
             and row.get('source_commit') == SOURCE and row.get('config_sha256') == CONFIG_SHA,
             'Outer source/job/config/driver changed')
        m = row['metrics']
        context['gate'].validate_outer(m, profile='author200')
        for key in ('actor/opsd_loss', 'actor/grad_norm'):
            need(type(m.get(key)) in (int, float) and math.isfinite(m[key]), 'Loss/gradient metric nonfinite')
        need(m['actor/grad_norm'] >= 0, 'Negative gradient norm')
        need(type(m.get('actor/lr')) in (float, int) and m['actor/lr'] == 5e-6,
             'Actual constant LR differs')
        for rank in range(8):
            suffix = '/rank_' + str(rank)
            for key in ('actor/opsd_loss_mean', 'actor/grad_norm_mean'):
                need(type(m.get(key + suffix)) in (int, float) and math.isfinite(m[key + suffix]),
                     'Rank loss/gradient mean nonfinite')
            need(m['actor/grad_norm_mean' + suffix] >= 0, 'Negative rank gradient norm')
            for key in ('actor/old_log_probs_present', 'actor/old_log_probs_finite'):
                need(m.get(key + suffix) == 1 and not isinstance(m.get(key + suffix), bool),
                     'Old log probabilities absent/nonfinite')
            need(type(m.get('actor/supervised_tokens' + suffix)) in (int, float)
                 and math.isfinite(m['actor/supervised_tokens' + suffix])
                 and m['actor/supervised_tokens' + suffix] > 0, 'Supervision absent')
            need(m.get('teacher/cuda_update_entries_cumulative_per_rank' + suffix) == step
                 and not isinstance(m.get('teacher/cuda_update_entries_cumulative_per_rank' + suffix), bool),
                 'Teacher CUDA outer clock differs')
    return len(rows)


def join_log_prefixes(driver, aggregate, context, saved_step):
    last = check_outer_rows(driver['outers'], context)
    need(last >= saved_step, 'Authoritative driver has not completed saved step')
    if aggregate['outers']:
        check_outer_rows(aggregate['outers'], context)
    k = len(aggregate['outers'])
    need(k <= last and all(canonical(r) == canonical(driver['outers'][i])
                          for i, r in enumerate(aggregate['outers'])), 'Aggregate prefix conflicts with own driver')
    return dict(authoritative_last_outer=last, aggregate_last_outer=k,
                aggregate_forwarding_lag=last - k, aggregate_behind_checkpoint=k < saved_step,
                interleaved_frames=dict(driver=driver.get('interleaved_frames', []),
                                        aggregate=aggregate.get('interleaved_frames', [])),
                warning_source_bindings=dict(fsdp=context['fsdp_warning_source'])
                    if 'fsdp_warning_source' in context else {},
                scope='OBSERVED_FIXED_PREFIX_CONTINUITY_NOT_WHOLE_FUTURE_LIFETIME')


def check_evaluation_sequence(records, last_outer, require_final=False):
    schedule = [('dev', 0, DEV_COUNTS), ('stage', 5, STAGE_COUNTS)]
    for step in (50, 100, 150, 200):
        schedule.extend([('dev', step, DEV_COUNTS), ('stage', step, STAGE_COUNTS)])
    schedule.append(('final', 200, FINAL_COUNTS))
    events = records['evaluations']
    need(len(events) <= len(schedule), 'Duplicated evaluation records')
    minimum = len([x for x in schedule[:-1] if x[1] <= last_outer])
    need(len(events) >= minimum, 'Completed outer has missing fixed evaluation')
    checked = []
    for i, metrics in enumerate(events):
        kind, step, counts = schedule[i]
        need(step <= last_outer + 1 and (kind != 'final' or last_outer == 200),
             'Evaluation ahead of observed outer budget')
        check_evaluation(metrics, counts)
        position = records['evaluation_positions'][i]
        if kind == 'final':
            need(position > records['outer_positions'][199], 'Final evaluation before outer200')
        elif step <= last_outer:
            need(position < records['outer_positions'][max(step - 1, 0)],
                 'Evaluation chronology conflicts with completed outer')
            if step > 0:
                need(position > records['outer_positions'][step - 2], 'Evaluation belongs to older outer')
                outer_metrics = records['outers'][step - 1]['metrics']
                if kind == 'stage':
                    outer_metrics = {k.removeprefix('opsd_stage/'): v for k, v in outer_metrics.items()
                                     if k.startswith('opsd_stage/')}
                need(all(canonical(outer_metrics.get(k)) == canonical(v) for k, v in metrics.items()),
                     'Console/outer evaluation JSON differs')
        checked.append(dict(kind=kind, step=step, count=sum(counts.values()), source_counts=counts,
                            original_metrics=metrics, console_line=position))
    final = records['final']
    need(len(final) <= 1, 'Duplicated final marker')
    if final:
        need(last_outer == 200 and len(events) == len(schedule)
             and canonical(final[0]) == canonical(events[-1])
             and records['final_positions'][0] > records['evaluation_positions'][-1],
             'Final marker not paired to actual final evaluation')
        check_evaluation(final[0], FINAL_COUNTS)
    if require_final:
        need(last_outer == 200 and len(final) == 1 and len(events) == len(schedule),
             'Full200/final51713 evidence absent')
    return dict(events=checked, final_marker_present=bool(final),
                per_question_raw_evidence_available=False,
                scope='EXACT_FIXED_SET_AGGREGATE_PROTOCOL_NOT_EFFECT_IMPROVEMENT')


def member_unique(records):
    need(len({r['physical_path'] for r in records.values()}) == len(records), 'New checkpoint path alias')
    need(len({(r['device'], r['inode']) for r in records.values()}) == len(records),
         'New checkpoint inode alias')


def checkpoint_files(run, step, expected_actor_members=None):
    need(type(step) is int and step in STEPS, 'Only later reviewed save steps allowed')
    run = Path(run)
    actor = run / f'checkpoints/actor/global_step_{step}'
    native = actor / 'training_state'
    need(actor.resolve(strict=True) == actor and native.resolve(strict=True) == native,
         'Physical checkpoint directories required')
    names = {f'rank_{rank}{ext}' for rank in range(8) for ext in ('.pt', '.json')}
    need({p.name for p in native.iterdir()} == names, 'Missing/extra native rank files')
    files = {}
    for path in sorted(actor.rglob('*')):
        if path.is_dir():
            need(path == native and not path.is_symlink(), 'Unexpected HF directory')
            continue
        record, _ = stable_file(path, run)
        files['actor/' + str(path.relative_to(actor))] = record
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json'}
    need(all('actor/' + name in files for name in required), 'HF config/tokenizer/index incomplete')
    if expected_actor_members is not None:
        need({role for role in files if not role.startswith('actor/training_state/')}
             == set(expected_actor_members), 'HF member inventory differs from actual accepted architecture')
    index = json_file(actor / 'model.safetensors.index.json', run,
                      files['actor/model.safetensors.index.json']['sha256'])[0]
    need(isinstance(index.get('weight_map'), dict) and len(index['weight_map']) == 435,
         'Complete HF435 tensor index required')
    shards = set(index['weight_map'].values())
    need(shards and all(type(name) is str and Path(name).name == name
                        and name.endswith('.safetensors') for name in shards), 'Invalid HF shard paths')
    actual = {p.name for p in actor.iterdir() if p.is_file() and p.name.endswith('.safetensors')}
    need(shards == actual, 'Missing/extra HF shards')
    for name in shards:
        need('actor/' + name in files, 'HF shard not inventoried')
    driver = run / f'full_state/global_step_{step}'
    need(driver.resolve(strict=True) == driver and {p.name for p in driver.iterdir()} == {'driver.pt', 'driver.json'},
         'Later driver directory must contain original PT/JSON, not fabricated manifest')
    for name in ('driver.pt', 'driver.json'):
        files['driver/' + name] = stable_file(driver / name, run)[0]
    algo = run / f'checkpoints/algorithm/global_step_{step}/state.json'
    files['algorithm'] = stable_file(algo, run)[0]
    member_unique(files)
    return files


def checkpoint_reports(files, step, context):
    run = context['run']
    ranks = []
    for rank in range(8):
        rec = files[f'actor/training_state/rank_{rank}.json']
        r = json_file(rec['physical_path'], run, rec['sha256'])[0]
        pt = files[f'actor/training_state/rank_{rank}.pt']
        accepted = context['workers'][rank]
        need(r['version'] == 3 and r['algorithm'] == 'pure_opsd'
             and r['status'] == 'FULL_STATE_SERIALIZED_OK'
             and (r['rank'], r['world_size'], r['step']) == (rank, 8, step)
             and r['model_state_dict_type'] == 'SHARDED_STATE_DICT'
             and r['model_dtensor_entries'] == 435 and r['optimizer_state_entries'] > 0
             and r['scheduler_last_epoch'] == step, 'Wrong later rank status/clock/schema')
        need(r['pid'] == accepted['pid'] and r['object_ids'] == accepted['object_ids']
             and r['path'] == pt['physical_path']
             and r['sha256'] == r['state_sha256'] == pt['sha256']
             and re.fullmatch('[0-9a-f]{64}', r['state_digest']), 'Rank same-object/file binding differs')
        teacher = r['teacher']
        need(teacher['counters']['cuda_update_entries'] == step
             and teacher['parameter_count'] == 3085938688
             and teacher['execution_device'] == 'cuda' and teacher['storage_device'] == 'cpu',
             'Saved teacher identity/outer clock differs')
        ranks.append(r)
    dr = files['driver/driver.json']
    driver = json_file(dr['physical_path'], run, dr['sha256'])[0]
    need(driver['status'] == 'DRIVER_STATE_SERIALIZED_OK' and driver['step'] == step
         and driver['pid'] == DRIVER_PID and driver['uid'] == os.getuid()
         and driver['job_id'] == JOB
         and driver['state_sha256'] == files['driver/driver.pt']['sha256']
         and re.fullmatch('[0-9a-f]{64}', driver['digest']), 'Wrong later driver status/clock/file binding')
    ar = files['algorithm']
    algo = json_file(ar['physical_path'], run, ar['sha256'])[0]
    need(algo == dict(version=1, method='pure_opsd', global_step=step,
                     distillation=context['config']['baselines']['distillation'],
                     resume_scope='same_live_objects_same_topology', outcome_rewards_train_policy=False),
         'Saved pureOPSD algorithm/config differs')
    return ranks, driver


def execution_identity():
    """Current CPU process identity only; does not pretend it is saved training."""
    cgroup = Path('/proc/self/cgroup')
    return dict(pid=os.getpid(), uid=os.getuid(), node=os.uname().nodename,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                slurm_step_id=os.environ.get('SLURM_STEP_ID'),
                cgroup=cgroup.read_text() if cgroup.exists() else None,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                scope='OBSERVED_INSPECTION_PROCESS_NOT_SAVED_TRAINING_IDENTITY')


def parser_anchors(parser):
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--accepted5-manifest', required=True)
    parser.add_argument('--accepted5-manifest-sha256', required=True)
    parser.add_argument('--accepted5-receipt', required=True)
    parser.add_argument('--accepted5-receipt-sha256', required=True)
    parser.add_argument('--public-key', required=True)
    parser.add_argument('--public-key-sha256', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--native-import-closure', required=True)
    parser.add_argument('--native-import-closure-sha256', required=True)
    parser.add_argument('--training-log', required=True)
    parser.add_argument('--training-log-sha256', required=True)
    parser.add_argument('--training-log-prefix-bytes', required=True, type=int)
    parser.add_argument('--driver-log', required=True)
    parser.add_argument('--driver-log-sha256', required=True)
    parser.add_argument('--driver-log-prefix-bytes', required=True, type=int)


def observed_prefixes(args, context):
    # Driver files normally live in the owned short Ray runtime, not in RUN.
    driver_path = Path(args.driver_log)
    allowed = (Path('/tmp/o4e-3261/ray'), context['run'])
    need(any(driver_path.is_relative_to(root) for root in allowed), 'Driver log outside actual owned runtime/run')
    if driver_path.is_relative_to(allowed[0]):
        need(driver_path == LIVE_DRIVER_LOG, 'Runtime input is not the actual original driver log')
    draw, drec = prefix_file(driver_path, None, args.driver_log_prefix_bytes, args.driver_log_sha256)
    araw, arec = prefix_file(args.training_log, context['run'], args.training_log_prefix_bytes,
                             args.training_log_sha256)
    need((drec['device'], drec['inode']) != (arec['device'], arec['inode']), 'Driver/aggregate log alias')
    driver, aggregate = parse_records(draw, context), parse_records(araw, context)
    drec['origin_scope'] = ('ACTUAL_FIXED_DRIVER_LOG_PATH' if driver_path == LIVE_DRIVER_LOG
                           else 'SUPPLIED_IMMUTABLE_RUN_COPY_REQUIRES_ORIGINAL_PROVENANCE_JOIN')
    return driver, aggregate, drec, arec


def verify_dataset_files(context):
    manifest = context['common_manifest']
    splits = manifest['splits']
    need(splits['train']['rows'] == 169028 and splits['validation']['rows'] == 512
         and splits['stage']['rows'] == 64 and splits['final']['rows'] == 51713
         and splits['validation']['source_counts'] == DEV_COUNTS
         and splits['stage']['source_counts'] == STAGE_COUNTS
         and splits['final']['source_counts'] == FINAL_COUNTS, 'Shared manifest counts differ')
    config = context['config']
    paths = {'train': config['data']['train_files'], 'validation': config['data']['val_files'],
             'stage': config['baselines']['opsd_e1']['stage_files'], 'final': config['data']['final_val_files']}
    result = {}
    for role, path in paths.items():
        expected = ASSETS / 'common-data' / splits[role]['path']
        need(Path(path) == expected and expected.name == {'train': 'train.parquet',
             'validation': 'validation.parquet', 'stage': 'stage.parquet', 'final': 'final.parquet'}[role],
             'Evaluation/train data path differs from shared artifact')
        rec = stable_file(expected, ASSETS / 'common-data', splits[role]['sha256'])[0]
        result[role] = dict(file=rec, manifest_rows=splits[role]['rows'],
                            manifest_source_counts=splits[role]['source_counts'],
                            manifest_question_ids_sha256=splits[role]['question_ids_sha256'])
    return dict(manifest_sha256=DATA_SHA, actual_files=result,
                scope='ACTUAL_PARQUET_FULL_SHA_AND_SIGNED_MANIFEST_IDENTITY_NOT_RAW_GENERATION_REPLAY')
