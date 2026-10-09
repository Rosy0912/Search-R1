"""Reviewable, exclusive publication of an externally reviewed SOURCE080 release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aligned_first5_cpu_evidence as cpu
ROOT = cpu.ROOT
SOURCE = cpu.SOURCE
SCHEMA = 'aligned-first5-external-release-publication-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return cpu.regular(Path(path), ROOT).read_bytes()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'explicit reviewed SHA256 required')
    raw = read(path)
    require(sha_bytes(raw) == expected, 'reviewed bytes changed: ' + str(path))
    return raw


def stable_process(value):
    return {key: value[key] for key in ('pid', 'start_ticks', 'command', 'uids', 'executable', 'selected_env')}


def source_gate():
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'unoptimized interpreter required')
    require(HERE == ROOT / 'tools/deployment/first5-node11-3357' and Path(cpu.__file__).resolve() == HERE / 'aligned_first5_cpu_evidence.py', 'actual deployed publisher/extractor required')
    cpu.cpu_only()
    return cpu.bind_source()


def check_record(record, owned=True, originals=None):
    path = (cpu.regular_member(record['path'], ROOT, originals) if owned and originals is not None
            else cpu.regular(record['path'], ROOT if owned else None))
    require(cpu.binding(path) == record, 'original evidence changed: ' + str(path))
    return path


def validate_candidate(receipt_path, receipt_sha, report_sha, cpu_sha, gate):
    """Never manufactures approval: gate consumes precisely the external receipt."""
    raw = pinned(receipt_path, receipt_sha)
    receipt = json.loads(raw)
    require(receipt.get('publication_schema') == SCHEMA, 'external publication schema required')
    require(receipt.get('source_commit') == SOURCE and receipt.get('baseline_aligned') is True, 'source080 aligned external receipt required')
    submit_path = ROOT / 'scheduler-head/formal-submit-receipt.json'
    submit_raw = read(submit_path)
    submit = json.loads(submit_raw)
    who = cpu.identity(submit, ROOT)
    run = who['run']
    require(receipt.get('run_root') == str(run), 'receipt is not the actual new formal RUN')
    require(receipt.get('formal_submission') == cpu.binding(submit_path), 'actual formal submission bytes not signed')
    require(isinstance(submit.get('files'), dict) and bool(submit['files']), 'actual deployed formal source pins required')
    for name, expected in submit['files'].items():
        require(Path(name).name == name, 'unsafe formal source name')
        require(cpu.digest(cpu.regular(cpu.formal_tool_path(name), ROOT)) == expected, 'formal source changed')
    review = receipt.get('independent_review', {})
    pinned(review.get('path', ''), report_sha)
    require(review.get('sha256') == report_sha, 'independent report SHA differs')
    cpu_binding = receipt.get('cpu_evidence', {})
    proof = json.loads(pinned(cpu_binding.get('path', ''), cpu_sha))
    require(cpu_binding == cpu.binding(cpu_binding['path']) and cpu_binding['sha256'] == cpu_sha, 'CPU evidence bytes not signed')
    require(proof.get('status') == 'CPU_EVIDENCE_COLLECTED_NOT_INDEPENDENTLY_REVIEWED_NOT_RELEASE'
            and proof.get('acceptance') is False and proof.get('release_receipt') is False
            and proof.get('source_commit') == SOURCE and proof.get('root') == str(ROOT)
            and proof.get('issues') == [], 'complete actual CPU evidence required')
    require(proof.get('actual_submission') == receipt['formal_submission'], 'CPU extraction belongs to another submission')
    require(proof.get('cohost_watch',{}).get('scope')=='ACTUAL_COHOST_RESERVE_PREFIX_NOT_INDEPENDENT_ACCEPTANCE','actual node11 resource prefix required')
    cpu.verify_watch_prefix(proof['cohost_watch'])
    runtime = proof.get('runtime', {})
    require(runtime.get('platform') == 'linux' and runtime.get('cuda_visible_devices') == ''
            and runtime.get('cuda_available') is False and runtime.get('cuda_initialized') is False, 'actual CPU-only extraction runtime required')
    first = proof['first5']
    original_stats = {}
    def observe(path):
        path = Path(path)
        stat = path.stat()
        original_stats[str(path)] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    for path in (receipt_path, submit_path, review['path'], cpu_binding['path'],proof['cohost_watch']['bootstrap']['path'],proof['cohost_watch']['snapshot']['path']):
        observe(path)
    manifest = run / 'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json'
    observe(manifest)
    manifest_doc = json.loads(read(manifest))
    originals = cpu.validated_original_closure(manifest_doc,gate,run,first.get('driver_pid'))
    for row in receipt.get('evidence', []):
        observe(cpu.regular_member(row['path'], ROOT, originals))
    snapshot = receipt.get('train_log_prefix', {}).get('snapshot_path')
    if snapshot:
        observe(cpu.regular(snapshot, ROOT))
    require(first.get('manifest') == cpu.binding(manifest) and first.get('source_commit') == SOURCE
            and first.get('run_root') == str(run) and first.get('job') == who['job'] and first.get('token') == who['token'], 'CPU first5 identity differs')
    require(first.get('train_log_prefix') == receipt.get('train_log_prefix')
            and first.get('pause_nonce') == receipt.get('pause_nonce'), 'CPU reviewed prefix/pause differs')
    manifest_doc = json.loads(read(manifest))
    require(first.get('original_member_bindings') == manifest_doc['files'], 'CPU original checkpoint member inventory differs')
    require(first.get('native_roundtrip_reports') == manifest_doc['rank_roundtrips'], 'CPU native GPU producer reports differ')
    hf = proof.get('cpu_hf', {})
    require(hf.get('status') == 'ACTUAL_CPU_HF_CHECKED_NOT_RELEASE' and hf.get('scope') == 'official'
            and hf.get('parameters') == 3085938688 and hf.get('tied_input_output_storage') is True
            and hf.get('all_parameters_finite') is True
            and hf.get('forward', {}).get('device') == 'cpu' and hf['forward'].get('finite') is True
            and hf.get('parameter_delta', {}).get('all_official_and_export_tensors_finite') is True
            and type(hf['parameter_delta'].get('changed_elements')) is int and hf['parameter_delta']['changed_elements'] > 0
            and all(not hf.get('loading_info', {}).get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), 'official actual finite CPU HF load/forward/delta required')
    hf_records = proof.get('hf_files', [])
    require(len(hf_records) == 12 and len({r['path'] for r in hf_records}) == 12, 'all twelve actual fifth HF files required')
    for record in hf_records:
        require(run / 'checkpoints/actor/global_step_5' in Path(record['path']).parents, 'foreign HF export')
        check_record(record)
        observe(record['path'])
    # Official large assets were already fully scanned by the CPU extractor and
    # independent reviewer. Bind their pinned original manifest, not rehash them.
    official = proof.get('official_model', {})
    official_manifest = official.get('manifest', {})
    require(official_manifest.get('sha256') == cpu.MODEL_SHA, 'official initial model manifest pin differs')
    model = cpu.official_model_path(Path(first['config']['actor_rollout_ref']['model']['path']))
    require(official_manifest.get('path') == str(model / 'VERIFIED-MODEL-MANIFEST.json'), 'foreign official model manifest')
    check_record(official_manifest, owned=False)
    observe(official_manifest['path'])
    expected_model = json.loads(Path(official_manifest['path']).read_text())['files']
    expected_records = [dict(path=str(model / row['file']), sha256=row['sha256'], bytes=row['bytes']) for row in expected_model]
    require(len(expected_records) == 12 and official.get('files') == expected_records, 'complete pinned official model originals differ')
    records = first.get('original_member_bindings', [])
    require(records, 'original GPU/driver/EM/stage member bindings required')
    for record in records + proof.get('formal_tool_bindings', []) + proof.get('helper_files', []):
        check_record(record, originals=originals)
        observe(record['path'])
    helper_records = {Path(row['path']).name: row for row in proof.get('helper_files', [])}
    require(all(row['path'] == str(HERE / name) for name, row in helper_records.items()), 'CPU extraction used another helper directory')
    require(set(helper_records) == {'aligned_first5_cpu_evidence.py', 'aligned_first5_math.py', 'aligned-first5-math-provenance.json'}
            and helper_records['aligned_first5_cpu_evidence.py']['sha256'] == proof.get('helper_sha256'), 'actual complete extractor source bindings required')
    require({row['path']: row['sha256'] for row in proof.get('formal_tool_bindings', [])}
            == {str(cpu.formal_tool_path(name)): sha for name, sha in submit['files'].items()}, 'CPU deployed formal source inventory differs')
    require(proof.get('helper_files') and proof.get('formal_tool_bindings'), 'actual extractor/deployed tools original bindings absent')
    processes = first.get('processes', [])
    pid = first.get('driver_pid')
    require(type(pid) is int and pid > 1 and len(processes) == 9 and len({p['pid'] for p in processes}) == 9, 'actual driver plus eight original rank processes required')
    require(processes[0]['pid'] == pid and receipt.get('driver_birth') == stable_process(processes[0]), 'independent original driver birth not signed')
    require({p['pid'] for p in processes[1:]} == {r['pid'] for r in manifest_doc['rank_roundtrips']}, 'original native rank PIDs differ')
    for before in processes:
        require(stable_process(cpu.process_identity(before['pid'], run, who['job'])) == stable_process(before), 'original paused process died/reused/changed')
    def check_live_pause():
        live = read(run / 'train.log').decode('utf-8')
        pauses = cpu.marked(live, 'FRESH200_STEP5_PAUSED_JSON=')
        require(len(pauses) == 1 and pauses[0].get('pid') == pid and pauses[0].get('manifest_path') == str(manifest)
                and pauses[0].get('manifest_sha256') == cpu.digest(manifest), 'actual original driver fifth pause absent')
        require(not cpu.marked(live, 'FRESH200_STEP5_RELEASED_JSON='), 'original driver already released')
        require(len(cpu.marked(live, 'FRESH200_ATTEMPT_JSON=')) == 5, 'live training no longer at the original five attempts')
    check_live_pause()
    # Actual source080 performs the complete native/driver/state/EM/stage/ledger,
    # fixed20/noSKIP/config/immutable-prefix/evidence validation; no copied gate.
    require(gate.validate_release(receipt_path, manifest, ROOT / 'repo', expected_driver_pid=pid) == receipt, 'actual source gate result differs')
    configs = [row for row in manifest_doc['files'] if Path(row['path']).name == 'resolved-config.json' and run in Path(row['path']).parents]
    require(len(configs) == 1, 'unique resolved config required')
    config = json.loads(read(configs[0]['path']))
    require(first.get('config') == config, 'CPU resolved configuration differs from original manifest')
    target = run / 'logs/fresh200-step5-independent-release.json'
    require(config['trainer']['fresh200_gate_receipt'] == str(target), 'actual configured release target differs')
    require(target.parent.is_dir() and target.parent.resolve() == target.parent, 'canonical existing gate directory required')
    require(pinned(receipt_path, receipt_sha) == raw, 'external receipt changed during validation')
    for path, token in original_stats.items():
        stat = cpu.regular(path).stat()
        require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == token, 'original changed during gate validation')
    pinned(cpu_binding['path'], cpu_sha)
    cpu.verify_watch_prefix(proof['cohost_watch'])
    pinned(review['path'], report_sha)
    require(cpu.binding(submit_path) == receipt['formal_submission'], 'formal submission changed during gate validation')
    check_live_pause()
    for before in processes:
        require(stable_process(cpu.process_identity(before['pid'], run, who['job'])) == stable_process(before), 'original process changed before publication')
    return dict(run=str(run), job=who['job'], token=who['token'], driver_pid=pid, target=str(target)), raw


def prepare(receipt, receipt_sha, report_sha, cpu_sha, output, gate):
    identity, raw = validate_candidate(receipt, receipt_sha, report_sha, cpu_sha, gate)
    output = Path(output)
    require(output.parent == ROOT / 'evidence' and output.parent.resolve() == output.parent and not output.is_symlink(), 'new owned plan directory required')
    output.mkdir()
    with (output / 'release-payload.json').open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    plan = dict(schema=SCHEMA, scope='REVIEWABLE_EXTERNAL_RECEIPT_PUBLICATION_NOT_SELF_ACCEPTANCE', source=SOURCE,
                identity=identity, receipt=str(Path(receipt)), receipt_sha256=receipt_sha,
                independent_report_sha256=report_sha, cpu_evidence_sha256=cpu_sha,
                publisher_sha256=cpu.digest(__file__), payload=cpu.binding(output / 'release-payload.json'))
    cpu.write_new(output / 'release-plan.json', plan)
    return plan


def publish(plan_path, plan_sha, payload_sha, gate):
    plan = json.loads(pinned(plan_path, plan_sha))
    require(plan.get('schema') == SCHEMA and plan.get('source') == SOURCE and plan.get('publisher_sha256') == cpu.digest(__file__), 'publication plan source/tool differs')
    raw = pinned(plan['payload']['path'], payload_sha)
    require(plan['payload'] == cpu.binding(plan['payload']['path']) and payload_sha == plan['receipt_sha256'], 'reviewed payload differs')
    target = Path(plan['identity']['target'])
    who = cpu.identity(json.loads(read(ROOT / 'scheduler-head/formal-submit-receipt.json')), ROOT)
    require(plan['identity']['run'] == str(who['run']) and plan['identity']['job'] == who['job']
            and plan['identity']['token'] == who['token']
            and target == who['run'] / 'logs/fresh200-step5-independent-release.json', 'plan is not the actual formal gate target')
    if target.exists() or target.is_symlink():
        require(read(target) == raw, 'existing gate receipt differs; never overwrite')
        return dict(status='IDENTICAL_RECEIPT_ALREADY_PRESENT_NO_WRITE', path=str(target), sha256=payload_sha)
    identity, original = validate_candidate(plan['receipt'], plan['receipt_sha256'], plan['independent_report_sha256'], plan['cpu_evidence_sha256'], gate)
    require(identity == plan['identity'] and original == raw, 'reviewed plan identity changed')
    return atomic_publish(target, raw)


def atomic_publish(target, raw):
    target = Path(target)
    require(target.parent.resolve() == target.parent and not target.is_symlink(), 'canonical gate target required')
    fd, name = tempfile.mkstemp(prefix='.first5-release-', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        # Hardlink is an atomic no-replace publication; consumers see full bytes.
        os.link(temporary, target, follow_symlinks=False)
        parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        temporary.unlink()
    return dict(status='EXTERNAL_REVIEWED_RECEIPT_PUBLISHED_NOT_TRAINING_COMPLETION', path=str(target), sha256=sha_bytes(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--receipt', type=Path, required=True)
    prep.add_argument('--receipt-sha256', required=True)
    prep.add_argument('--report-sha256', required=True)
    prep.add_argument('--cpu-evidence-sha256', required=True)
    prep.add_argument('--output', type=Path, required=True)
    pub = commands.add_parser('publish')
    pub.add_argument('--plan', type=Path, required=True)
    pub.add_argument('--plan-sha256', required=True)
    pub.add_argument('--payload-sha256', required=True)
    args = parser.parse_args()
    gate = source_gate()
    if args.action == 'prepare':
        result = prepare(args.receipt, args.receipt_sha256, args.report_sha256, args.cpu_evidence_sha256, args.output, gate)
    else:
        result = publish(args.plan, args.plan_sha256, args.payload_sha256, gate)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
