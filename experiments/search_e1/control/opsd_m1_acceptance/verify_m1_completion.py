"""Exclusive, CPU-only final M1 release. Never import models, query HTTP, or signal.

Read-only inputs; output is a NEW private directory. Source/model end hashing
happens once outside token loops. No absence-to-exit-code inference.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
VERSIONS = ('source100', 'update1', 'update5', 'epoch1')
REGION_SHA = '55bfd3f74aed1dd5b51bb5fa91244cc3940ecd02ac2c937c3282208942df311e'
PRESERVATION_SHA = 'd872cfc0da2329e319167f6d8d6ae85ac48ce309d21fd83d92c6b75bf61ed45b'
START_SHA = 'f8b1b80ca4301641a2dcb3e006a3a2ba349d2bfe8865ad880fabe181762998e5'
CASE_SHA = '1703b15b1a1b3c5b91b13422fa314ab9a3b83f7bf7bf8a85dd7afcdaecdbf082'
RUNTIME_MANIFEST_SHA = '3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc'
MAX_JSON = 16 << 20
SKIP_REASON = 'empty_response_or_mask_or_joint_context_limit_no_independent_truncation'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stat(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def file_proof(path, expected_sha=None, expected_stat=None):
    path = Path(path).resolve(); before = stat(path); h = hashlib.sha256()
    if expected_stat:
        require(all(before[k] == expected_stat[k] for k in before), 'Bound file metadata changed: ' + str(path))
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    require(stat(path) == before, 'File changed during fresh hash: ' + str(path))
    actual = h.hexdigest()
    require(expected_sha is None or actual == expected_sha, 'SHA mismatch: ' + str(path))
    return dict(path=str(path), stat=before, sha256=actual, fresh_cpu_hash=True)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite JSON numeric value')
    elif isinstance(value, dict):
        for x in value.values():
            finite_tree(x)
    elif isinstance(value, list):
        for x in value:
            finite_tree(x)


def json_file(path, expected_sha=None):
    path = Path(path).resolve(); before = stat(path)
    require(before['bytes'] <= MAX_JSON, 'JSON file exceeds bound')
    data = path.read_bytes()
    require(stat(path) == before, 'JSON changed while reading')
    sha = digest(data); require(expected_sha is None or sha == expected_sha, 'JSON SHA mismatch: ' + str(path))
    value = json.loads(data); finite_tree(value)
    return value, dict(path=str(path), stat=before, sha256=sha)


def inside(path, directory):
    path = Path(path).resolve(); directory = Path(directory).resolve()
    require(path == directory or directory in path.parents, 'Artifact escaped its preserved directory')
    return path


def load_region(path):
    path = Path(path).resolve(); proof = file_proof(path, REGION_SHA)
    spec = importlib.util.spec_from_file_location('m1_pinned_cpu_region', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path and stat(path) == proof['stat'], 'Region import source changed')
    # Validate all JSON numbers as each row is already streamed for the native
    # probe, without an extra token-file pass. Audited mask/loss function unchanged.
    original_lines = module.lines
    def checked_lines(*args, **kwargs):
        for row in original_lines(*args, **kwargs):
            finite_tree(row)
            if 'distributions' in row:
                has_teacher = row['teacher_student_clipped_kl'] is not None
                expected = {'student', 'base_ordinary'} | ({'teacher_privileged'} if has_teacher else set())
                require(set(row['distributions']) == expected, 'Token teacher/student distribution identity differs')
                for distribution in row['distributions'].values():
                    for field in ('target_logprob', 'target_probability', 'entropy'):
                        require(type(distribution[field]) in (int, float) and math.isfinite(distribution[field]), 'Missing/nonfinite distribution scalar')
                    require(0 <= distribution['target_probability'] <= 1 + 1e-6, 'Invalid recorded probability')
                    require(len(distribution['topk']) == 32, 'Expected topk32 missing')
                    for entry in distribution['topk']:
                        require(type(entry['token_id']) == int and type(entry['logprob']) in (int, float) and
                                type(entry['probability']) in (int, float), 'Missing distribution topk numeric field')
            yield row
    module.lines = checked_lines
    return module, proof


def process(pid):
    directory = Path('/proc') / str(pid)
    raw = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, uid=directory.stat().st_uid, starttick=int(raw[19]), state=raw[0])


def terminated(owned, reader=process):
    observations = []
    for identity in owned:
        try:
            current = reader(identity['pid'])
        except (FileNotFoundError, ProcessLookupError):
            observations.append(dict(identity=identity, state='ABSENT')); continue
        require((current['uid'], current['starttick']) != (identity['uid'], identity['starttick']),
                'Owned process remains present, including zombie: ' + str(identity['pid']))
        observations.append(dict(identity=identity, state='PID_REUSED', current=current))
    return observations


def no_orphans(run_id, uid):
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            values = (directory / 'environ').read_bytes().split(b'\0')
            if ('E1_RUN_ID=' + run_id).encode() in values:
                found.append(process(int(directory.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
        # A same-UID unreadable process makes a complete orphan claim unknown.
        except PermissionError:
            raise ValueError('Unable to inspect same-UID process for M1 orphan')
    require(not found, 'Owned M1 run still has descendants/orphans')
    return found


def validate_terminal(start, config, final, children, reading):
    require(start.get('verified_live') is True and start['m1_run_id'] == config['run_id'], 'Unbound live start proof')
    c = start['controller']; pane = start['pane_process']
    require(c['pid'] == config['controller_pid'] and c['uid'] == config['controller_uid'] == 1005,
            'Actual controller UID/PID differs')
    require(c['ppid'] == pane['pid'] and c['pgid'] == pane['pgid'], 'Live controller/pane ownership differs')
    require(c['state'] != 'Z' and pane['state'] != 'Z' and start['captured_unix'] >= config['started_unix'], 'Invalid live start time/state')
    require(final['status'] == 'WORKERS_COMPLETE' and final['run_id'] == config['run_id'] and
            final['controller_pid'] == c['pid'] and final['output'] == config['output'], 'Controller final not complete/bound')
    require(final['children'] == children and len(children) == 3 and len({x['pid'] for x in children}) == 3,
            'Final children changed/duplicated')
    exits = final['child_exit_codes']
    require(set(exits) == {str(x['pid']) for x in children} and all(type(x) == int and x == 0 for x in exits.values()),
            'Actual child exit0 evidence missing')
    require(final.get('no_process_signals') is True and final['ended_unix'] >= start['captured_unix'], 'Invalid final timing/ownership')
    require(reading['pane_id'] == start['tmux_pane_id'] and reading['pane_pid'] == pane['pid'] and
            reading['session_name'] == start['tmux_session'] and reading['pane_start_command'] == start['tmux_start_command'],
            'Actual tmux pane/start command identity differs')
    require(reading['pane_dead'] is True and type(reading['pane_dead_status']) == int and reading['pane_dead_status'] == 0,
            'Actual tmux terminal status0 is missing; absence is unknown')
    return dict(pid=c['pid'], uid=c['uid'], starttick=c['starttick'], exit_code=0)


def tmux_read(start):
    fields = ['session_name', 'pane_id', 'pane_pid', 'pane_dead', 'pane_dead_status', 'pane_start_command']
    fmt = '\t'.join('#{' + field + '}' for field in fields)
    raw = subprocess.check_output(['tmux', 'display-message', '-p', '-t', start['tmux_pane_id'], fmt], text=True).rstrip('\n')
    parts = raw.split('\t', 5); require(len(parts) == 6, 'Unknown actual tmux terminal reading')
    require(parts[3] in ('0', '1'), 'Unknown actual pane_dead')
    require(parts[4] != '', 'Empty actual pane_dead_status is unknown')
    return dict(zip(fields, [parts[0], parts[1], int(parts[2]), parts[3] == '1', int(parts[4]), parts[5]]))


def fresh_sources(root, manifest, expected_count):
    require(type(manifest) == dict and len(manifest) == expected_count, 'Frozen source manifest count differs')
    files = {}
    for relative, sha in sorted(manifest.items()):
        rel = Path(relative)
        require(not rel.is_absolute() and '..' not in rel.parts and rel.suffix != '.pyc', 'Unsafe source manifest path')
        path = inside(Path(root) / rel, root)
        files[relative] = file_proof(path, sha)
    return files


def fresh_weights(manifest):
    base = manifest['source100']; require(len(base['files']) == 12, 'Source100 must bind12 files')
    proof = {'source100': [], 'adapters': {}}
    require(len({x['relative_path'] for x in base['files']}) == 12, 'Duplicate source100 file')
    for item in base['files']:
        path = inside(item['path'], base['path'])
        require(path == (Path(base['path']) / item['relative_path']).resolve(), 'Source100 relative path mismatch')
        proof['source100'].append(file_proof(path, item['sha256'], item))
    require(set(manifest['adapters']) == set(VERSIONS), 'Required four adapters missing')
    for version in VERSIONS:
        item = manifest['adapters'][version]; directory = Path(item['path'])
        proof['adapters'][version] = {
            'config': file_proof(directory / 'adapter_config.json', item['adapter_config_sha256'], item['config_stat']),
            'model': file_proof(directory / 'adapter_model.safetensors', item['adapter_model_sha256'], item['model_stat'])}
    return proof


def check_skip(summary):
    require(summary.get('status') == 'skipped' and summary.get('reason') == SKIP_REASON, 'Unexplained probe skip')
    n, m = summary['response_tokens'], summary['valid_policy_tokens']
    require(type(n) == int and type(m) == int and 0 <= m <= n, 'Bad skipped counts')
    prefixes = [summary['ordinary_prefix_tokens']] + ([summary['privileged_prefix_tokens']] if summary['privileged_prefix_tokens'] is not None else [])
    require(all(type(x) == int and x >= 0 for x in prefixes), 'Bad skipped prefix counts')
    require(n == 0 or m == 0 or any(x + n > 8192 for x in prefixes), 'Skip has no actual empty/mask/context reason')
    require('tokens_file' not in summary, 'Skipped probe unexpectedly claims token evidence')
    return dict(status='allowed_skip', reason=SKIP_REASON, response_tokens=n, valid_policy_tokens=m)


def npy_header(path):
    with Path(path).open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'Invalid NPY magic')
        version = tuple(stream.read(2)); require(version in ((1, 0), (2, 0), (3, 0)), 'Unknown NPY version')
        size = struct.unpack('<H' if version == (1, 0) else '<I', stream.read(2 if version == (1, 0) else 4))[0]
        require(size <= 65536, 'NPY header exceeds bound')
        header = ast.literal_eval(stream.read(size).decode('latin1' if version != (3, 0) else 'utf8'))
        return header, stream.tell()


def full_vocab(summary, token_directory, npy_cache, counts):
    info = summary.get('full_vocabulary_dump'); require(type(info) == dict, 'Complete probe lacks full vocabulary status')
    status = info.get('status'); counts[status] += 1
    if status == 'complete':
        teacher = summary['reference_status'] == 'successful_original_reference'
        expected = {d + '__' + q for d in (['student', 'base_ordinary'] + (['teacher_privileged'] if teacher else [])) for q in ('probabilities', 'logprobs')}
        require(set(info['files']) == expected, 'Full vocabulary dump files differ')
        proofs = {}
        for key, item in sorted(info['files'].items()):
            path = inside(item['path'], token_directory)
            require(item['dtype'] == 'float32' and item['shape'] == [summary['response_tokens'], 151936], 'Dump dtype/shape changed')
            require(key == item['distribution'] + '__' + item['quantity'], 'Dump label changed')
            if str(path) not in npy_cache:
                p = file_proof(path, item['sha256']); header, offset = npy_header(path)
                require(header['descr'] == '<f4' and header['fortran_order'] is False and list(header['shape']) == item['shape'], 'Actual NPY header mismatch')
                require(p['stat']['bytes'] == item['bytes'] == offset + summary['response_tokens'] * 151936 * 4, 'Actual NPY byte count mismatch')
                require(stat(path) == p['stat'], 'NPY changed after header read')
                npy_cache[str(path)] = p
            require(npy_cache[str(path)]['sha256'] == item['sha256'], 'Reused NPY content mismatch')
            proofs[key] = npy_cache[str(path)]
        return dict(status=status, files=proofs, fresh_semantic_array_reconstruction=False)
    if status == 'cap_skip':
        require(info['bytes_requested_with_header_allowance'] > 0 and info['worker_bytes_reserved'] >= 0 and
                info['worker_bytes_reserved'] + info['bytes_requested_with_header_allowance'] > info['worker_cap_bytes'], 'Unjustified full vocabulary cap_skip')
    else:
        require(status in ('not_requested', 'not_requested_generated_response_scope'), 'Unknown full vocabulary status')
    return dict(status=status, explicit_producer_reason=info.get('reason'))


def checked_probe(module, directory, result_summary, name, version, case, response, spans, origin, npy_cache, counts, vocabulary):
    path = directory / ('token-summary-' + version + '-' + name + '.json')
    summary, proof = json_file(path)
    require(summary == result_summary, 'Embedded result and persisted probe summary differ')
    has_teacher = case['original_pair'] is not None
    require(summary['version'] == version and summary['case_id'] == case['case_id'] and summary['response_origin'] == origin,
            'Probe version/case/origin differs')
    require(summary['reference_status'] == ('successful_original_reference' if has_teacher else 'not_applicable_no_successful_reference'), 'Reference status changed')
    require(summary['original_training_objective'] is (origin == 'preserved_student'), 'Probe original-objective flag differs')
    require(summary['kind'] == 'fixed_response_teacher_forcing' and summary['actual_autoregressive_generation'] is False and
            summary['implicit_temperature'] == 1.0 and summary['contribution_upper_clip'] == 10.0 and summary['topk'] == 32 and
            summary['max_context'] == 8192 and summary['token_reduction'] == 'valid policy token mean per trajectory', 'Native diagnostic objective/budget fields differ')
    if name.startswith('original-'):
        require(summary['preserved_trajectory_name'] == name, 'Preserved trajectory name mismatch')
    if summary['status'] == 'skipped':
        checked = check_skip(summary); counts['allowed_skip'] += 1
        return dict(summary=proof, check=checked)
    require(summary['status'] == 'complete' and summary.get('all_token_metrics_finite') is True, 'Incomplete/nonfinite probe')
    tokens = directory / ('tokens-' + version + '-' + name) / 'tokens.jsonl'
    require(Path(summary['tokens_file']['path']).resolve() == tokens.resolve(), 'Token path is not the expected artifact')
    checked = module.probe(summary, tokens, response, spans)
    require(checked['status'] == 'complete' and checked['reconstruction_passed'] is True, 'Native probe failed')
    if not has_teacher:
        for key in ('trajectory_mean', 'frozen_pair_loss', 'per_token_contribution_sum_float64'):
            require(summary.get(key) is None, 'Hard trajectory has fabricated teacher scalar')
        require(summary['frozen_pair_loss_check']['status'] == 'not_applicable_no_successful_reference', 'Hard native scalar status differs')
    counts['complete'] += 1; counts['token_rows'] += checked['response_tokens']
    # Keep only numerical/hash proof, never original response/QA text in output.
    compact = {k: checked[k] for k in ('status', 'valid_tokens', 'response_tokens', 'response_sha256', 'policy_spans_sha256',
                                      'token_mask_alignment_sha256', 'token_file', 'reconstruction_passed', 'teacher_present')}
    dump = full_vocab(summary, tokens.parent, npy_cache, vocabulary)
    return dict(summary=proof, check=compact, full_vocabulary=dump)


def generation(path, embedded=None):
    value, proof = json_file(path)
    if embedded is not None:
        require(value == embedded, 'Persisted/embedded teacher generation differs')
    require(value['kind'] == 'actual_autoregressive_generation', 'Generation kind differs')
    if value.get('status') != 'not_applicable_no_successful_reference':
        require(value['temperature'] == 0.0 and value['max_new_tokens_per_turn'] == 500 and value['max_context'] == 8192 and
                value['max_search_calls'] == 4 and value['topk_documents'] == 3 and value['teacher_forced'] is False,
                'Actual generation budget/model mode differs')
    return value, proof


def case_probes(module, case, result, directory, npy_cache, counts, vocabulary):
    require(set(result['versions']) == set(VERSIONS), 'Missing fixed-weight version')
    pair = case['original_pair']; hard = pair is None; item = dict(case_id=case['case_id'], probes={}, generations={}, teacher_invariance={})
    teacher, tproof = generation(directory / 'teacher-generation.json', result['teacher_generation']); item['generations']['teacher'] = tproof
    if hard:
        require(teacher['status'] == 'not_applicable_no_successful_reference' and teacher['privileged_reference'] is False and
                result['actual_teacher_response_teacher_forced_probe'] is None, 'Hard teacher not strictly inapplicable/null')
        trajectories = [('original-failed-sibling-' + str(x['row']['sibling']), x['row']['response'], x['row']['policy_spans']) for x in case['original_siblings']]
        require(len(trajectories) == 5 and {x[0] for x in trajectories} == {'original-failed-sibling-' + str(n) for n in range(5)}, 'Hard five siblings missing')
    else:
        require(teacher['privileged_reference'] is True and teacher['active_adapter'] == 'disabled' and
                teacher['fixed_weight_source'] == 'source100' and teacher['role'] == 'frozen_source100_privileged_teacher',
                'Teacher fixed-base disabled-adapter identity differs')
        trajectories = [('original-mined-student', pair['row']['student'], pair['row']['student_policy_spans'])]
        item['probes']['teacher-own-generation'] = checked_probe(module, directory, result['actual_teacher_response_teacher_forced_probe'],
            'actual-teacher-generation', 'source100', case, teacher['response'], teacher['policy_spans'], 'new_teacher_generation', npy_cache, counts, vocabulary)
    for version in VERSIONS:
        v = result['versions'][version]; require(len(v['token_probes']) == len(trajectories), 'Expected original trajectory slots missing')
        gen, gproof = generation(directory / ('student-generation-' + version + '.json')); item['generations'][version] = gproof
        require(gen['active_adapter'] == version and gen['fixed_weight_source'] == 'source100' and gen['role'] == 'ordinary_student', 'Student fixed-weight identity differs')
        require(gen['privileged_reference'] is False, 'Ordinary student generation used a privileged reference')
        require(gen['protocol_em'] == v['generation_em'] and gen['termination'] == v['generation_termination'] and gen['search_calls'] == v['generation_search_calls'], 'Embedded generation metrics differ')
        item['probes'][version] = {}
        for (name, response, spans), summary in zip(trajectories, v['token_probes']):
            item['probes'][version][name] = checked_probe(module, directory, summary, name, version, case, response, spans,
                'preserved_failed_sibling' if hard else 'preserved_student', npy_cache, counts, vocabulary)
        item['probes'][version]['actual-student-generation'] = checked_probe(module, directory,
            v['actual_student_response_teacher_forced_probe'], 'actual-student-generation', version, case,
            gen['response'], gen['policy_spans'], 'new_student_generation', npy_cache, counts, vocabulary)
    for name, _, _ in trajectories:
        proofs = [item['probes'][v][name]['check'] for v in VERSIONS]
        if all(p['status'] == 'complete' for p in proofs):
            require(len({(p['response_sha256'], p['policy_spans_sha256'], p['token_mask_alignment_sha256']) for p in proofs}) == 1,
                    'Original shared history/token alignment differs across checkpoints')
            if not hard:
                for version in VERSIONS[1:]:
                    left = proofs[0]['token_file']; right = item['probes'][version][name]['check']['token_file']
                    item['teacher_invariance'][version] = module.teacher_invariance(left['path'], right['path'], left, right)
            else:
                item['teacher_invariance']['hard-' + name] = {'status': 'not_applicable_no_successful_reference'}
        else:
            item['teacher_invariance'][name] = {'status': 'unknown_due_to_allowed_probe_skip'}
    return item


def source_stats_unchanged(tree):
    if isinstance(tree, dict):
        if {'path', 'stat', 'sha256'}.issubset(tree):
            require(stat(tree['path']) == tree['stat'], 'Previously checked immutable file changed by final end check')
        else:
            for value in tree.values():
                source_stats_unchanged(value)
    elif isinstance(tree, list):
        for value in tree:
            source_stats_unchanged(value)


def exit_evidence(run_id, controller, reading, preservation, start_proof):
    return dict(schema='m1-actual-owned-tmux-terminal-v1', m1_run_id=run_id, controller=controller,
                pane_dead=True, pane_dead_status=0, observed_unix=time.time(), actual_tmux_reading=reading,
                preservation_receipt=preservation, controller_start_proof=start_proof,
                no_exit_code_inferred_from_absence=True)


def release_references(final_proof, exit_path):
    return dict(controller_final=final_proof, tmux_exit_proof=file_proof(exit_path))


def execute(config, output):
    started = time.time(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    verifier_proof = file_proof(Path(__file__))
    progress = output / 'progress.jsonl'
    def stage(name, **extra):
        with progress.open('ab') as stream:
            stream.write(canonical(dict(phase=name, unix=time.time(), **extra)) + b'\n')
    try:
        require(os.getuid() == config['uid'] == 1005, 'Wrong actual UID')
        run = Path(config['run_dir']).resolve(); require(run.name == 'debug-' + config['m1_run_id'], 'Run directory ID differs')
        start, sp = json_file(config['controller_start_proof'], START_SHA)
        preservation, pp = json_file(config['preservation_receipt'], PRESERVATION_SHA)
        current, cp = json_file(run / 'config.json', start['config_sha256'])
        require(current['run_id'] == config['m1_run_id'] and Path(current['output']).resolve() == run, 'Configuration run mismatch')
        require(Path(start['config_path']).resolve() == run / 'config.json' and
                Path(start['controller']['argv'][1]).resolve() == Path(config['diagnostic_source']).resolve() / 'launch_gpu_debug.py',
                'Live controller/config/source path identity differs')
        argv = current['worker_argv_common']
        for field, option in (('runtime', '--runtime'), ('cases_dir', '--cases-dir'), ('weight_manifest', '--weight-manifest')):
            require(argv.count(option) == 1 and Path(argv[argv.index(option)+1]).resolve() == Path(config[field]).resolve(),
                    'Acceptance/frozen worker input path differs: ' + field)
        final, fp = json_file(run / 'controller-final.json'); children, children_proof = json_file(run / 'children-start.json')
        reading = tmux_read(start); controller = validate_terminal(start, current, final, children, reading)
        owned = [controller] + children + [start['pane_process']]
        terminal = terminated(owned); no_orphans(current['run_id'], config['uid'])
        stage('TERMINAL_EXIT0_VERIFIED')
        module, region_proof = load_region(config['region_helper'])
        debug_manifest, debug_mp = json_file(config['debug_source_manifest'])
        require(debug_manifest == current['source_manifest'] and debug_manifest['launch_gpu_debug.py'] == start['controller_source_sha256'], 'Startup diagnostic source binding differs')
        runtime_manifest, runtime_mp = json_file(config['runtime_manifest'], RUNTIME_MANIFEST_SHA)
        weight_manifest, wmproof = json_file(config['weight_manifest'], current['weight_manifest_sha256'])
        require(Path(argv[argv.index('--model')+1]).resolve() == Path(weight_manifest['source100']['path']).resolve(), 'Worker/base model path differs')
        for version in VERSIONS:
            option = '--adapter-' + version
            require(Path(argv[argv.index(option)+1]).resolve() == Path(weight_manifest['adapters'][version]['path']).resolve(), 'Worker/fixed adapter path differs')
        case_manifest, cmproof = json_file(Path(config['cases_dir']) / 'manifest.json')
        require(case_manifest['selected_cases'] == 192 and case_manifest['per_cell'] == 32 and case_manifest['seed'] == 42 and case_manifest['strict_reward_checked'] is True, 'Case strata selector changed')
        index, cproof = module.index_cases(Path(config['cases_dir']) / 'cases.jsonl')
        require(len(index) == 192 and cproof['sha256'] == case_manifest['cases']['sha256'] == current['cases_sha256'] == CASE_SHA, '192 case source SHA changed')
        source = {'debug': fresh_sources(config['diagnostic_source'], debug_manifest, 15),
                  'runtime': fresh_sources(config['runtime'], runtime_manifest, 305)}
        for relative, identity in case_manifest['science'].items():
            require(source['runtime'][relative]['sha256'] == identity['sha256'], 'Selector/native scientific source differs')
        stage('320_SOURCE_FILES_FRESH_HASHED')
        weights = fresh_weights(weight_manifest); stage('12_BASE_AND_8_ADAPTER_FILES_FRESH_HASHED')
        counts, vocabulary, cells, seen = Counter(), Counter(), Counter(), set(); npy_cache = {}; workers = []
        artifact_path = output / 'case-artifact-proofs.jsonl'
        with artifact_path.open('xb') as artifact_stream:
            for wi, child in enumerate(children):
                require(child['uid'] == 1005 and child['physical_gpu'] == [4, 5, 6][wi] and child['gpu_uuid'] == current['gpu_uuids'][wi], 'Worker owned GPU identity differs')
                require(child['argv'] == argv + ['--worker-index', str(wi)], 'Actual child scientific argv differs')
                directory = run / ('worker-' + str(wi)); status, stproof = json_file(directory / 'status.json')
                require(status['status'] == 'COMPLETE' and status['assigned_cases'] == status['completed_cases'] == 64 and status['workers'] == 3 and status['worker_index'] == wi, 'Worker lacks complete64')
                require(status['pid'] == child['pid'] and status['uid'] == child['uid'] and status['proc_starttick'] == child['starttick'] and status['ppid'] == controller['pid'] and status['pgid'] == child['pid'], 'Worker actual PID/tick/UID ownership differs')
                require(status['no_training'] is True and status['seed'] == 42 and status['actual_generation_temperature'] == 0.0 and status['diagnostic_sample_only'] is True, 'Diagnostic fixed-model semantics changed')
                require(status['cases']['sha256'] == cproof['sha256'], 'Worker source cases changed')
                require(0 <= status['full_vocab_worker_bytes_reserved'] <= current['full_vocab_global_cap_bytes']//3,
                        'Worker full-vocabulary reservation exceeds declared partition')
                results_proof = {}; worker_seen = set()
                for result in module.lines(directory / 'case-results.jsonl', proof=results_proof):
                    key = result['case_id']; require(key in index and key not in seen and result['status'] == 'complete', 'Duplicate/unknown/incomplete completed result')
                    case = module.indexed_case(Path(config['cases_dir']) / 'cases.jsonl', index[key], cproof)
                    require(result['case_index'] == case['case_index'] and case['case_index'] % 3 == wi and result['source_case_sha256'] == case['case_semantic_sha256'] and
                            (result['dataset'], result['group']) == (case['dataset'], case['group']), 'Result semantic identity/worker assignment differs')
                    require(case['case_index'] in range(192), 'Case index out of range')
                    item = case_probes(module, case, result, directory / key, npy_cache, counts, vocabulary)
                    artifact_stream.write(canonical(item) + b'\n'); seen.add(key); worker_seen.add(key); cells[(case['dataset'], case['group'])] += 1
                    if len(seen) % 8 == 0:
                        artifact_stream.flush(); stage('TOKEN_AND_ARRAY_AUDIT', completed_cases=len(seen), complete_probes=counts['complete'], allowed_skips=counts['allowed_skip'])
                require(len(worker_seen) == 64 and results_proof['sha256'] == status['results_sha256'] and set(status['execution_order']) == worker_seen and len(status['execution_order']) == 64, 'Worker final count/order/SHA mismatch')
                workers.append(dict(index=wi, completed_cases=64, status=stproof, results=results_proof, child=child))
        require(seen == set(index) and cells == Counter({(d, g): 32 for d in ('nq', 'hotpotqa') for g in ('correct_correct', 'correct_wrong', 'all5_wrong')}), 'Full192/sixcell32 coverage incomplete')
        require(counts['complete'] + counts['allowed_skip'] == 2688, 'Expected2688 nominal probe slots differ')
        # Explicitly reused semantic checks are only the selected existing arrays;
        # all dump files still receive one fresh end content hash in npy_cache.
        prior = []
        for entry in config.get('prior_full_vocab_proofs', []):
            value, proof = json_file(entry['path'], entry['sha256']); prior.append(dict(proof=proof, schema=value.get('schema')))
            if value.get('schema') == 'existing-full-vocab-dump-independent-cpu-reconstruction-v1':
                require(value.get('passed') is True, 'Prior selected array semantics failed')
                for check in value['checks']:
                    require(check['passed'] is True, 'Prior selected version semantics failed')
                    for key, path in check['fp32_dump_paths'].items():
                        require(path in npy_cache and npy_cache[path]['sha256'] == check['bound_dump_sha256'][key], 'Prior selected array semantic identity differs from end hash')
        immutable = dict(source=source, weights=weights, case=cproof, manifests=[sp, pp, cp, fp, children_proof, region_proof, debug_mp, runtime_mp, wmproof, cmproof, verifier_proof], workers=workers, npy_files=list(npy_cache.values()))
        source_stats_unchanged(immutable)
        # Recheck actual terminal and orphan conditions after the long CPU audit.
        rereading = tmux_read(start); require(rereading == reading, 'Tmux terminal identity changed during audit')
        terminated(owned); no_orphans(current['run_id'], config['uid'])
        exit_path = output / 'tmux-exit-proof.json'
        exit_proof = exit_evidence(current['run_id'], controller, reading, pp, sp)
        write_exclusive(exit_path, exit_proof)
        release = dict(schema='m1-complete-independent-cpu-release-v1', accepted=True, m1_run_id=current['run_id'],
                       completed_cases=192, cells={d + '/' + g: n for (d, g), n in sorted(cells.items())}, worker_counts=[64]*3,
                       controller=controller, **release_references(fp, exit_path),
                       weight_manifest_sha256=wmproof['sha256'], debug_source_manifest_sha256=debug_mp['sha256'],
                       source_and_weight_end_verified=True, source_and_weight_checks=immutable,
                       counts=dict(counts), full_vocabulary_statuses=dict(vocabulary), fresh_npy_files=len(npy_cache),
                       selected_existing_full_vocabulary_semantic_proofs=prior,
                       full_vocabulary_scope='All actual dump SHA/header bytes fresh verified; only selected existing dumps independently CPU semantic-reconstructed previously. No new reconstruction claim for other arrays.',
                       teacher_invariance_scope='Observed teacher target/topk at identical preserved history only. Missing fields/skips explicit unknown; no full-vocab q equivalence claim.',
                       artifact_proofs=file_proof(artifact_path), process_terminal_observations=terminal,
                       no_signals_sent=True, no_cuda_or_http=True, started_unix=started, ended_unix=time.time(), diagnostic_strata_not_benchmark=True)
        write_exclusive(output / 'release.json', release); stage('ACCEPTED', completed_cases=192)
        return release
    except Exception as error:
        rejected = dict(schema='m1-complete-independent-cpu-release-v1', accepted=False, m1_run_id=config.get('m1_run_id'),
                        error_type=type(error).__name__, reason=str(error), no_signals_sent=True, no_cuda_or_http=True,
                        started_unix=started, ended_unix=time.time())
        write_exclusive(output / 'rejected.json', rejected); stage('REJECTED', error_type=type(error).__name__)
        raise


def write_exclusive(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); config, proof = json_file(args.config)
    try:
        result = execute(config, args.output)
    except Exception as error:
        print(json.dumps(dict(accepted=False, error_type=type(error).__name__, reason=str(error), output=str(Path(args.output).resolve())))); return 1
    print(json.dumps(dict(accepted=True, completed_cases=result['completed_cases'], counts=result['counts'],
                         elapsed_seconds=result['ended_unix']-result['started_unix'], release=str(Path(args.output).resolve()/'release.json')))); return 0


if __name__ == '__main__':
    sys.exit(main())
