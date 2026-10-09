#!/usr/bin/env python3
"""Read-only local coverage of already accepted, committed debug artifacts."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import types

AGGREGATE_SHA256 = 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a'
DISTRIBUTIONS = ('student', 'teacher', 'outer_old', 'mixture')
NPY_DISTRIBUTIONS = ('student_logp', 'teacher_logp', 'outer_old_logp', 'target_logp')
CAP = 24 * (1 << 30)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metadata(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mode=s.st_mode,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def source_bytes(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and
            all(not p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink source required')
    before = metadata(path)
    with path.open('rb') as handle:
        raw = handle.read()
        require(metadata(path) == before, 'Source changed while read')
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'Source SHA mismatch')
    return raw, dict(path=str(path), sha256=digest, **before)


def load_aggregate(reference):
    require(reference['sha256'] == AGGREGATE_SHA256, 'Published aggregate source must remain exact')
    raw, proof = source_bytes(reference['path'], reference['sha256'])
    module = types.ModuleType('_sha_bound_recorded_clipmass')
    module.__file__ = reference['path']
    exec(compile(raw, reference['path'], 'exec'), module.__dict__)
    require('torch' not in sys.modules, 'Coverage must remain stdlib CPU-only')
    return module, proof


def reference(backups, path):
    require(path in backups.files, 'Expected artifact absent from accepted inventory: ' + path)
    row = backups.files[path][0]
    return {key: row[key] for key in ('path', 'sha256', 'bytes')}


def bound_reference(backups, ref):
    actual = reference(backups, ref['path'])
    require(all(ref[k] == actual[k] for k in ('path', 'sha256', 'bytes')), 'Artifact reference/inventory differs')
    return actual


def read_json(helper, backups, ref, category):
    handle, row, local, path = backups.opened(ref)
    with handle:
        raw = handle.read()
        backups.finish(ref, handle, row, local, path, hashlib.sha256(raw).hexdigest(), len(raw), category)
    return json.loads(raw)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def parse_http(backups, ref):
    handle, row, local, path = backups.opened(ref)
    digest = hashlib.sha256(); size = count = 0
    with handle:
        for line in handle:
            digest.update(line); size += len(line)
            require(line.endswith(b'\n'), 'Partial teacher HTTP record')
            value = json.loads(line)
            require(value['schema'] == 'opsd-online-private-http-request-v1' and value['rank'] == 0
                    and value['request_id'] == count and value['sampling_model_identity'] == 'source100-teacher-diagnostic',
                    'Teacher HTTP identity/order changed')
            require('request_raised' not in value and 'response_json_parse_error' not in value
                    and isinstance(value['response_json'], dict) and isinstance(value['response_body_text'], str),
                    'Teacher complete HTTP response absent')
            require(finite(value['started_at_unix']) and finite(value['ended_at_unix']) and
                    value['ended_at_unix'] >= value['started_at_unix'], 'Teacher HTTP time invalid')
            count += 1
        backups.finish(ref, handle, row, local, path, digest.hexdigest(), size, 'teacher_http')
    return count


def token_fields(backups, ref, run, expected_outer):
    handle, row, local, path = backups.opened(ref)
    digest = hashlib.sha256(); size = valid = top = scalar_only = 0; dumps = []
    with handle:
        line = handle.readline(); digest.update(line); size += len(line)
        require(line.endswith(b'\n'), 'Partial token header')
        header = json.loads(line)
        require(header['schema'] == 'opsd-online-training-token-header-v1' and header['run_id'] == run
                and header['outer_index'] == expected_outer and header['topk_cap'] == 8192
                and header['full_vocab_study_cap_bytes'] == CAP and header['loss_scope'] == 'actual_training_forward_before_optimizer',
                'Token header identity/cap/scope differs')
        mask = header['policy_mask']; ids = header['token_ids']
        require(len(mask) == len(ids) == len(header['token_offsets']) and all(v in (0, 1) for v in mask), 'Invalid header mask')
        positions = [i for i, enabled in enumerate(mask) if enabled]
        require(positions, 'Committed empty trajectory')
        for line in handle:
            digest.update(line); size += len(line)
            require(line.endswith(b'\n'), 'Partial token row')
            value = json.loads(line)
            if 'full_vocab_dump' in value:
                require(value['full_vocab_dump'] in ('complete', 'cap_skip') and value['token_index'] in positions
                        and value['identity']['run_id'] == run and value['identity']['outer_index'] == expected_outer,
                        'Invalid full-vocabulary request identity')
                if value['full_vocab_dump'] == 'complete':
                    require(len(value['files']) == 4 and {p['distribution'] for p in value['files']} == set(NPY_DISTRIBUTIONS)
                            and value['probabilities_persisted'] is False and value['probabilities_reconstructible_by_exp'] is True
                            and type(value['study_used_bytes']) is int and 0 <= value['study_used_bytes'] <= CAP,
                            'Invalid complete full-vocabulary record')
                    for fp in value['files']:
                        require(fp['dtype'] == 'float32' and fp['shape'] == [151936], 'Full-vocabulary declared shape/dtype differs')
                        bound_reference(backups, fp)
                else:
                    require(value['study_cap_bytes'] == CAP and 'files' not in value, 'Invalid explicit cap-skip record')
                dumps.append(dict(outer=expected_outer, pair_index=header['pair_index'], optimizer_update=header['optimizer_update'],
                                  token_index=value['token_index'], record=value))
                continue
            require(valid < len(positions) and value['token_index'] == positions[valid]
                    and value['token_id'] == ids[positions[valid]] and value['policy_mask'] is True
                    and value['teacher_forced_fixed_student_prefix'] is True and value['actual_generation_scores'] is False,
                    'Token position/origin mismatch')
            vocabulary = value['vocabulary_size']; require(vocabulary == 151936, 'Vocabulary differs')
            flags = [name + suffix in value for name in DISTRIBUTIONS for suffix in ('_top32_ids', '_top32_logp')]
            if any(flags):
                require(all(flags), 'Partial top32 distributions')
                for name in DISTRIBUTIONS:
                    tid = value[name + '_top32_ids']; logp = value[name + '_top32_logp']
                    require(len(tid) == len(logp) == 32 and len(set(tid)) == 32
                            and all(type(v) is int and 0 <= v < vocabulary for v in tid)
                            and all(finite(v) for v in logp), 'Top32 shape/finite/ID range invalid')
                top += 1
            else:
                scalar_only += 1
            valid += 1
        require(valid == len(positions), 'Token scalar coverage incomplete')
        backups.finish(ref, handle, row, local, path, digest.hexdigest(), size, 'committed_token_debug_fields')
    qid = header['identity']['question_id']
    require(isinstance(qid, str), 'Header question identity absent')
    return dict(header=header, valid_rows=valid, top32_positions=top, scalar_only=scalar_only,
                preselection_sha256=hashlib.sha256(('vocab-v1:42:' + qid).encode()).hexdigest(),
                first_valid_positions=positions[:2], dumps=dumps)


def npy_vector(backups, ref):
    handle, row, local, path = backups.opened(ref)
    digest = hashlib.sha256(); size = 0
    def read(count):
        nonlocal size
        value = handle.read(count); digest.update(value); size += len(value)
        require(len(value) == count, 'Truncated NPY vector')
        return value
    with handle:
        require(read(6) == b'\x93NUMPY', 'NPY magic differs')
        major, minor = read(2)
        require(major in (1, 2, 3), 'Unsupported NPY header version')
        length = struct.unpack('<H' if major == 1 else '<I', read(2 if major == 1 else 4))[0]
        require(length <= 65536, 'Unexpected NPY header length')
        header = ast.literal_eval(read(length).decode('utf8' if major == 3 else 'latin1').strip())
        require(header['descr'] == '<f4' and header['shape'] == (151936,) and header['fortran_order'] is False,
                'Actual NPY FP32 shape/order differs')
        header_bytes = size; remaining = 151936 * 4; count = 0
        while remaining:
            raw = read(min(65536, remaining)); remaining -= len(raw)
            for (value,) in struct.iter_unpack('<f', raw):
                require(math.isfinite(value), 'Nonfinite full-vocabulary payload')
                count += 1
        require(handle.read(1) == b'' and count == 151936 and size == header_bytes + 151936 * 4, 'NPY trailing bytes/size differs')
        backups.finish(ref, handle, row, local, path, digest.hexdigest(), size, 'full_vocabulary_fresh_sha_header_payload_finite')
    return dict(ref=ref, shape=[151936], dtype='<f4', values_checked=count, header_bytes=header_bytes,
                finite=True, fresh_sha256=True, normalized_or_KL_recomputed=False)


def embedded_exit(value, run, config_sha, expected_count, uid):
    require(value['accepted'] is True and len(value['child_exit_evidence']) == expected_count, 'Missing embedded accepted exits')
    for child in value['child_exit_evidence']:
        identity = child['identity']
        require(child['actual_wait_returncode'] == 0 and identity['run_id'] == run and identity['uid'] == uid
                and identity.get('starttick') is not None, 'Embedded child exit identity/rc differs')
    require(value['gpu_release']['all_owned_release']['accepted'] is True, 'Embedded release missing')
    require(value['actual_terminal']['path'] and value['actual_terminal']['sha256'], 'External terminal reference absent')
    return dict(ref=value['actual_terminal'], independently_read=False,
                scope='Outside rawouter archives; only fresh-SHA parent embedded rc0/release is inspected')


def aggregate_group(helper, spec, facts, old_proof, old_summary):
    variant = spec['variant']; config, config_ref = helper.stable_json(spec['config']['path'], spec['config']['sha256'])
    run = config['run_id']; require(config['variant'] == variant and config['scientific']['outer_iterations'] == 10, 'Wrong variant/config')
    require(old_proof['config']['sha256'] == config_ref['sha256'] and old_summary['run_id'] == run
            and old_summary['status'] == 'COMPLETE', 'Prior aggregate group binding differs')
    outerfacts = sorted([o for o in facts['outers'] if o['variant'] == variant], key=lambda o: o['outer'])
    require(len(outerfacts) == 10 and [o['outer'] for o in outerfacts] == list(range(10))
            and all(o['status'] == 'COMPLETE_FACTS' and o['run_id'] == run for o in outerfacts), 'Ten committed facts required')
    backups = helper.FixedBackups(spec['backups'], run)
    prior = old_proof['files']; require(len({x['source_path'] for x in prior}) == len(prior), 'Duplicate prior committed reference')
    require(set(x['category'] for x in prior) == {'committed_update_receipt', 'committed_token_log'}, 'Unexpected prior category')
    token_refs = {}; requests = {}; documents = {}; http = {}; tokens = {}; receipts = {}
    def request(path, category):
        ref = reference(backups, path); require(path not in requests, 'Duplicate coverage request'); requests[path] = (ref, category)
    native_refs = {u['source']['path']: u['source'] for outer in outerfacts for u in outer['updates']}
    for item in prior:
        ref = bound_reference(backups, dict(path=item['source_path'], sha256=item['sha256'], bytes=item['bytes']))
        if item['category'] == 'committed_update_receipt':
            require(ref['path'] in native_refs and all(ref[k] == native_refs[ref['path']][k] for k in ('sha256', 'bytes')), 'Committed receipt not in native facts')
            request(ref['path'], 'committed_receipt')
        else:
            token_refs[ref['path']] = ref; request(ref['path'], 'committed_token_fields')
    require(sum(x['category'] == 'committed_update_receipt' for x in prior) == len(native_refs)
            and len(token_refs) == old_summary['pair_exposures'], 'Prior committed counts differ')
    roots = {}
    for outer in outerfacts:
        index = outer['outer']; root = str(Path(config['paths']['output_root']) / 'outers' / ('outer-%05d' % index)); roots[index] = root
        teacher = root + '/teacher-diagnostics/'
        for path in backups.files:
            if path.startswith(teacher) and (path.endswith('.json') or path.endswith('-http.jsonl')):
                request(path, 'teacher_http' if path.endswith('-http.jsonl') else 'teacher_json')
        request(root + '/sampling-complete.json', 'sampling_parent')
        for worker in range(8):
            candidates = [p for p in backups.files if p.startswith(root + '/sampling-attempt-')
                          and p.endswith('/rollouts-%d.jsonl.worker-complete.json' % worker)]
            # Closed backups may retain failed sampling attempts. Only parent references below are accepted.
            for path in candidates:
                request(path, 'sampling_worker')
    def callback(ref):
        path = ref['path']; category = requests[path][1]
        if category == 'committed_token_fields':
            indices = [index for index, root in roots.items() if path.startswith(root + '/training/')]
            require(len(indices) == 1, 'Token path outside exact outer'); tokens[path] = token_fields(backups, ref, run, indices[0])
        elif category == 'teacher_http':
            http[path] = parse_http(backups, ref)
        else:
            value = read_json(helper, backups, ref, category)
            if category == 'committed_receipt': receipts[path] = value
            else: documents[path] = value
    backups.visit([value[0] for value in requests.values()], callback)
    expected_tokens = {}; pair_metrics = {}
    for path, value in receipts.items():
        require(value['accepted'] is True and value['run_id'] == run, 'Actual committed receipt differs')
        require(sorted(x['rank'] for x in value['all_rank_actual_token_metrics']) == list(range(8)), 'Missing committed rank')
        for rank in value['all_rank_actual_token_metrics']:
            for pair in rank['pairs']:
                ref = pair['token_log']; require(ref['path'] not in expected_tokens, 'Repeated committed token ref')
                require(bound_reference(backups, ref) == token_refs.get(ref['path']), 'Committed/prior token reference differs')
                expected_tokens[ref['path']] = (value, rank['rank'], pair)
    require(set(expected_tokens) == set(tokens), 'Committed token cover differs')
    top = {}; teacher_rows = []; dump_records = []; header_preselection = []; student_rows = []
    for outer in outerfacts:
        index = outer['outer']; current = [v for p, v in tokens.items() if v['header']['outer_index'] == index]
        require(len(current) == outer['counts']['K'] and {v['header']['pair_index'] for v in current} == set(range(outer['counts']['K'])), 'Outer pair coverage differs')
        for path, value in tokens.items():
            if value['header']['outer_index'] != index: continue
            receipt, rank, pair = expected_tokens[path]; header = value['header']
            require(header['optimizer_update'] == receipt['optimizer_updates'] and header['outer_index'] == receipt['outer_index']
                    and header['rank'] == rank and header['pair_index'] == pair['pair_index']
                    and value['valid_rows'] == pair['valid_tokens'], 'Committed/header rank/denominator differs')
        valid = sum(v['valid_rows'] for v in current); saved = sum(v['top32_positions'] for v in current)
        require(valid == outer['counts']['valid_tokens'] and saved == min(8192, valid), 'Actual top32 cap/coverage differs')
        top[index] = dict(valid_tokens=valid, committed_pairs=len(current), saved_top32_positions=saved,
                          scalar_only_positions=sum(v['scalar_only'] for v in current), top32_cap=8192,
                          all_four_vectors_shape32_finite_ID_range=True,
                          no_explicit_top32_cap_skip_field=True,
                          selection_order_reexecuted=False)
        dump_records.extend(record for value in current for record in value['dumps'])
        if index == 0:
            header_preselection = current
        root = roots[index]; tr = root + '/teacher-diagnostics/'
        selection = documents[tr + 'selection.json']; parent = documents[tr + 'complete.json']
        require(parent['accepted'] is True and parent['run_id'] == run and parent['variant'] == variant
                and parent['outer_index'] == index and parent['config_sha256'] == config_ref['sha256'], 'Teacher parent binding differs')
        require(parent['selection']['sha256'] == reference(backups, tr + 'selection.json')['sha256']
                and parent['count'] == selection['count'] == len(selection['selected']) <= 8
                and selection['seed'] == 42 and selection['max_cases'] == 8, 'Teacher selection/count differs')
        for value in (selection, parent):
            require(value['diagnostic_teacher_autoregressive'] is True and value['original_training_objective'] is False
                    and value['diagnostic_precision_common_bf16'] is True, 'Teacher scope flags differ')
        require(parent['training_or_validation_denominator_contribution'] == 0, 'Teacher entered scientific denominator')
        count = parent['count']; case_rows = []
        if selection['not_applicable']:
            require(count == 0 and parent['child'] is None and parent['exit_proof'] is None, 'NA invents teacher execution')
            terminal = None
        else:
            child_path = tr + 'child-complete.json'; child = documents[child_path]; request_doc = documents[tr + 'request.json']
            require(bound_reference(backups, parent['child']) == reference(backups, child_path)
                    and bound_reference(backups, parent['request']) == reference(backups, tr + 'request.json'), 'Teacher child/request reference differs')
            require(child['accepted'] is True and child['run_id'] == run and child['variant'] == variant
                    and child['config_sha256'] == config_ref['sha256'] and child['count'] == count
                    and child['selection'] == parent['selection'] and child['pairs'] == parent['pairs']
                    and child['request'] == parent['request'] and request_doc['outer_index'] == index,
                    'Teacher child input chain differs')
            require(child['source_identity'] == request_doc['source_identity']
                    and child['source_identity']['kind'] == 'raw-source100'
                    and child['source_identity']['source_model_identity'] == config['source_model_manifest']['sha256'], 'Teacher source changed')
            require(child['diagnostic_teacher_autoregressive'] is True and child['original_training_objective'] is False
                    and child['diagnostic_precision_common_bf16'] is True
                    and child['training_or_validation_denominator_contribution'] == 0
                    and child['rng_isolation'].startswith('Independent child'), 'Teacher RNG/scope differs')
            terminal = embedded_exit(parent['exit_proof'], run, config_ref['sha256'], 1, config['owner']['uid'])
            actual = parent['exit_proof']['child_exit_evidence'][0]['identity']
            require(child['pid'] == actual['pid'] and child['uid'] == actual['uid'] and len(child['cases']) == count, 'Teacher actual child differs')
            for ref, selected in zip(child['cases'], selection['selected']):
                bound_reference(backups, ref); case = documents[ref['path']]; trace = case['trace']
                require(case['selected'] == selected and case['run_id'] == run and case['variant'] == variant
                        and case['outer_index'] == index and case['original_pairs'] == child['pairs']
                        and case['source_model_identity'] == config['source_model_manifest']['sha256'], 'Teacher case input differs')
                require(case['diagnostic_teacher_autoregressive'] is True and case['original_training_objective'] is False
                        and case['diagnostic_precision_common_bf16'] is True and trace['teacher_forced'] is False
                        and trace['generation_options_modified'] is False and trace['return_objects_modified'] is False
                        and not trace['logging_errors'], 'Teacher actual transparent trace incomplete')
                href = bound_reference(backups, case['http']); require(href['path'] in http, 'Teacher HTTP not inspected')
                require(http[href['path']] == case['result']['search_calls'], 'Teacher HTTP/search count differs')
                events = [e for e in trace['events'] if e['kind'] == 'native_generate']
                require([e['call'] for e in events] == list(range(len(events)))
                        and all('raw_output_ids' in e and 'args' in e and 'kwargs' in e for e in events), 'Teacher generation IDs/options absent')
                case_rows.append(dict(reference=ref, stratum=selected['stratum'], actual_generate_calls=len(events), http_requests=http[href['path']], flags_verified=True))
        teacher_rows.append(dict(outer=index, cases=count, selected_strata=selection['selected_strata'],
                                 not_applicable=selection['not_applicable'], terminal=terminal, cases_private=case_rows))
        sampling = documents[root + '/sampling-complete.json']
        require(sampling['accepted'] is True and sampling['run_id'] == run and len(sampling['workers']) == 8, 'Student sampling parent incomplete')
        embedded_exit(sampling['exit_proof'], run, config_ref['sha256'], 8, config['owner']['uid'])
        workers = []
        for rank, ref in enumerate(sampling['workers']):
            bound_reference(backups, ref); worker = documents[ref['path']]
            require(worker['accepted'] is True and worker['rank'] == rank and worker['run_id'] == run
                    and worker['variant'] == variant and worker['rows'] == 320
                    and worker['config_sha256'] == config_ref['sha256'], 'Student worker receipt differs')
            for key in ('rollouts', 'generation_trace', 'http_trace'): bound_reference(backups, worker[key])
            workers.append(dict(rank=rank, native_rows=320, generation_trace=worker['generation_trace'], http_trace=worker['http_trace'],
                                HTTP_requests=worker['http_requests'], original_full_member_SHA_reused=True,
                                trace_contents_not_rescanned=True))
        student_rows.append(dict(outer=index, workers_private=workers, native_rows=2560, preserved_generation_and_HTTP_files=16))
    selected = min(header_preselection, key=lambda h: h['preselection_sha256']) if header_preselection else None
    expected_positions = selected['first_valid_positions'] if selected else []
    require(len({(d['outer'], d['pair_index'], d['token_index']) for d in dump_records}) == len(dump_records), 'Duplicate full-vocabulary request')
    require(all(d['outer'] == 0 and d['pair_index'] == selected['header']['pair_index'] for d in dump_records), 'Dump not deterministic eligible preselection')
    require(sorted(d['token_index'] for d in dump_records) == expected_positions, 'Dump requests differ from first two valid positions')
    npy_refs = []
    for dump in dump_records:
        if dump['record']['full_vocab_dump'] == 'complete': npy_refs.extend(dump['record']['files'])
    require(len({p['path'] for p in npy_refs}) == len(npy_refs), 'Repeated full-vocabulary file')
    npy_results = []
    backups.visit(npy_refs, lambda ref: npy_results.append(npy_vector(backups, ref)))
    saved_bytes = sum(p['bytes'] for p in npy_refs); require(saved_bytes <= CAP, 'Current committed subset exceeds study cap')
    all_npy = [row for row, _ in backups.files.values() if row['path'].endswith('.npy')]
    inventory_bytes = sum(row['bytes'] for row in all_npy); require(inventory_bytes <= CAP, 'Current raw backup NPY subset exceeds cap')
    public = dict(status='COMPLETE', variant=variant, run_id=run,
        teacher_autoregressive=dict(total_cases=sum(x['cases'] for x in teacher_rows), max_per_outer=8,
            per_outer=[{k: x[k] for k in ('outer', 'cases', 'selected_strata', 'not_applicable')} for x in teacher_rows],
            actual_parent_embedded_rc0_release_verified=True, external_terminal_files_independently_read=False,
            isolated_child_source100_common_bf16=True, training_validation_denominator=0),
        student_trace_persistence=dict(outers=10, native_rows=25600, original_member_SHA_binding=True,
            generation_trace_files=80, HTTP_trace_files=80, trace_contents_rescanned=False),
        top32=dict(outers=[dict(outer=index, **row) for index, row in sorted(top.items())],
            valid_tokens=sum(x['valid_tokens'] for x in top.values()), saved_positions=sum(x['saved_top32_positions'] for x in top.values()),
            scalar_only_positions=sum(x['scalar_only_positions'] for x in top.values()), distributions_per_saved_position=4,
            actual_arrays_shape_finite_ID_range_verified=True, numerical_rank_or_normalization_recomputed=False),
        full_vocabulary=dict(requested_positions=len(dump_records), saved_positions=sum(d['record']['full_vocab_dump'] == 'complete' for d in dump_records),
            explicit_cap_skip_positions=sum(d['record']['full_vocab_dump'] == 'cap_skip' for d in dump_records),
            saved_files=len(npy_refs), committed_NPY_bytes=saved_bytes, all_backed_up_subset_NPY_bytes=inventory_bytes,
            shape=[151936], dtype='float32', all_committed_payload_values_finite=True, fresh_NPY_member_SHA=True,
            deterministic_preselection_verified=True, selected_pair_index=selected['header']['pair_index'] if selected else None,
            selected_qid_selection_digest=selected['preselection_sha256'] if selected else None, first_valid_token_positions=expected_positions,
            cap_bytes=CAP, only_current_subset_below_cap=True, studywide_shared_ledger_verification='PENDING',
            KL_or_full_distribution_normalization_recomputed=False))
    details = dict(config=config_ref, backups=backups.inputs, fresh_selected_members=backups.reads,
                   teacher=teacher_rows, student=student_rows, full_vocabulary_records=dump_records, NPY_checks=npy_results,
                   unreferenced_inventory_NPY=[row for row in all_npy if row['path'] not in {p['path'] for p in npy_refs}])
    return public, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'manifest-sha256', 'output'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); out = Path(args.output)
    require(out.is_absolute() and '..' not in out.parts and not out.exists()
            and all(not p.is_symlink() for p in out.parents), 'New canonical output directory required')
    # Bootstrap JSON parsing before importing even the SHA-bound stdlib helper.
    raw, manifest_proof = source_bytes(args.manifest, args.manifest_sha256); manifest = json.loads(raw)
    require(manifest['schema'] == 'opsd-ablation-recorded-debug-coverage-inputs-v1', 'Explicit debug-coverage input required')
    helper, source_proof = load_aggregate(manifest['aggregate_source'])
    inputs, inputs_ref = helper.stable_json(**dict(path=manifest['clipmass_inputs']['path'], expected_sha=manifest['clipmass_inputs']['sha256']))
    prior, prior_ref = helper.stable_json(manifest['committed_proof']['path'], manifest['committed_proof']['sha256'])
    summary, summary_ref = helper.stable_json(manifest['clipmass_summary']['path'], manifest['clipmass_summary']['sha256'])
    derived, derived_ref = helper.stable_json(manifest['clipmass_derived_manifest']['path'], manifest['clipmass_derived_manifest']['sha256'])
    facts, facts_ref = helper.stable_json(inputs['facts']['path'], inputs['facts']['sha256'])
    require(inputs['schema'] == 'seven-group-closed-clipmass-inputs-v1' and prior['accepted'] is True
            and summary['accepted'] is True and derived['accepted'] is True and facts['accepted'] is True
            and facts['schema'] == 'opsd-ablation-outer-facts-v1', 'Existing accepted input/proof required')
    require(derived['source_code_sha256'] == AGGREGATE_SHA256
            and derived['artifacts']['clipmass-publication.json']['sha256'] == summary_ref['sha256']
            and derived['artifacts']['referenced-files-proof.private.json']['sha256'] == prior_ref['sha256']
            and prior['manifest']['sha256'] == inputs_ref['sha256'] and prior['facts']['sha256'] == facts_ref['sha256']
            and summary['input_digests'] == dict(manifest=inputs_ref['sha256'], facts=facts_ref['sha256']), 'Prior aggregate provenance chain differs')
    specs = inputs['groups']; variants = [s['variant'] for s in specs]
    require(specs and len(set(variants)) == len(variants) and set(variants) <= set(helper.VARIANTS)
            and set(variants) == set(prior['groups']) == set(summary['groups']), 'Explicit accepted subset differs')
    started = time.time(); public = {}; details = {}
    for spec in specs:
        variant = spec['variant']; public[variant], details[variant] = aggregate_group(helper, spec, facts, prior['groups'][variant], summary['groups'][variant])
    own_raw, own_proof = source_bytes(Path(__file__).resolve())
    result = dict(schema='opsd-ablation-recorded-debug-coverage-v1', accepted=True, started_unix=started, ended_unix=time.time(),
        groups=public, coverage=dict(completed_groups=len(specs), expected_groups=7, completed_outers=10 * len(specs), expected_outers=70,
            subset=variants, all70_debug_coverage_complete=False, shared_24GiB_ledger_complete=False),
        provenance=dict(source_sha256=own_proof['sha256'], aggregate_source_sha256=source_proof['sha256'], input_manifest_sha256=manifest_proof['sha256'],
            clipmass_inputs_sha256=inputs_ref['sha256'], prior_committed_proof_sha256=prior_ref['sha256'], clipmass_summary_sha256=summary_ref['sha256']),
        limitations=['Recorded committed artifacts only; no models, scoring, logits/KL/gradients or RNG replay.',
            'Teacher autonomous privileged history differs from fixed-D teacher-forced training q; no scientific denominator contribution.',
            'Student full trace/HTTP content preservation reuses accepted full member SHA; this pass does not reparse their raw content.',
            'Teacher external phase-exit files are outside these raw archives; embedded parent rc0/release only, not independent external terminal read.',
            'Top32 four arrays checked shape/finite/ID range, not full-vocabulary normalization or probability reconstruction.',
            'Full-vocabulary finite payload and fresh member SHA checked; subset bytes do not verify final all-seven shared ledger or its reservations.'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'debug-coverage-publication.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    private = dict(accepted=True, CLI_argv=sys.argv, inputs=dict(manifest=manifest_proof, aggregate_source=source_proof, clipmass_inputs=inputs_ref,
        prior_committed_proof=prior_ref, clipmass_summary=summary_ref, derived=derived_ref, facts=facts_ref), groups=details)
    (out / 'details.private.json').write_text(json.dumps(private, indent=2, allow_nan=False) + '\n')
    artifacts = {}
    for name in ('debug-coverage-publication.json', 'details.private.json'):
        raw = (out / name).read_bytes(); artifacts[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (out / 'derived-manifest.json').write_text(json.dumps(dict(accepted=True, source_sha256=own_proof['sha256'],
        input_sha256=manifest_proof['sha256'], artifacts=artifacts, coverage=result['coverage'], no_node_or_model_IO=True), indent=2) + '\n')
    print(json.dumps(dict(accepted=True, coverage=result['coverage'])))


if __name__ == '__main__': main()
