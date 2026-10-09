#!/usr/bin/env python3
"""Private local accepted-backup teacher wait evidence; never models or live gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

PARSER_SHA = 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a'
PRODUCER_SHA = 'c8aa9c8f81bb669707085d0a586843608058258a3882006b380a28c556f26b56'
VARIANTS = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_module(ref, expected=PARSER_SHA, execute=True):
    path = Path(ref['path'])
    require(path.is_absolute() and '..' not in path.parts and all(not p.is_symlink() for p in (path, *path.parents)), 'Unsafe source path')
    def identity(st):
        return (st.st_dev, st.st_ino, st.st_size, st.st_mode, st.st_mtime_ns, st.st_ctime_ns)
    before = identity(path.stat())
    with path.open('rb') as handle:
        require(identity(os.fstat(handle.fileno())) == before, 'Source opened identity changed')
        raw = handle.read()
        require(identity(os.fstat(handle.fileno())) == before, 'Source changed during read')
    require(identity(path.stat()) == before and hashlib.sha256(raw).hexdigest() == ref['sha256'] == expected, 'Pinned parser source SHA differs')
    if not execute:
        return None
    namespace = {'__name__': 'sha_bound_terminal_backup_reader', '__file__': str(path)}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


def bound(backups, ref):
    require(ref['path'] in backups.files, 'Reference absent from accepted backup: ' + ref['path'])
    row = backups.files[ref['path']][0]
    require(all(ref[k] == row[k] for k in ('path', 'sha256', 'bytes')), 'Reference SHA/bytes differ')
    recorded = row.get('pinned_metadata', row)
    for key in ('device', 'inode', 'mtime_ns', 'ctime_ns', 'mode'):
        if key in ref:
            require(ref[key] == recorded[key], 'Reference pinned metadata differs: ' + key)
    return row


def same_producer_fileproof(request_ref, facts_ref):
    """871 producer proof has seven fields; collector may additionally record mode."""
    keys = ('path', 'sha256', 'bytes', 'device', 'inode', 'mtime_ns', 'ctime_ns')
    require(set(request_ref) in (set(keys), set(keys) | {'mode'}), 'Unknown/missing producer config identity fields')
    require(all(request_ref[k] == facts_ref[k] for k in keys), 'Producer config identity differs')
    if 'mode' in request_ref:
        require(request_ref['mode'] == facts_ref['mode'], 'Producer config mode differs')
    return True


def verify_external(actual, exits, run, config_sha, variant, uid):
    """Necessary 871 Controller.verify_exit semantics, with additional schema/variant checks."""
    require(exits.get('accepted') is True and len(exits['child_exit_evidence']) == 1, 'Accepted single teacher exit missing')
    require(actual['schema'] == 'opsd-ablation-actual-phase-terminal-v1'
            and actual['run_id'] == run and actual['variant'] == variant
            and actual['config_sha256'] == config_sha and actual['phase'] == 'sampling-running'
            and actual['child_exit_evidence'] == exits['child_exit_evidence'], 'Actual external wait identity/evidence differs')
    for entry in actual['child_exit_evidence']:
        identity = entry['identity']
        require(type(entry['actual_wait_returncode']) is int and entry['actual_wait_returncode'] == 0
                and identity['uid'] == uid and identity['run_id'] == run
                and type(identity.get('pid')) is int and identity['pid'] > 0
                and type(identity.get('starttick')) is int and identity['starttick'] > 0
                and not identity.get('identity_uncertain'), 'Actual external child rc/UID/tick/run invalid')
    require(exits['gpu_release']['all_owned_release']['accepted'] is True, 'Parent embedded full release absent')
    return actual['child_exit_evidence'][0]['identity']


def group(helper, spec, facts):
    stable_json = helper['stable_json']
    config, config_ref = stable_json(spec['config']['path'], spec['config']['sha256'])
    variant = spec['variant']; run = config['run_id']; uid = config['owner']['uid']
    require(variant in VARIANTS and config['variant'] == variant and config['scientific']['outer_iterations'] == 10, 'Wrong group contract')
    outers = sorted((r for r in facts['outers'] if r['variant'] == variant), key=lambda r: r['outer'])
    require(len(outers) == 10 and [r['outer'] for r in outers] == list(range(10))
            and all(r['status'] == 'COMPLETE_FACTS' and r['run_id'] == run for r in outers), 'Exact ten accepted facts required')
    source_config = spec['source_config']
    require(source_config in facts['configs'] and source_config['sha256'] == config_ref['sha256']
            and source_config['bytes'] == config_ref['bytes'], 'Facts source config not same identity')
    backups = helper['FixedBackups'](spec['backups'], run)
    # Optional independently SHA-bound small CPU readbacks resolve absent C0 terminals.
    supplements = {}
    for item in spec.get('external_readbacks', []):
        source = item['source']; value, local_ref = stable_json(item['local']['path'], item['local']['sha256'])
        require(local_ref['sha256'] == source['sha256'] and local_ref['bytes'] == source['bytes'], 'External readback bytes differ')
        require(source['path'] not in supplements and source['path'] not in backups.files, 'Duplicate external readback')
        supplements[source['path']] = (source, value, local_ref)
    selected = {}; roots = {}
    for row in outers:
        index = row['outer']; root = str(Path(config['paths']['output_root']) / 'outers' / ('outer-%05d' % index) / 'teacher-diagnostics')
        roots[index] = root
        for name in ('complete.json', 'selection.json'):
            path = root + '/' + name
            require(path in backups.files, 'Required teacher parent/selection missing: ' + path)
            selected[path] = backups.files[path][0]
        for name in ('request.json', 'child-complete.json'):
            path = root + '/' + name
            if path in backups.files:
                selected[path] = backups.files[path][0]
    # One archive traversal reads only teacher small JSON and phase-exit JSON.
    # The compressed stream necessarily decompresses skipped bytes; no token/NPY parsing.
    phase_root = str(Path(config['paths']['output_root']) / 'phase-exits') + '/'
    for path, (ref, _) in backups.files.items():
        if path.startswith(phase_root) and path.endswith('.json'):
            selected[path] = ref
    documents = {}
    def visit(ref):
        documents[ref['path']] = backups.receipt(ref)
    backups.visit(list(selected.values()), visit)
    results = []; missing = []; consumed_supplements = set()
    for index in range(10):
        root = roots[index]; parent = documents[root + '/complete.json']; selection = documents[root + '/selection.json']
        require(parent['schema'] == 'opsd-ablation-teacher-diagnostic-parent-v1' and parent['accepted'] is True
                and parent['run_id'] == run and parent['variant'] == variant
                and parent['outer_index'] == index and parent['config_sha256'] == config_ref['sha256'], 'Teacher parent identity differs')
        bound(backups, parent['selection'])
        require(parent['selection']['path'] == root + '/selection.json'
                and parent['count'] == selection['count'] == len(selection['selected']) <= 8
                and selection['seed'] == 42 and selection['max_cases'] == 8, 'Teacher selection differs')
        for value in (parent, selection):
            require(value['diagnostic_teacher_autoregressive'] is True and value['original_training_objective'] is False
                    and value['diagnostic_precision_common_bf16'] is True, 'Teacher AR scope differs')
        require(parent['training_or_validation_denominator_contribution'] == 0, 'Teacher contributes to scientific denominator')
        base = dict(outer=index, teacher_cases=parent['count'], parent=selected[root + '/complete.json'])
        if selection['not_applicable']:
            require(parent['count'] == 0 and parent['not_applicable'] is True
                    and parent['request'] is None and parent['child'] is None and parent['exit_proof'] is None, 'NA invents teacher child')
            results.append(dict(base, status='NOT_APPLICABLE_NO_CHILD')); continue
        require(parent['not_applicable'] is False and parent['count'] > 0, 'Applicable teacher count invalid')
        for key, name in (('request', 'request.json'), ('child', 'child-complete.json')):
            bound(backups, parent[key]); require(parent[key]['path'] == root + '/' + name, 'Teacher child/request path differs')
        child = documents[parent['child']['path']]; request = documents[parent['request']['path']]
        require(child['schema'] == 'opsd-ablation-teacher-diagnostic-child-v1' and child['accepted'] is True
                and child['run_id'] == run and child['variant'] == variant and child['config_sha256'] == config_ref['sha256']
                and child['count'] == parent['count'] and child['selection'] == parent['selection']
                and child['pairs'] == parent['pairs'] and child['request'] == parent['request']
                and same_producer_fileproof(request['config'], source_config) and request['run_id'] == run
                and request['variant'] == variant and request['outer_index'] == index, 'Child/request chain differs')
        require(child['source_identity'] == request['source_identity']
                and child['source_identity']['kind'] == 'raw-source100'
                and child['source_identity']['source_model_identity'] == config['source_model_manifest']['sha256'], 'Teacher source differs')
        for key, expected in (('diagnostic_teacher_autoregressive', True), ('original_training_objective', False), ('diagnostic_precision_common_bf16', True), ('training_or_validation_denominator_contribution', 0)):
            require(child[key] == expected, 'Teacher child scope differs')
        exits = parent['exit_proof']; terminal = exits['actual_terminal']
        require(Path(terminal['path']).parent == Path(phase_root.rstrip('/')) and Path(terminal['path']).suffix == '.json', 'Teacher external terminal outside group phase-exits')
        # Validate embedded evidence even for missing external bytes, without labeling it independent.
        embedded_actual = dict(schema='opsd-ablation-actual-phase-terminal-v1', run_id=run, variant=variant,
                               config_sha256=config_ref['sha256'], phase='sampling-running', child_exit_evidence=exits['child_exit_evidence'])
        identity = verify_external(embedded_actual, exits, run, config_ref['sha256'], variant, uid)
        require(child['pid'] == identity['pid'] and child['uid'] == identity['uid'], 'Teacher child process differs')
        if terminal['path'] in documents:
            bound(backups, terminal); actual = documents[terminal['path']]; origin = 'accepted_backup_member'
        elif terminal['path'] in supplements:
            source, actual, local_ref = supplements[terminal['path']]
            require(all(source[k] == terminal[k] for k in ('path', 'sha256', 'bytes')), 'Supplement not exact original terminal')
            consumed_supplements.add(terminal['path']); origin = 'exact_SHA_small_readback'
        else:
            missing.append(dict(variant=variant, run_id=run, outer=index, source=terminal, reason='External bytes absent from accepted backups/readbacks'))
            results.append(dict(base, status='MISSING_EXTERNAL_WAIT_BYTES', external_terminal=terminal,
                                release_scope='parent embedded only', external_wait_independently_read=False)); continue
        verify_external(actual, exits, run, config_ref['sha256'], variant, uid)
        results.append(dict(base, status='EXTERNAL_WAIT_VERIFIED', external_terminal=terminal, origin=origin,
                            external_wait_independently_read=True, release_scope='parent embedded historical all_owned_release only; no separate external release/live rescan',
                            child_identity=identity))
    require(consumed_supplements == set(supplements), 'Unreferenced supplementary terminal supplied')
    complete = not missing
    return dict(variant=variant, run_id=run, config=config_ref, complete=complete,
                outer_slots=10, teacher_cases=sum(r['teacher_cases'] for r in results), results=results,
                missing_external_read_requests=missing, accepted_backup_inputs=backups.inputs,
                selected_small_member_reads=backups.reads, token_NPY_or_case_payloads_read=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True); parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output)
    require(output.is_absolute() and '..' not in output.parts and not output.exists()
            and all(not p.is_symlink() for p in output.parents), 'New canonical output directory required')
    output.mkdir(mode=0o700)
    try:
        preliminary = json.loads(Path(args.manifest).read_bytes())
        helper = source_module(preliminary['parser_source'])
        manifest, manifest_ref = helper['stable_json'](args.manifest, args.manifest_sha256)
        require(manifest['parser_source'] == preliminary['parser_source'], 'Manifest changed during parser binding')
        source_module(manifest['producer_source'], PRODUCER_SHA, execute=False)
        require(manifest['schema'] == 'opsd-ablation-teacher-terminal-audit-inputs-v1', 'Wrong input schema')
        facts, facts_ref = helper['stable_json'](manifest['facts']['path'], manifest['facts']['sha256'])
        require(facts['accepted'] is True, 'Facts not accepted')
        variants = [g['variant'] for g in manifest['groups']]
        require(variants and len(set(variants)) == len(variants) and set(variants) <= set(VARIANTS), 'Explicit unique group subset required')
        groups = [group(helper, spec, facts) for spec in manifest['groups']]
        result = dict(schema='opsd-ablation-teacher-terminal-audit-v2', checked_unix=time.time(),
                      complete=all(g['complete'] for g in groups), scope_groups=variants,
                      expected_outer_slots=10 * len(groups), manifest=manifest_ref, facts=facts_ref,
                      parser_source=manifest['parser_source'], producer_source=manifest['producer_source'], groups=groups,
                      missing_external_read_requests=[r for g in groups for r in g['missing_external_read_requests']],
                      limits=['No live process/release rescan; external wait bytes and historical parent embedded release are distinct.',
                              'No model/Torch/GPU/HTTP/scoring/token/NPY payload analysis. Gzip must decompress sequential skipped bytes.',
                              'Only explicit accepted ten-outer subset, not all70/5632 or final24GiB ledger acceptance.'])
        path = output / 'result.private.json'
        with path.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps(dict(complete=result['complete'], groups=variants, output=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    except Exception as exc:
        with (output / 'failed.private.json').open('x') as handle:
            json.dump(dict(error=repr(exc), checked_unix=time.time(), argv=vars(args)), handle); handle.flush(); os.fsync(handle.fileno())
        raise


if __name__ == '__main__':
    main()
