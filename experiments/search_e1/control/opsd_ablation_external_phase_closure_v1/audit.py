#!/usr/bin/env python3
"""Fresh-read external teacher/sampling terminal receipts from accepted HKUST copies."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import types

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
ORDER = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
AGGREGATE_SHA = 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a'
DEBUG_COMPLETE_SHA = '17e16f318cbe2480173445c91ab925dc30b39b48217bbe1f16c56ea6bdba3d17'
SOURCE_OPERATION_SHA = '34857f4a6d0cfea4da2e2c4b71a5af662aca9a49e7ee7134ab97cd3c88185222'
LIMIT = 64 << 20


def require(value, reason):
    if not value:
        raise ValueError(reason)


def metadata(value):
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mode=value.st_mode, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def checked(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts
            and all(not p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink input required')
    before = metadata(path.stat())
    require(stat.S_ISREG(before['mode']) and before['bytes'] <= LIMIT, 'Small regular recorded input required')
    with path.open('rb') as handle:
        require(metadata(os.fstat(handle.fileno())) == before, 'Opened input identity changed')
        raw = handle.read()
        require(metadata(os.fstat(handle.fileno())) == before, 'Input changed during read')
    require(metadata(path.stat()) == before, 'Input path changed during read')
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'Pinned input/source SHA differs: ' + str(path))
    return raw, dict(path=str(path), sha256=digest, **before)


def bound_json(reference):
    raw, proof = checked(reference['path'], reference['sha256'])
    if 'bytes' in reference:
        require(proof['bytes'] == reference['bytes'], 'Pinned input bytes differ')
    return json.loads(raw), proof


def option(argv, name):
    require(isinstance(argv, list) and argv.count(name) == 1, 'Missing/duplicated recorded argument: ' + name)
    index = argv.index(name)
    require(index + 1 < len(argv), 'Recorded argument value absent')
    return argv[index + 1]


def load_aggregate(reference):
    require(reference['sha256'] == AGGREGATE_SHA, 'Frozen FixedBackups source must remain exact')
    raw, proof = checked(reference['path'], AGGREGATE_SHA)
    require(proof['bytes'] == reference['bytes'], 'Aggregate source byte count differs')
    module = types.ModuleType('_frozen_external_phase_backups')
    module.__file__ = reference['path']
    exec(compile(raw, reference['path'], 'exec'), module.__dict__)
    require('torch' not in sys.modules, 'External terminal audit must remain stdlib-only')
    return module, proof


def copy_chain(complete):
    batches, batches_ref = bound_json(complete['copy_batches'])
    require(batches['schema'] == 'opsd-ablation-remote-copy-batches-v1' and batches['accepted'] is True,
            'Original actually waited copy batches missing')
    require(len(batches['batches']) == len(complete['batch_proofs']) == 2, 'The two accepted original copy batches must remain bound')
    groups = {}
    refs = [batches_ref]
    for batch, recorded in zip(batches['batches'], complete['batch_proofs']):
        require(batch['complete'] == recorded['copies'] and batch['actual_wait'] == recorded['actual_wait'],
                'Copy batch/recorded proof reference differs')
        copied, copy_ref = bound_json(recorded['copies'])
        waited, wait_ref = bound_json(recorded['actual_wait'])
        require(copied['schema'] == 'opsd-ablation-remote-raw-copy-v1' and copied['accepted'] is True
                and copied['source_operation']['sha256'] == SOURCE_OPERATION_SHA
                and waited['actual_ssh_returncode'] == 0
                and waited['output'] == str(Path(copy_ref['path']).parent), 'Actual copy wait0/output/source binding differs')
        refs.extend([copy_ref, wait_ref])
        for variant, reference in copied['groups'].items():
            require(variant in ORDER and variant not in groups, 'Duplicate/unknown copied group')
            closed, closure_ref = bound_json(reference)
            require(closed['accepted'] is True and closed['variant'] == variant
                    and closed['independent_inode_copy'] is True and closed['node_originals_retained'] is True,
                    'Actual independent group copy closure absent')
            require(all(item['kind'] == 'direct' for item in closed['backups']), 'This audit requires actual independent direct copies')
            groups[variant] = dict(variant=variant, config=closed['config'], backups=closed['backups'])
            refs.append(closure_ref)
    require(set(groups) == set(ORDER), 'Seven actually copied groups required')
    return groups, refs


class Documents:
    def __init__(self, backups):
        self.backups = backups
        self.cache = {}

    def reference(self, path):
        require(path in self.backups.files, 'Recorded artifact absent from accepted independent inventory: ' + path)
        row = self.backups.files[path][0]
        return {key: row[key] for key in ('path', 'sha256', 'bytes')}

    def read(self, reference, category):
        actual = self.reference(reference['path'])
        require(all(reference[key] == actual[key] for key in ('path', 'sha256', 'bytes')), 'Parent artifact reference/inventory differs')
        if reference['path'] in self.cache:
            return self.cache[reference['path']]
        handle, source, local, path = self.backups.opened(reference)
        with handle:
            raw = handle.read()
            require(len(raw) <= LIMIT, 'Terminal/closure JSON unexpectedly large')
            self.backups.finish(reference, handle, source, local, path,
                                hashlib.sha256(raw).hexdigest(), len(raw), category)
        value = json.loads(raw)
        self.cache[reference['path']] = value
        return value

    def path(self, path, category):
        return self.read(self.reference(path), category)


def identity(value, run):
    require(value['uid'] == value['euid'] == 1005 and value['run_id'] == run,
            'Original real/effective UID or run differs')
    require(type(value['pid']) is int and value['pid'] > 0
            and type(value['starttick']) is int and value['starttick'] > 0,
            'Original PID/starttick absent')
    require(isinstance(value['argv'], list) and value['argv']
            and all(isinstance(arg, str) and arg for arg in value['argv']), 'Original argv absent')
    require(not value.get('identity_uncertain') and not value.get('identity_error'), 'Uncertain child identity cannot become accepted')


def terminal(documents, embedded, config, config_ref, expected_count, seen):
    require(embedded['accepted'] is True and len(embedded['child_exit_evidence']) == expected_count,
            'Parent embedded accepted phase/count absent')
    require(embedded['gpu_release']['all_owned_release']['accepted'] is True, 'Parent owned full-release proof absent')
    reference = embedded['actual_terminal']
    require(reference['path'] not in seen, 'External terminal reused across distinct phases')
    seen.add(reference['path'])
    actual = documents.read(reference, 'external_phase_terminal_fresh_sha')
    require(actual['schema'] == 'opsd-ablation-actual-phase-terminal-v1'
            and actual['run_id'] == config['run_id'] and actual['variant'] == config['variant']
            and actual['config_sha256'] == config_ref['sha256'] and actual['phase'] == 'sampling-running'
            and actual['child_exit_evidence'] == embedded['child_exit_evidence'],
            'Fresh external terminal schema/run/config/phase/evidence differs')
    require(type(actual['checked_unix']) in (int, float) and math.isfinite(actual['checked_unix'])
            and actual['checked_unix'] > 0, 'Original terminal timestamp absent')
    identities = set()
    for evidence in actual['child_exit_evidence']:
        require(type(evidence['actual_wait_returncode']) is int and evidence['actual_wait_returncode'] == 0,
                'Nonzero/unknown actual wait cannot be borrowed as successful completion')
        child = evidence['identity']
        identity(child, config['run_id'])
        marker = (child['pid'], child['starttick'])
        require(marker not in identities, 'Duplicated actual child within phase')
        identities.add(marker)
    return dict(reference=documents.reference(reference['path']), actual=actual,
                embedded_evidence_exactly_equal=True, independently_read=True)


def group_audit(helper, spec):
    config, config_ref = helper.stable_json(spec['config']['path'], spec['config']['sha256'])
    variant = spec['variant']; run = config['run_id']
    require(config['variant'] == variant and config['scientific']['outer_iterations'] == 10
            and config['owner']['uid'] == 1005, 'Original ten-outer group configuration differs')
    backups = helper.FixedBackups(spec['backups'], run)
    documents = Documents(backups)
    require(all(item['kind'] == 'direct' for item in spec['backups']), 'Only actual independent direct copies admitted')
    seen, rows = set(), []
    counts = dict(outers=10, teacher_phases=0, teacher_actual_children=0,
                  teacher_not_applicable=0, teacher_diagnostic_cases=0,
                  sampling_phases=0, sampling_actual_children=0)
    for index in range(10):
        outer = str(Path(config['paths']['output_root']) / 'outers' / ('outer-%05d' % index))
        teacher_root = outer + '/teacher-diagnostics/'
        parent = documents.path(teacher_root + 'complete.json', 'teacher_parent')
        require(parent['schema'] == 'opsd-ablation-teacher-diagnostic-parent-v1' and parent['accepted'] is True
                and parent['run_id'] == run and parent['variant'] == variant
                and parent['config_sha256'] == config_ref['sha256'] and parent['outer_index'] == index,
                'Teacher parent group/outer binding differs')
        require(parent['diagnostic_teacher_autoregressive'] is True and parent['original_training_objective'] is False
                and parent['diagnostic_precision_common_bf16'] is True
                and parent['training_or_validation_denominator_contribution'] == 0, 'Teacher diagnostic scope changed')
        selection = documents.read(parent['selection'], 'teacher_selection')
        require(parent['selection']['path'] == teacher_root + 'selection.json'
                and parent['count'] == selection['count'] == len(selection['selected'])
                and type(parent['count']) is int and 0 <= parent['count'] <= 8
                and selection['outer_index'] == index, 'Teacher selection/count/outer differs')
        teacher = None
        if selection['not_applicable']:
            require(parent['not_applicable'] is True and parent['count'] == 0
                    and parent['request'] is None and parent['child'] is None and parent['exit_proof'] is None,
                    'NA teacher invents a request/child/terminal')
            counts['teacher_not_applicable'] += 1
        else:
            require(parent['not_applicable'] is False and parent['count'] > 0,
                    'Applicable teacher has no selected case')
            teacher = terminal(documents, parent['exit_proof'], config, config_ref, 1, seen)
            child = documents.read(parent['child'], 'teacher_child_complete')
            request = documents.read(parent['request'], 'teacher_request')
            require(parent['child']['path'] == teacher_root + 'child-complete.json'
                    and parent['request']['path'] == teacher_root + 'request.json', 'Teacher child/request path differs')
            require(child['schema'] == 'opsd-ablation-teacher-diagnostic-child-v1' and child['accepted'] is True
                    and child['run_id'] == run and child['variant'] == variant and child['config_sha256'] == config_ref['sha256']
                    and child['count'] == parent['count'] == len(child['cases'])
                    and child['selection'] == parent['selection'] and child['pairs'] == parent['pairs']
                    and child['request'] == parent['request'], 'Teacher child input/process chain differs')
            require(request['run_id'] == run and request['variant'] == variant and request['outer_index'] == index
                    and request['config'] == spec['config'] and request['selection'] == parent['selection']
                    and request['pairs'] == parent['pairs'], 'Teacher request provenance differs')
            actual = teacher['actual']['child_exit_evidence'][0]['identity']
            require(child['pid'] == actual['pid'] and child['uid'] == actual['uid']
                    and child['run_id'] == actual['run_id'], 'Teacher completion differs from actual child')
            expected_argv = [config['paths']['python'], str(Path(config['paths']['helper_root']) / 'teacher_diag.py'),
                '--request', parent['request']['path'], '--request-sha256', parent['request']['sha256'],
                '--receipt', parent['child']['path']]
            require(actual['argv'] == expected_argv, 'Teacher actual argv differs from its recorded request/output')
            counts['teacher_phases'] += 1; counts['teacher_actual_children'] += 1
            counts['teacher_diagnostic_cases'] += parent['count']
        sampling = documents.path(outer + '/sampling-complete.json', 'sampling_parent')
        require(sampling['schema'] == 'opsd-online-sampling-complete-v1' and sampling['accepted'] is True
                and sampling['run_id'] == run and len(sampling['workers']) == 8, 'Sampling parent completion differs')
        sampled = terminal(documents, sampling['exit_proof'], config, config_ref, 8, seen)
        workers = []
        for rank, reference in enumerate(sampling['workers']):
            worker = documents.read(reference, 'sampling_worker_complete')
            actual = sampled['actual']['child_exit_evidence'][rank]['identity']
            require(worker['schema'] == 'opsd-online-sampling-worker-complete-v1' and worker['accepted'] is True
                    and worker['rank'] == rank and worker['rows'] == 320 and worker['run_id'] == run
                    and worker['variant'] == variant and worker['config_sha256'] == config_ref['sha256']
                    and worker['pid'] == actual['pid'] and worker['uid'] == actual['uid']
                    and worker['run_id'] == actual['run_id'], 'Sampling worker rank/run/config/process differs')
            attempt = str(Path(worker['rollouts']['path']).parent)
            require(attempt.startswith(outer + '/sampling-attempt-') and '/' not in attempt[len(outer) + 1:]
                    and reference['path'] == worker['rollouts']['path'] + '.worker-complete.json'
                    and worker['rollouts']['path'] == attempt + '/rollouts-%d.jsonl' % rank
                    and worker['generation_trace']['path'] == attempt + '/generation-trace-%d.jsonl' % rank
                    and worker['http_trace']['path'] == attempt + '/http-requests-%d.jsonl' % rank,
                    'Sampling actual output/attempt/rank paths differ')
            model_ref = documents.reference(attempt + '/sampling-model.json')
            model = documents.read(model_ref, 'sampling_model_receipt')
            require(model == sampling['sampling_model_identity'] and worker['sampling_model_identity'] == model['sha256'],
                    'Sampling model receipt/worker/parent identity differs')
            expected_argv = [config['paths']['python'], str(Path(config['paths']['helper_root']) / 'sampling.py'),
                '--config', spec['config']['path'], '--config-sha256', config_ref['sha256'],
                '--questions', attempt + '/questions-%d.jsonl' % rank,
                '--model-identity', model_ref['path'], '--model-identity-sha256', model_ref['sha256'],
                '--rank', str(rank), '--output', worker['rollouts']['path'],
                '--trace', worker['generation_trace']['path'], '--http-trace', worker['http_trace']['path']]
            require(actual['argv'] == expected_argv, 'Sampling actual argv differs from rank/config/model/output chain')
            workers.append(dict(rank=rank, completion=documents.reference(reference['path']), identity=actual))
        counts['sampling_phases'] += 1; counts['sampling_actual_children'] += 8
        rows.append(dict(outer=index, teacher=teacher, teacher_not_applicable=selection['not_applicable'],
                         teacher_cases=parent['count'], sampling=sampled, workers=workers))
    require(counts['teacher_phases'] + counts['teacher_not_applicable'] == 10
            and counts['sampling_phases'] == 10 and counts['sampling_actual_children'] == 80,
            'Ten closed teacher/sampling opportunities absent')
    return counts, dict(config=config_ref, backups=backups.inputs, fresh_read_files=backups.reads, outers=rows)


def save(path, value):
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return checked(path)[1]


def main():
    parser = argparse.ArgumentParser()
    for name in ('debug-complete', 'debug-complete-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(os.getuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Original HKUST UID and CPU-only environment required')
    require(args.debug_complete_sha256 == DEBUG_COMPLETE_SHA, 'This supplement binds the actually accepted full-seven debug pass')
    output = Path(args.output)
    require(output.is_absolute() and '..' not in output.parts and not output.exists()
            and output.resolve().is_relative_to(BASE / 'analysis')
            and all(not path.is_symlink() for path in output.parents), 'New independent HKUST analysis output required')
    complete, complete_ref = bound_json(dict(path=args.debug_complete, sha256=args.debug_complete_sha256))
    require(complete['schema'] == 'opsd-ablation-remote-recorded-readers-v1' and complete['accepted'] is True
            and complete['groups'] == list(ORDER) and complete['processing_host'] == 'HKUST1'
            and complete['Mac_raw_payloads'] is False, 'Full-seven accepted remote debug completion required')
    helper, source_ref = load_aggregate(complete['source_refs']['clipmass'])
    groups, copy_refs = copy_chain(complete)
    executions = {}
    for role in ('facts', 'clipmass', 'coverage'):
        execution, proof = bound_json(complete['executions'][role])
        require(execution['actual_wait_returncode'] == 0 and execution['processing_host'] == 'HKUST1'
                and execution['Mac_raw_payloads'] is False, 'Original debug child actual wait0 absent')
        require(execution['argv'][2] == complete['source_refs'][role]['path'], 'Recorded debug child executable differs')
        executions[role] = dict(value=execution, reference=proof)
    clip_exec = executions['clipmass']['value']['argv']
    inputs, inputs_ref = bound_json(dict(path=option(clip_exec, '--manifest'), sha256=option(clip_exec, '--manifest-sha256')))
    summary, summary_ref = bound_json(complete['clipmass_summary'])
    coverage, coverage_ref = bound_json(complete['debug_summary'])
    require(inputs['schema'] == 'seven-group-closed-clipmass-inputs-v1'
            and summary['accepted'] is True and coverage['accepted'] is True
            and summary['coverage']['all70_token_records_complete'] is True
            and summary['coverage']['completed_outers'] == 70
            and coverage['coverage']['completed_outers'] == 70
            and summary['input_digests']['manifest'] == inputs_ref['sha256'], 'All70 actual input/coverage chain differs')
    require(inputs['groups'] == [groups[variant] for variant in ORDER], 'Input groups differ from actually waited independent copies')
    started = time.time(); public_groups = {}; private_groups = {}
    for spec in inputs['groups']:
        counts, details = group_audit(helper, spec)
        public_groups[spec['variant']] = counts; private_groups[spec['variant']] = details
    totals = {key: sum(item[key] for item in public_groups.values()) for key in public_groups['C0']}
    require(totals['outers'] == 70 and totals['sampling_phases'] == 70 and totals['sampling_actual_children'] == 560,
            'Requested seventy original sampling phases not proven')
    # The bound chain is read again after selected member reads; the payload copies are
    # separately checked by FixedBackups at every actual selected read.
    for reference in [complete_ref, source_ref, inputs_ref, summary_ref, coverage_ref, *copy_refs,
                      *[entry['reference'] for entry in executions.values()]]:
        checked(reference['path'], reference['sha256'])
    require('torch' not in sys.modules, 'Audit imported Torch unexpectedly')
    own_ref = checked(Path(__file__).resolve())[1]
    output.mkdir(parents=True, exist_ok=False)
    private_ref = save(output / 'external-phase-details.private.json', dict(accepted=True,
        source=own_ref, debug_complete=complete_ref, aggregate_source=source_ref, clipmass_inputs=inputs_ref,
        clipmass_summary=summary_ref, debug_summary=coverage_ref, actual_copy_chain=copy_refs,
        original_reader_executions=executions, groups=private_groups, CLI_argv=sys.argv))
    public_ref = save(output / 'external-phase-publication.json', dict(
        schema='opsd-ablation-external-phase-closure-v1', accepted=True, groups=public_groups, totals=totals,
        external_terminal_files_independently_read=True, external_terminal_and_embedded_child_evidence_exactly_equal=True,
        all_actual_wait_returncodes_zero=True, child_UID_run_PID_starttick_argv_verified=True,
        original_owned_full_release_evidence_bound=True, no_live_process_or_GPU_recheck_claim=True,
        not_applicable_diagnostics_do_not_invent_children=True, processing_host='HKUST1', Mac_raw_payloads=False,
        started_unix=started, ended_unix=time.time(),
        digests=dict(source_sha256=own_ref['sha256'], debug_complete_sha256=complete_ref['sha256'],
                     aggregate_source_sha256=source_ref['sha256'], clipmass_inputs_sha256=inputs_ref['sha256'],
                     private_details_sha256=private_ref['sha256']),
        limitations=['Historical persisted phase exits and accepted independent copy provenance only; not new live GPU admission.',
                     'No scoring, generation, complete student trace parsing, token/logit or optimizer reconstruction.',
                     'Teacher autonomous diagnostics retain zero contribution to train/validation denominators.']))
    final_ref = save(output / 'complete.json', dict(schema='opsd-ablation-external-phase-closure-complete-v1',
        accepted=True, publication=public_ref, private_details=private_ref, totals=totals,
        source=own_ref, debug_complete=complete_ref, processing_host='HKUST1', Mac_raw_payloads=False))
    print(json.dumps(dict(accepted=True, complete=final_ref, totals=totals,
                          external_terminal_files_independently_read=True)))


if __name__ == '__main__':
    main()
