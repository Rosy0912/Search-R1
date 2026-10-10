"""CPU evidence adapter; reuse original endpoint acceptance and paired counts."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
try:
    from .common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, read, rows,
                         checked_file, same, exclusive, bind_helper)
except ImportError:
    from common import (VARIANTS, ORIGINAL_SHA, STUDY_ROOT, require, read, rows,
                        checked_file, same, exclusive, bind_helper)

COMPLETION_SHA = 'b36aff538310bbd92c947e790e32254ce3267e9f1f62531aa4729a74bc719163'
ANALYSIS_SHA = '649472b2312cd3c9225b2b5ea1e8c1ad8e70d20ca6591d51fdcbe0194dd3e4de'
COMMITS = dict(original='00fec71f6e2481f9e1bc4576d956748bdda77aaa',
               completion='7d9fea2333d85716cce1bd08c3d10ee0f3248e28',
               analysis='235e37654e57ca6e19ce2ef33846891951deef04')
SCHEMA = 'opsd-signal-single-endpoint-pair-request-v1'


def published(spec, expected_manifest=None, expected_commit=None):
    """Explicit SHA, current source bytes and historical publication; no Git call."""
    manifest, mp = read(spec['source_manifest']); publication, pp = read(spec['publication'])
    require(publication.get('accepted') is True and publication.get('branch') == 'search-e1'
            and publication.get('source_commit') == publication.get('remote_readback_sha')
            and isinstance(publication.get('source_commit'), str) and len(publication['source_commit']) == 40
            and all(char in '0123456789abcdef' for char in publication['source_commit'])
            and publication.get('source_manifest_sha256') == mp['sha256'], 'Helper publication/readback differs')
    require(expected_manifest is None or mp['sha256'] == expected_manifest, 'Wrong frozen helper manifest')
    require(expected_commit is None or publication['source_commit'] == expected_commit, 'Wrong frozen helper source commit')
    root = Path(spec['helper_root']).resolve()
    require({str(p.relative_to(root)) for p in root.rglob('*.py')} ==
            {name for name in manifest['files'] if name.endswith('.py')}, 'Helper Python inventory changed')
    for name, entry in manifest['files'].items():
        relative = Path(name); require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source manifest path')
        checked_file(dict(path=str(root / relative), sha256=entry['sha256'], bytes=entry['bytes']))
    return dict(helper_root=str(root), source_manifest=mp, publication=pp,
                source_commit=publication['source_commit'])


def bind_inputs(request):
    require(request['schema'] == SCHEMA, 'Wrong single-endpoint request')
    sources = {'reader': published(request['reader_source'])}
    require(Path(sources['reader']['helper_root']) / 'reader.py' == Path(__file__).resolve(),
            'Actual reader entry differs from its own published source')
    for name, digest in [('original', ORIGINAL_SHA), ('completion', COMPLETION_SHA), ('analysis', ANALYSIS_SHA)]:
        sources[name] = published(request[name + '_source'], digest, COMMITS[name])
    study, study_ref = read(request['study_manifest'])
    require(str(Path(study['study_base']).resolve()) == STUDY_ROOT
            and [g['variant'] for g in study['groups']] == list(VARIANTS)
            and same(study['helper_source_manifest'], sources['original']['source_manifest'])
            and same(study['publication'], sources['original']['publication'])
            and str(Path(study['helper_root']).resolve()) == sources['original']['helper_root'],
            'Original ordered study/helper/publication differs')
    require(set(request['configs']) == set(VARIANTS), 'Four exact original configs required')
    native = bind_helper(sources['original']['helper_root'], sources['original']['source_manifest'],
                         ('checkpoint', 'prepare', 'scoring', 'evaluation', 'training_metrics'))
    completion = bind_helper(sources['completion']['helper_root'], sources['completion']['source_manifest'], ('assemble',))['assemble']
    analysis = bind_helper(sources['analysis']['helper_root'], sources['analysis']['source_manifest'], ('summary', 'replay_math'))
    configs = {}; refs = {}; sources['config_publications'] = {}
    shared = ('source_model_manifest', 'runtime_manifest', 'common_initial_adapter_manifest',
              'common_initial_adapter_tensor_sha256', 'common_cohort_manifest', 'train_identity', 'validation_identity')
    for group in study['groups']:
        variant = group['variant']; cfg, cf = read(request['configs'][variant])
        require(same(cf, group['config']) and cfg['variant'] == variant and cfg['study_id'] == study['study_id']
                and cfg['source_commit'] == COMMITS['original']
                and same(cfg['helper_source_manifest'], sources['original']['source_manifest'])
                and str(Path(cfg['paths']['helper_root']).resolve()) == sources['original']['helper_root'],
                'Config quartet differs from immutable original study/source')
        cfg['_config_sha256'] = cf['sha256']; native['prepare'].validate_config(cfg, launch=True)
        # The study helper readback and the four run-scoped gate receipts have
        # distinct immutable files. Preserve the native scope/source admission
        # rather than requiring these five references to name one file.
        publication_ref = checked_file(cfg['gates']['publication'])
        admission = native['evaluation'].gates.publication_admission(
            cfg, native['evaluation'].gates.scoped(cfg, native['evaluation'].gates.bound(cfg['gates']['publication'])))
        require(checked_file(cfg['gates']['publication']) == publication_ref,
                'Config publication changed during native admission')
        sources['config_publications'][variant] = dict(reference=publication_ref, admission=admission)
        if configs:
            require(all(cfg[key] == configs['G_POS'][key] for key in shared)
                    and cfg['paths']['model'] == configs['G_POS']['paths']['model'],
                    'Four arms no longer share source100/initial/runtime/validation')
        configs[variant] = cfg; refs[variant] = cf
    require(isinstance(request['analysis_run_id'], str) and bool(request['analysis_run_id'])
            and os.environ.get('E1_RUN_ID') == request['analysis_run_id']
            and request['analysis_run_id'] not in {cfg['run_id'] for cfg in configs.values()},
            'Reader needs its independent explicit run marker')
    sources['binding_checks'] = dict(
        runtime=native['prepare'].verify_runtime(configs['G_POS']),
        common=native['prepare'].verify_common(configs['G_POS']))
    return sources, study_ref, configs, refs, native, completion, analysis


def bind_only(request_ref):
    """Read-only preflight of published sources/configs/runtime/common inputs.

    It does not inspect a partial target, read endpoint evidence, create an
    output directory, or claim source100 weight-file SHA/EM/Adam acceptance.
    """
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Endpoint reader needs HKUST4 UID1001, empty CUDA and no Torch')
    request, rp = read(request_ref)
    sources, study, configs, refs, _, _, _ = bind_inputs(request)
    require('torch' not in sys.modules, 'CPU preflight unexpectedly imported Torch')
    return dict(schema='opsd-signal-endpoint-bindings-v2', accepted=True,
                endpoint_accepted=False, full_study_accepted=False,
                request=rp, study_manifest=study, study_id=configs['G_POS']['study_id'],
                configs=refs, sources=sources, no_sampling=True,
                no_endpoint_evidence_read=True, no_output_files_written=True,
                source100_weight_payload_fresh_SHA=False,
                fresh_numeric_Adam_or_tensor_deserialization=False,
                torch_imported=False, new_sampling_or_training_or_GPU_HTTP_process_queries=0,
                scope='Published binding/runtime/common identity preflight only; no512/EM/Adam/endpoint acceptance')


def verify_target(target, config, cfg, native, completion, validation):
    """No Adam deserialization: original full-file/metadata/end-cursor checks."""
    number = target['completed_outer']
    require(type(number) is int and number in (15, 25) and target['variant'] == config['variant'],
            'Only this arm actual completed15/25 is accepted')
    manifest, mp = read(target['outer_manifest']); index = number - 1
    outer_root = Path(config['paths']['output_root']).resolve() / 'outers' / ('outer-%05d' % index)
    evaluation_root = Path(config['paths']['output_root']).resolve() / 'evaluations' / ('outer-%02d' % number)
    require(Path(mp['path']) == outer_root / 'outer-manifest.json'
            and Path(target['outer_complete']['path']).resolve() == outer_root / 'training' / 'outer-complete.json'
            and Path(target['trainer_exit']['path']).resolve().parent == outer_root / 'trainer-logs'
            and Path(target['parent']['path']).resolve() == evaluation_root / 'parent-complete.json'
            and Path(target['child']['path']).resolve() == evaluation_root / 'child-complete.json',
            'Explicit endpoint paths are outside the actual native outer/evaluation')
    require(manifest['schema'] == 'opsd-online-outer-manifest-v1' and manifest['outer_index'] == index
            and manifest['run_id'] == config['run_id'] and manifest['config_sha256'] == cfg['sha256']
            and manifest['variant'] == config['variant'] and manifest['contract'] == native['checkpoint'].science_contract(config)
            and manifest['runtime_identity'] == config['runtime_manifest']['sha256']
            and manifest['source_model_identity'] == config['source_model_manifest']['sha256']
            and manifest['common_initial_adapter_manifest'] == config['common_initial_adapter_manifest']
            and manifest['common_cohort_manifest'] == config['common_cohort_manifest'],
            'Explicit end-outer training manifest/config/source differs')
    previous_ref = manifest['previous_complete_checkpoint']
    require(previous_ref is not None and manifest['explicit_fresh'] is False
            and same(manifest['outer_old_adapter'], previous_ref), 'Endpoint outer cannot fresh-reset Adam or its outer-old')
    previous, _ = native['checkpoint'].check_checkpoint(previous_ref, config, config['run_id'], expected_outer=index - 1)
    require(previous['cursor']['next_pair_cursor'] == previous['cursor']['usable_pair_count']
            and manifest['cumulative_optimizer_updates_before'] == previous['optimizer_updates'],
            'Endpoint prior outer is unfinished or its Adam counter differs')
    complete, complete_proof = native['checkpoint'].check_outer_complete(
        target['outer_complete'], config, config['run_id'], index,
        manifest['cumulative_optimizer_updates_before'], manifest['P'])
    require(same(complete['checkpoint'], target['raw_checkpoint']), 'Explicit raw checkpoint differs from complete end-outer')
    expected_cp_root = Path(config['paths']['output_root']).resolve() / 'checkpoints' / (
        'update-%08d-outer-%05d' % (complete['optimizer_updates'], index))
    require(Path(target['raw_checkpoint']['path']).resolve() == expected_cp_root / 'manifest.json',
            'Raw endpoint checkpoint is a cache/best/foreign checkpoint path')
    pinned, cp = native['checkpoint'].check_checkpoint(target['raw_checkpoint'], config, config['run_id'],
                                                     complete['optimizer_updates'], index)
    require(same(pinned['training_input']['outer_manifest'], mp)
            and same(pinned['outer_old_adapter'], previous_ref)
            and pinned['sampling_model_identity'] == manifest['sampling_model_identity']
            and complete['cumulative_pair_exposure'] == pinned['cumulative_pair_exposure'] == previous['cumulative_pair_exposure'] + complete['K'],
            'End raw checkpoint does not bind this actual completed training outer')
    trainer_exit, te = read(target['trainer_exit'])
    completion.wait_release(trainer_exit, 1, config, 'training-running')
    # count=1 means one torchrun phase-child record; its actual returncode must be0.
    endpoint = completion.evaluation(target['parent'], target['child'], config, cfg, number,
                                     target['raw_checkpoint'], native, validation)
    child, ch = read(target['child'])
    require(same(child['summary'], target['summary']) and same(child['request'], target['evaluation_request']),
            'Explicit endpoint summary/request pins differ from actual child')
    summary, sr = read(target['summary']); records, rr = rows(child['results'])
    return dict(endpoint=endpoint, outer_manifest=mp, outer_complete=complete_proof['completion'],
                raw_checkpoint=cp['manifest'], trainer_exit=te, summary=sr, results=rr,
                evaluation_request=target['evaluation_request'], optimizer_updates=complete['optimizer_updates'],
                K=complete['K'], complete_cursor=complete['K']), summary, records


def verify_zero(spec, configs, refs, native, completion, validation):
    """Shared zero child belongs to the original G_POS config, even for other arms."""
    cfg = configs['G_POS']; cf = refs['G_POS']
    require(same(spec['config'], cf), 'Common-zero actual config must be G_POS, not target arm')
    require(Path(spec['wrapper']['path']).resolve() == Path(cfg['evaluation']['common_zero_receipt_path']).resolve(),
            'Common-zero wrapper must be the one original shared study receipt')
    native['evaluation'].verify_common_zero(cfg, spec['wrapper'])
    wrapper, wp = read(spec['wrapper'])
    require(wrapper['schema'] == 'opsd-signal-common-zero-reference-v1'
            and same(wrapper['parent'], spec['parent']) and same(wrapper['child'], spec['child']),
            'Explicit common-zero parent/child differ from original shared wrapper')
    proof = completion.evaluation(spec['parent'], spec['child'], cfg, cf, 0, None, native, validation)
    child, ch = read(spec['child'])
    require(same(child['summary'], spec['summary']) and same(child['request'], spec['evaluation_request']),
            'Explicit common-zero summary/request pins differ from actual child')
    summary, sp = read(spec['summary']); records, rp = rows(child['results'])
    return dict(endpoint=proof, wrapper=wp, config=cf, summary=sp, results=rp,
                evaluation_request=spec['evaluation_request']), summary, records


def pair_counts(zero_rows, end_rows, analysis):
    """All probabilities remain NA; only call original paired count implementation."""
    require(len(zero_rows) == len(end_rows) == 512, 'Paired endpoint needs both complete512 arrays')
    features = analysis['replay_math'].FEATURES
    def prepared(records):
        require([r['validation_index'] for r in records] == list(range(512)), 'Formal512 order/coverage differs')
        value = copy.deepcopy(records)
        for row in value:
            require('replay' not in row, 'Native endpoint unexpectedly contains replay fields')
            row['replay'] = dict(metrics={name: None for name in features})
        return value
    left, right = prepared(zero_rows), prepared(end_rows)
    paired = analysis['summary'].paired_cases(left, right)
    labels = {(1, 0): 'lost', (0, 1): 'gained', (1, 1): 'preserved_correct', (0, 0): 'still_wrong'}
    slots = [dict(validation_index=a['validation_index'], dataset=a['dataset'],
                  zero_em=a['protocol_em'], endpoint_em=b['protocol_em'],
                  outcome=labels[a['protocol_em'], b['protocol_em']]) for a, b in zip(left, right)]
    # Reconciliation only; original function owns all reported paired arithmetic.
    require(all(sum(row['outcome'] == label for row in slots) == paired[label] for label in labels.values()),
            'Per-slot classifications disagree with original paired counts')
    compact = {key: value for key, value in paired.items() if key != 'outcome_groups'}
    compact['probability_metric_changes'] = 'NA_no_probability_replay_performed'
    return compact, slots


def run(request_ref, output_dir):
    require(os.getuid() == 1001 and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
            'Endpoint reader needs HKUST4 UID1001, empty CUDA and no Torch')
    request, rp = read(request_ref); output = Path(output_dir).resolve()
    require(output.is_relative_to(Path(STUDY_ROOT) / 'analysis') and not output.exists(),
            'New endpoint output must remain under remote study analysis; do not overwrite')
    sources, study, configs, refs, native, completion, analysis = bind_inputs(request)
    target = request['target']; require(target['variant'] in VARIANTS, 'Foreign target arm')
    cfg = configs[target['variant']]; cf = refs[target['variant']]
    validation, vp = rows(cfg['validation_identity'])
    zero_proof, zero_summary, zero_rows = verify_zero(request['common_zero'], configs, refs, native, completion, validation)
    end_proof, end_summary, end_rows = verify_target(target, cfg, cf, native, completion, validation)
    paired, slots = pair_counts(zero_rows, end_rows, analysis)
    require('torch' not in sys.modules, 'CPU reader unexpectedly imported Torch')
    output.mkdir(parents=True, exist_ok=False)
    slots_path = output / 'paired-slots.jsonl'
    with slots_path.open('x') as stream:
        for row in slots:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    import hashlib
    slots_ref = checked_file(dict(path=str(slots_path), sha256=hashlib.sha256(slots_path.read_bytes()).hexdigest()))
    report = dict(schema='opsd-signal-single-endpoint-pair-v1', accepted=True,
                  study_id=cfg['study_id'], variant=cfg['variant'], completed_outer=target['completed_outer'],
                  raw_outer_index=target['completed_outer'] - 1, optimizer_updates=end_proof['optimizer_updates'],
                  formal_endpoint_slots=512, new_formal_slots_or_generation=0, full_study_accepted=False,
                  request=rp, study_manifest=study, configs=refs, sources=sources, validation=vp,
                  common_zero=zero_proof, endpoint=end_proof,
                  zero=dict(overall=zero_summary['overall'], datasets=zero_summary['datasets']),
                  postraw=dict(overall=end_summary['overall'], datasets=end_summary['datasets']),
                  paired=paired, paired_slots=slots_ref, no_probability_replay=True,
                  all_probability_changes_NA=True, paired_function='published235e_summary.paired_cases',
                  source100_and_common_initial_fresh_file_SHA=True, metadata_checkpoint_acceptance=True,
                  fresh_numeric_Adam_or_tensor_deserialization=False, torch_imported=False,
                  new_sampling_or_training_or_GPU_HTTP_process_queries=0,
                  scope='One real postraw15/25 endpoint versus common zero; no all100/eight-test/final-goal acceptance')
    proof = exclusive(output / 'summary.json', report)
    return dict(accepted=True, output=proof, variant=cfg['variant'], completed_outer=target['completed_outer'],
                zero=report['zero'], postraw=report['postraw'], paired=paired, paired_slots=slots_ref,
                formal_endpoint_slots=512, full_study_accepted=False, no_probability_replay=True,
                fresh_numeric_Adam_or_tensor_deserialization=False)


def main():
    p = argparse.ArgumentParser()
    for name in ('request', 'request-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--bind-only', action='store_true')
    p.add_argument('--output-dir')
    a = p.parse_args()
    if a.bind_only:
        if a.output_dir:
            p.error('--bind-only must not create an output directory')
        result = bind_only(dict(path=a.request, sha256=a.request_sha256))
    else:
        if not a.output_dir:
            p.error('--output-dir is required for endpoint acceptance')
        result = run(dict(path=a.request, sha256=a.request_sha256), a.output_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
