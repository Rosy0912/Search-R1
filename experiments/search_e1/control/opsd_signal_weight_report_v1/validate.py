"""Formal report contracts. Reuse accepted statistics; never recompute them."""
import math
try:
    from .common import VARIANTS, ENDPOINTS, Reader, require, same, finite, native_packet, published
except ImportError:
    from common import VARIANTS, ENDPOINTS, Reader, require, same, finite, native_packet, published


def counts(value, denominator, label):
    require(value.get('denominator', value.get('questions')) == denominator and type(value.get('hits')) is int
            and 0 <= value['hits'] <= denominator, 'Wrong EM count/denominator: ' + label)
    require(math.isclose(finite(value['em'], label), value['hits'] / denominator, abs_tol=1e-12), 'Inconsistent EM: ' + label)
    require(type(value['final_invalid']) is int and type(value['valid_gold_mismatch']) is int and
            value['final_invalid'] >= 0 and value['valid_gold_mismatch'] >= 0 and
            value['hits'] + value['final_invalid'] + value['valid_gold_mismatch'] == denominator,
            'Failure classes do not partition EM: ' + label)
    return {k: value[k] for k in ('hits', 'em', 'final_invalid', 'valid_gold_mismatch', 'no_executed_search')}


def validate(request_ref, actual_entry=None):
    reader = Reader(); req = reader.json(request_ref)
    require(req['schema'] == 'opsd-signal-final-report-request-v1', 'Wrong report request schema')
    published(reader, req['source'])
    if actual_entry is not None:
        from pathlib import Path
        require(Path(actual_entry).resolve() == Path(req['source']['helper_root']).resolve() / 'report.py',
                'Actual report entry outside published namespace')
    cov, _ = native_packet(reader, req['coverage'], 'coverage')
    require(cov['schema'] == 'opsd-signal-final-coverage-v1' and cov.get('coverage_accepted') is True and
            cov.get('final_accepted') is True and cov['complete_groups'] == 4 and cov['complete_outers'] == 100 and
            cov['train_trajectory_slots'] == 256000 and cov['formal_native512_slots'] == 4608 and
            cov['study_id'] == req['study_id'] and cov['torch_imported'] is False,
            'Formal final report requires actual final-accepted100/256000/4608 coverage')
    required_external={'final_numeric_adam','protected_and_outer_end_backups','published_source','full_vocab_scope','effect_and_relationship_analysis'}
    require(set(cov['final_requirements'])==required_external and
            all(e.get('accepted') is True for e in cov['final_requirements'].values()),
            'Final coverage contradicts an incomplete external requirement')
    coverage_request = reader.json(req['coverage_request'])
    require(coverage_request['schema'] == 'opsd-signal-final-coverage-request-v1' and
            same(coverage_request['study_manifest'], cov['study_manifest']) and
            same(cov['request'],req['coverage_request']), 'Original coverage request differs from actual CLI output')
    require(set(req['configs']) == set(req['train_ledgers']) == set(cov['final_checkpoints']) == set(VARIANTS),
            'Four original configs/ledgers/final checkpoints required')
    original_groups = {g['variant']: g for g in coverage_request['groups']}
    require(set(original_groups) == set(VARIANTS), 'Coverage request group quartet incomplete')
    outer_index = {(r['variant'], r['outer_index']): r for r in cov['outer_rows']}
    require(len(cov['outer_rows']) == len(outer_index) == 100 and set(outer_index) == {(v, i) for v in VARIANTS for i in range(25)},
            'Final coverage ordered100 outer rows incomplete')
    configs = {}; curves = {}; final_cps = {}
    for v in VARIANTS:
        require(same(req['configs'][v], original_groups[v]['config']) and
                same(req['train_ledgers'][v], original_groups[v]['train_ledger']), 'Original config/ledger changed: ' + v)
        cfg = reader.json(req['configs'][v]); configs[v] = cfg
        require(cfg['variant'] == v and cfg['study_id'] == req['study_id'], 'Foreign config/run/study')
        ledger = reader.json(req['train_ledgers'][v])
        require(ledger.get('accepted') is True and ledger['variant'] == v and ledger['run_id'] == cfg['run_id'] and
                ledger['outer_iterations'] == 25 and [r['outer_index'] for r in ledger['rows']] == list(range(25)),
                'Formal TRAIN ledger must contain original25 ordered complete outer rows')
        curve = []
        for i, row in enumerate(ledger['rows']):
            original = outer_index[v, i]
            require(row['schema'] == 'opsd-signal-outer-summary-v1' and row.get('accepted') is True and
                    row['variant'] == v and row['run_id'] == cfg['run_id'] and row['config_sha256'] == req['configs'][v]['sha256'] and
                    row['train_denominator'] == 2560 and row['train_post_update'] is False and
                    row['train_correct'] == original['train_correct'] and same(row['raw_checkpoint'], original['checkpoint']) and
                    row['optimizer_updates'] == original['optimizer_updates'] and row['K'] == original['K'],
                    'Partial/wrongrun/postupdate TRAIN row: ' + v + '/' + str(i))
            require(type(row['train_correct']) is int and 0 <= row['train_correct'] <= 2560 and
                    math.isclose(finite(row['train_em'], 'TRAIN EM'), row['train_correct']/2560, abs_tol=1e-12), 'TRAIN EM count differs')
            curve.append(dict(outer=i+1, strict_correct=row['train_correct'], denominator=2560, em=row['train_em'],
                              optimizer_updates=row['optimizer_updates'], K=row['K'], P=row['P'],
                              unweighted_post_loss=row['actual_outer_unweighted_post_loss'],
                              weighted_post_loss=row['actual_outer_weighted_post_loss'],
                              raw_loss=row['actual_outer_raw_loss'], weight=row['weight']))
        cp = reader.json(cov['final_checkpoints'][v]); final_cps[v] = cov['final_checkpoints'][v]
        require(cp['run_id'] == cfg['run_id'] and cp['cursor']['outer_index'] == 24 and
                cp['cursor']['next_pair_cursor'] == cp['cursor']['usable_pair_count'] and
                same(cov['final_checkpoints'][v], outer_index[v,24]['checkpoint']) and
                cp['optimizer_updates'] == curve[-1]['optimizer_updates'], 'Final model is not actual rawouter24 checkpoint')
        curves[v] = curve
    zero = reader.json(req['common_zero'])
    require(zero['schema'] == 'opsd-signal-common-zero-reference-v1' and zero.get('accepted') is True and
            zero['study_id'] == req['study_id'] and same(req['common_zero'], coverage_request['common_zero']), 'Common zero changed')
    expected_evals = {(e['child']['path'],e['child']['sha256']) for e in cov['evaluations']}
    require(len(req['evaluations']) == 9 and {(e['child']['path'],e['child']['sha256']) for e in req['evaluations']} == expected_evals,
            'Formal nine native endpoint refs differ from final coverage')
    models = {}; endpoint_refs = {}
    for entry in req['evaluations']:
        parent = reader.json(entry['parent']); child = reader.json(entry['child']); summary = reader.json(entry['summary'])
        key = (child['version'], child['completed_outer']); name = key[0] + '/' + str(key[1])
        require(key in ENDPOINTS and name not in models and child['schema'] == 'opsd-signal-native512-complete-v1' and
                child.get('accepted') is True and child['primary_slots'] == 512 and child['actual_child_exit_and_release'] is True and
                child['checkpoint_metadata_unchanged'] is True and child['study_id'] == req['study_id'] and
                same(child['summary'], entry['summary']), 'Incomplete/foreign native endpoint')
        variant = child['variant']; cfg = configs[variant]
        require(child['run_id'] == cfg['run_id'] and same(child['config'],req['configs'][variant]) and
                parent['schema'] == 'opsd-signal-evaluation-parent-v1' and parent.get('accepted') is True and
                parent['run_id'] == cfg['run_id'] and parent['completed_outer'] == key[1] and
                parent['config_sha256'] == req['configs'][variant]['sha256'] and same(parent['child'],entry['child']),
                'Endpoint parent/config/run differs')
        require(summary['schema'] == 'opsd-signal-native512-summary-v1' and summary.get('accepted') is True and
                summary['run_id'] == cfg['run_id'] and summary['version'] == key[0] and summary['completed_outer'] == key[1],
                'Native summary belongs to another model/run/time')
        matching = next(e for e in cov['evaluations'] if same(e['child'], entry['child']))
        require(same(matching['parent'],entry['parent']) and matching['hits'] == summary['overall']['hits'], 'Final coverage parent/EM differs')
        if key[1] == 0:
            require(same(zero['parent'],entry['parent']) and same(zero['child'],entry['child']), 'Common zero wrapper points elsewhere')
        else:
            raw = outer_index[variant,key[1]-1]['checkpoint']
            require(parent['raw_checkpoint'] == raw and child['raw_checkpoint']['kind'] == 'raw_end_of_outer' and
                    same(child['raw_checkpoint']['manifest'],raw), 'Wrong15/25 endpoint checkpoint/cache/best')
        models[name] = dict(overall=counts(summary['overall'],512,name),
                            datasets={d:counts(summary['datasets'][d],256,name+'/'+d) for d in ('nq','hotpotqa')})
        endpoint_refs[name] = {k: entry[k] for k in ('parent','child','summary')}
    require(set(models) == {v+'/'+str(t) for v,t in ENDPOINTS}, 'Nine unique endpoint slots required')
    require(set(req['analysis']) == {'effects','train_relationship','heldout_relationship'}, 'Three original statistical reports required')
    analyses = {}
    evidence = cov['final_requirements']['effect_and_relationship_analysis']
    require(evidence['accepted'] is True, 'Coverage did not accept original analysis')
    for kind, packet in req['analysis'].items():
        report, native_request = native_packet(reader, packet, kind)
        binding = reader.json(packet['input_binding'])
        old = evidence['evidence'][kind]
        require(same(old['execution']['report'],packet['output']) and same(old['input_binding'],packet['input_binding']) and
                binding['schema'] == 'opsd-signal-analysis-input-binding-v1' and binding['input_format_bridge_only'] is True and
                same(binding['original_output'],packet['output']) and binding['study_id'] == req['study_id'] and
                binding['configs'] == req['configs'] and binding['final_checkpoints'] == final_cps and
                {(r['path'],r['sha256']) for r in binding['formal_evaluations']} == expected_evals,
                'Analysis output/bridge/config/checkpoint/endpoint binding differs')
        expected = {'effects':('opsd-signal-weight-final-effects-v1','formal_predictions',4608),
                    'heldout_relationship':('opsd-signal-heldout-response-relations-v1','formal_EM_rows',4608),
                    'train_relationship':('opsd-signal-train-analysis-summary-v1','completed_outers',100)}[kind]
        require(report['schema'] == expected[0] and report[expected[1]] == expected[2], 'Partial/foreign analysis report')
        analyses[kind] = report
    effects = analyses['effects']; heldout = analyses['heldout_relationship']; train = analyses['train_relationship']
    require(effects['primary_time'] == 25 and effects['primary_family_size'] == 2 and effects['best_checkpoint_selection'] is False,
            'Primary endpoint/family changed or best selected')
    primary_keys={'G_NEG/25->G_POS/25','H_NEG/25->H_POS/25'}
    require({k for k,p in effects['paired'].items() if p['primary_comparison']} == primary_keys,
            'Predeclared final primary comparison direction/family differs')
    for key,p in effects['paired'].items():
        left,right=key.split('->');require(left in models and right in models,'Foreign paired endpoint')
        require(p['questions']==512 and all(type(p[k]) is int and p[k]>=0 for k in ('gained','lost','preserved','still_wrong')) and
                sum(p[k] for k in ('gained','lost','preserved','still_wrong'))==512 and
                p['preserved']+p['lost']==models[left]['overall']['hits'] and
                p['preserved']+p['gained']==models[right]['overall']['hits'] and
                math.isclose(p['delta_percentage_points'],(p['gained']-p['lost'])*100/512,abs_tol=1e-12),
                'Paired counts/direction do not match original native endpoints')
        low=finite(p['ci95']['low'],'native CI low');high=finite(p['ci95']['high'],'native CI high')
        require(low<=high and p['ci95']['unit']=='percentage_points','Native interval format differs')
        if key in primary_keys:
            test=p['exact_mcnemar_holm']
            require(all(0<=finite(test[k],'native p')<=1 for k in ('raw_p','holm_p')),
                    'Native primary p-value outside probability bounds')
    require(train['study_id'] == req['study_id'] and train['complete_100_outers'] is True and set(train['groups']) == set(VARIANTS),
            'TRAIN analysis missing full100/study identity')
    for name, value in models.items():
        require(effects['models'][name]['overall']['hits'] == value['overall']['hits'] and
                heldout['models'][name]['all512']['hits'] == value['overall']['hits'], 'Statistical report EM disagrees with native endpoint')
        for dataset in ('nq','hotpotqa'):
            require(effects['models'][name]['datasets'][dataset]['hits'] == value['datasets'][dataset]['hits'] and
                    heldout['models'][name]['datasets'][dataset]['hits'] == value['datasets'][dataset]['hits'], 'Dataset EM disagrees')
    debug = reader.json(req['debug_retention']); movement = reader.json(req['parameter_movement'])
    require(debug['schema'] == 'opsd-signal-debug-retention-v1' and debug.get('accepted') is True and
            debug['full_study_complete'] is True and debug['complete_outers'] == 100 and debug['sampling_workers'] == 800 and
            debug['teacher_parents'] == 100 and debug['study_id'] == req['study_id'], 'Full debug retention evidence incomplete')
    require(same(debug['study_manifest'],cov['study_manifest']) and len(debug['per_outer'])==100 and
            {(e['variant'],e['outer_index']) for e in debug['per_outer']}==set(outer_index),
            'Debug retention references another study or incomplete outer chain')
    for entry in debug['per_outer']:
        v=entry['variant'];i=entry['outer_index']
        require(entry['run_id']==configs[v]['run_id'] and same(entry['config'],req['configs'][v]) and
                same(entry['end_checkpoint'],outer_index[v,i]['checkpoint']), 'Debug retention raw/run/config differs')
    require(movement['schema'] == 'opsd-signal-parameter-movement-v1' and movement.get('accepted') is True and
            movement['full_study_complete'] is True and movement['mode'] == 'final' and movement['configs'] == req['configs'],
            'Four-arm final parameter movement missing')
    require(same(movement['study'],cov['study_manifest']),'Parameter movement references another original study')
    comparisons = movement['comparisons']
    require(len(comparisons) == 12 and {(x['variant'],x['source'],x['destination']) for x in comparisons} ==
            {(v,a,b) for v in VARIANTS for a,b in (('common_zero','outer15'),('outer15','outer25'),('common_zero','outer25'))},
            'Movement must cover exact0->15/15->25/0->25, each variant')
    for v in VARIANTS:
        endpoint = movement['endpoints'][v]
        require([e['completed_outer'] for e in endpoint] == [15,25] and
                all(same(e['evidence']['manifest'],outer_index[v,e['outer_index']]['checkpoint']) for e in endpoint),
                'Movement raw endpoint identity differs')
    reader.stable_end()
    return dict(study_id=req['study_id'],curves=curves,models=models,endpoint_refs=endpoint_refs,analyses=analyses,
                debug=debug,movement=movement,coverage=cov,request=req,request_ref=request_ref,reader=reader,synthetic=False)
