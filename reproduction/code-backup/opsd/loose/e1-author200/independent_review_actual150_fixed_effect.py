#!/usr/bin/env python3
"""Review original fixed-set aggregate metrics; does not invoke training/tools."""
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

W = Path(__file__).resolve().parent
SNAP = W / 'actual-snapshots/3261-20261006T030139996224Z'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
OLD = '4558b1bcd9a66e4ea20d1d8d9531db1c22d5038f1a56c9c2693a301d9692ad29'
SUMMARY = '956660a9526bc78c3d149f077aa63e40f9bcc0d86a961646f1d6451d91909e2b'
SCHEDULE = [('dev', 0, 256), ('stage', 5, 32), ('dev', 50, 256), ('stage', 50, 32),
            ('dev', 100, 256), ('stage', 100, 32), ('dev', 150, 256), ('stage', 150, 32)]
ANSI = re.compile(rb'\x1b\[[0-9;]*m')
PREFIX = b'(main_task pid=1533312) '


def read(path, expected=None):
    path = Path(path)
    a = path.lstat()
    assert stat.S_ISREG(a.st_mode) and not path.is_symlink()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        b = os.fstat(fd)
        chunks = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        c = os.fstat(fd)
    finally:
        os.close(fd)
    d = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_mode')
    ids = [tuple(getattr(x, f) for f in fields) for x in (a, b, c, d)]
    assert all(x == ids[0] for x in ids)
    raw = b''.join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == b.st_size
    assert expected is None or digest == expected, str(path)
    return raw, {'physical_path': str(path), 'sha256': digest, 'bytes': len(raw)}


def pairs(xs):
    result = {}
    for k, v in xs:
        assert k not in result, 'Duplicate JSON key'
        result[k] = v
    return result


def reject_constant(x):
    raise ValueError('Nonfinite JSON: ' + x)


DECODER = json.JSONDecoder(object_pairs_hook=pairs, parse_constant=reject_constant)


def parse_json(raw):
    obj, end = DECODER.raw_decode(raw.decode())
    assert not raw.decode()[end:].strip(), 'Unexpected JSON suffix'
    return obj


def line_records(raw, aggregate):
    evaluations = []
    outers = {}
    consoles = {}
    lines = []
    offset = 0
    for number, chunk in enumerate(raw.split(b'\n')[:-1], 1):
        line = chunk + b'\n'
        clean = ANSI.sub(b'', line.rstrip(b'\r\n'))
        if aggregate:
            if clean.startswith(PREFIX):
                clean = clean[len(PREFIX):]
            elif any(m in clean for m in (b'BASELINE_EVALUATION_METRICS_JSON=', b'OPSD_E1_OUTER_JSON=', b'step:150 - ')):
                raise AssertionError('Unknown record owner')
        proof = {'line': number, 'offset': offset, 'bytes': len(line), 'raw_line_sha256': hashlib.sha256(line).hexdigest()}
        lines.append((clean, proof))
        if clean.startswith(b'BASELINE_EVALUATION_METRICS_JSON='):
            body = clean.split(b'=', 1)[1]
            assert clean.count(b'BASELINE_EVALUATION_METRICS_JSON=') == 1
            evaluations.append({'metrics': parse_json(body), 'json_sha256': hashlib.sha256(body).hexdigest(), 'proof': proof})
        if clean.startswith(b'OPSD_E1_OUTER_JSON='):
            body = clean.split(b'=', 1)[1]
            obj, end = DECODER.raw_decode(body.decode())
            step = obj['actual_outer_updates']
            assert type(step) is int and step not in outers
            assert obj['global_step'] == step and obj['source_commit'] == SOURCE and obj['config_sha256'] == CONFIG
            assert obj['driver_pid'] == 1533312 and obj['job_id'] == '3261'
            suffix = body.decode()[end:]
            # This effect review does not independently accept unrelated outer74 framing.
            # Every outer containing an evaluation, plus the latest outer, must be clean.
            if step in (5, 50, 100, 150, 151):
                assert not suffix.strip(), 'Evaluated outer has suffix'
            outers[step] = {'record': obj, 'proof': proof, 'unreviewed_non_evaluation_suffix': suffix}
        if re.match(rb'^step:(0|5|50|100|150) - ', clean):
            pieces = clean.decode().split(' - ')
            step = int(pieces[0][5:])
            assert step not in consoles
            values = {}
            for piece in pieces[1:]:
                key, value = piece.rsplit(':', 1)
                assert key not in values and math.isfinite(float(value))
                values[key] = value
            consoles[step] = {'values': values, 'proof': proof}
        offset += len(line)
    assert len(evaluations) == 8
    return {'evaluations': evaluations, 'outers': outers, 'consoles': consoles, 'lines': lines}


def initial_repr(records):
    for index, (line, proof) in enumerate(records['lines']):
        if line.startswith(b'("Initial validation metrics: '):
            collected = []
            for text, _ in records['lines'][index:]:
                collected.append(text)
                if text.endswith(b"}')"):
                    text_value = ast.literal_eval(b'\n'.join(collected).decode())
                    assert text_value.startswith('Initial validation metrics: ')
                    return ast.literal_eval(text_value.removeprefix('Initial validation metrics: ')), proof
    raise AssertionError('Initial pprint absent')


def math_review(metrics, per_source):
    sources = ('nq', 'hotpotqa')
    expected = {'val/qa_count/all'}
    expected.update('val/' + key + '/' + source for key in ('qa_count', 'f1', 'em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls') for source in (*sources, 'all'))
    expected.update('val/' + key + '/' + source for key in ('test_score', 'evaluated_count') for source in sources)
    assert set(metrics) == expected
    assert metrics['val/qa_count/all'] == 2 * per_source
    results = {}
    for source in sources:
        assert type(metrics['val/qa_count/' + source]) is int and metrics['val/qa_count/' + source] == per_source
        assert type(metrics['val/evaluated_count/' + source]) is int and metrics['val/evaluated_count/' + source] == per_source
        counts = {}
        for key in ('em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls'):
            value = metrics['val/' + key + '/' + source]
            assert type(value) in (int, float) and math.isfinite(value) and value >= 0
            if key in ('em', 'answer_format', 'format_rate'):
                assert value <= 1
            total = value * per_source
            assert total == round(total), (key, total)
            counts[key + '_sum'] = int(total)
        assert metrics['val/test_score/' + source] == metrics['val/em/' + source]
        f1 = metrics['val/f1/' + source]
        assert math.isfinite(f1) and 0 <= f1 <= 1
        counts['f1_mean'] = f1
        counts['f1_sum'] = f1 * per_source
        counts['count'] = per_source
        results[source] = counts
    total = 2 * per_source
    for key in ('em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls'):
        independently_weighted = sum(results[source][key + '_sum'] for source in sources) / total
        assert independently_weighted == metrics['val/' + key + '/all']
    weighted_f1 = math.fsum(results[s]['f1_sum'] for s in sources) / total
    assert math.isclose(weighted_f1, metrics['val/f1/all'], rel_tol=0, abs_tol=1e-14)
    results['all'] = {'count': total, 'em_correct': sum(results[s]['em_sum'] for s in sources),
                      'em_exact_fraction': str(sum(results[s]['em_sum'] for s in sources)) + '/' + str(total),
                      'f1_independently_weighted': weighted_f1,
                      'format_correct': sum(results[s]['format_rate_sum'] for s in sources),
                      'answer_tag_present': sum(results[s]['answer_format_sum'] for s in sources),
                      'generated_tokens_sum': sum(results[s]['generated_tokens_sum'] for s in sources),
                      'search_calls_sum': sum(results[s]['search_calls_sum'] for s in sources),
                      'weighted_reward_from_original_source_test_scores': math.fsum(metrics['val/test_score/' + s] * per_source for s in sources) / total}
    assert results['all']['weighted_reward_from_original_source_test_scores'] == metrics['val/em/all']
    return results


def main():
    old_raw, old_proof = read(W / 'ACTUAL100_PROGRESS_AND_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json', OLD)
    old = parse_json(old_raw)
    summary_raw, summary_proof = read(SNAP / 'observation-summary.json', SUMMARY)
    summary = parse_json(summary_raw)
    assert summary['expected_source'] == SOURCE and summary['job_id'] == '3261'
    assert len(summary['files']) == 496
    originals = []
    for relative, item in summary['files'].items():
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        raw, proof = read(SNAP / 'original-files' / relative, item['sha256'])
        assert len(raw) == item['bytes']
        originals.append({'relative': relative, **proof})
    assert summary['processes']['1533312']['start_ticks'] == '1829085316'
    assert summary['processes']['1533312']['environment_allowlist']['SLURM_JOB_ID'] == '3261'
    assert summary['processes']['1533312']['uids'] == [0, 0, 0, 0]
    data_raw, data_proof = read(SNAP / 'original-files/bindings/data_manifest.json', DATA)
    data = parse_json(data_raw)
    cfg_raw, cfg_proof = read(SNAP / 'original-files/full_state/global_step_5/resolved-config.json', CONFIG)
    cfg = parse_json(cfg_raw)
    assert cfg['data']['val_data_num'] is None
    root = '/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/common-data/'
    assert cfg['data']['val_files'] == root + 'validation.parquet'
    assert cfg['baselines']['opsd_e1']['stage_files'] == root + 'stage.parquet'
    assert cfg['baselines']['holdout']['manifest_sha256'] == DATA
    assert cfg['trainer']['test_freq'] == cfg['baselines']['opsd_e1']['stage_freq'] == 50
    assert cfg['reward_model']['structure_format_score'] == cfg['reward_model']['final_format_score'] == cfg['reward_model']['retrieval_score'] == 0
    assert cfg['baselines'].get('corrected_evaluation', {}).get('enabled', False) is False
    dataset_bindings = {}
    for name, count in [('validation', 256), ('stage', 32)]:
        split = data['splits'][name]
        assert split['rows'] == count * 2 and split['source_counts'] == {'nq': count, 'hotpotqa': count}
        dataset_bindings[name] = split

    closure_raw, closure_proof = read(SNAP / 'original-files/bindings/import_closure.json', '3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d')
    closure = parse_json(closure_raw)
    source_proofs = {}
    for module, relative in [('verl.trainer.ppo.ray_trainer', 'verl/trainer/ppo/ray_trainer.py'),
                             ('search_r1.baselines.controller', 'search_r1/baselines/controller.py'),
                             ('search_r1.baselines.evaluation', 'search_r1/baselines/evaluation.py')]:
        pin = closure['modules'][module]
        raw, proof = read(W.parent / 'opsd-e1-300' / relative, pin['sha256'])
        source_proofs[module] = {'native_import_path': pin['path'], **proof}

    driver_raw, driver_proof = read(SNAP / 'original-files/ray-stdout-pid-1533312.out')
    aggregate_raw, aggregate_proof = read(SNAP / 'original-files/train.log')
    driver = line_records(driver_raw, False)
    aggregate = line_records(aggregate_raw, True)
    assert max(driver['outers']) == 151 and max(aggregate['outers']) == 150
    for rec in (driver, aggregate):
        initial, _ = initial_repr(rec)
        assert initial == rec['evaluations'][0]['metrics']

    previous = Path(old['snapshot']) / 'original-files'
    previous_originals = {x['relative']: x for x in old['originals']}
    previous_proofs = {}
    for relative in ('ray-stdout-pid-1533312.out', 'train.log'):
        pin = previous_originals[relative]
        raw, proof = read(previous / relative, pin['sha256'])
        assert len(raw) == pin['bytes']
        previous_proofs[relative] = proof
        later = driver_raw if relative.startswith('ray-') else aggregate_raw
        assert later.startswith(raw), 'Historical full prefix differs'
        earlier = line_records(raw, not relative.startswith('ray-')) if raw.count(b'BASELINE_EVALUATION_METRICS_JSON=') == 8 else None
        # Original100 has six events; compare the original complete JSON bodies independently.
        old_events = []
        for line in raw.splitlines():
            clean = ANSI.sub(b'', line)
            if not relative.startswith('ray-') and clean.startswith(PREFIX):
                clean = clean[len(PREFIX):]
            if clean.startswith(b'BASELINE_EVALUATION_METRICS_JSON='):
                old_events.append(parse_json(clean.split(b'=', 1)[1]))
        assert len(old_events) == 6
        assert old_events == [x['metrics'] for x in driver['evaluations'][:6]]
        assert old_events == [x['original_metrics'] for x in old['fixed_evaluation_coverage']['events']]

    reviewed = []
    for i, (kind, step, per_source) in enumerate(SCHEDULE):
        d = driver['evaluations'][i]
        a = aggregate['evaluations'][i]
        assert d['metrics'] == a['metrics'] and d['json_sha256'] == a['json_sha256']
        math_result = math_review(d['metrics'], per_source)
        joins = {}
        for name, rec in [('driver', driver), ('aggregate', aggregate)]:
            event = rec['evaluations'][i]
            prefix = 'opsd_stage/' if kind == 'stage' else ''
            if step:
                outer = rec['outers'][step]
                assert event['proof']['line'] > rec['outers'][step - 1]['proof']['line']
                assert event['proof']['line'] < outer['proof']['line']
                assert all(outer['record']['metrics'][prefix + k] == v for k, v in event['metrics'].items())
                joins[name + '_outer'] = outer['proof']
            else:
                assert event['proof']['line'] < rec['outers'][1]['proof']['line']
            console = rec['consoles'][step]
            assert event['proof']['line'] < console['proof']['line']
            assert all(console['values'][prefix + k] == f'{float(v):.3f}' for k, v in event['metrics'].items())
            joins[name + '_console'] = console['proof']
        reviewed.append({'kind': kind, 'step': step, 'original_metrics': d['metrics'], 'independent_math': math_result,
                         'driver_JSON': d['proof'], 'aggregate_JSON': a['proof'], 'JSON_payload_sha256': d['json_sha256'], 'pairing_proofs': joins})

    latest = driver['outers'][151]['record']['metrics']
    dev = [e for e in reviewed if e['kind'] == 'dev']
    result = {'version': 1, 'status': 'PASS_ACTUAL150_FIXED_SET_AGGREGATE_EVIDENCE_WITH_NEGATIVE_BASELINE_EFFECT',
              'reviewer_role': 'INDEPENDENT_EFFECT_ORIGINALS_AND_MATH_REVIEWER', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'snapshot': str(SNAP), 'snapshot_summary': summary_proof, 'actual496_local_originals_full_SHA_verified': True,
              'originals': originals, 'source_commit': SOURCE, 'config_sha256': CONFIG, 'training_job_id': '3261',
              'driver_pid': 1533312, 'driver_start_ticks': '1829085316', 'native_import_closure': closure_proof,
              'fixed_evaluator_source_proofs': source_proofs, 'dataset_manifest': data_proof, 'frozen_config': cfg_proof,
              'fixed_dataset_bindings': dataset_bindings, 'fixed_dataset_scope': 'Same prebuilt manifest rows/IDs and same-process unshuffled fixed dataloaders; no per-evaluation raw question-ID or response dump.',
              'driver_original': driver_proof, 'aggregate_original': aggregate_proof, 'previous100_review': old_proof,
              'previous100_full_log_prefixes_unchanged': previous_proofs, 'first_six_evaluation_events_exactly_match_4558': True,
              'all_eight_events': reviewed, 'driver_latest_outer_marker': 151, 'aggregate_latest_outer_marker': 150,
              'latest151_random_training_batch_reward': latest['critic/rewards/mean'],
              'latest151_reward_scope': 'Random 512-question training batch, not fixed dev150/stage150.',
              'dev_EM_counts': [e['independent_math']['all']['em_correct'] for e in dev],
              'dev_EM_means': [e['original_metrics']['val/em/all'] for e in dev],
              'dev_F1_means': [e['original_metrics']['val/f1/all'] for e in dev],
              'effect_conclusion': 'Negative versus initialization: dev EM94/512 to31/512 and F1.241593 to.084989; 150 EM equals50 and exceeds100 descriptively, but still markedly below initialized model. Full-format rate is0. No improvement or significance claim.',
              'per_question_response_rescoring_performed': False, 'statistical_significance_established': False,
              'training_or_GPU_or_remote_execution_performed': False, 'canonical_edit_performed': False,
              'limitations': ['Aggregate checks do not recover absent full per-question responses/targets/masks.',
                             'No per-question paired significance test or rescore was possible.',
                             'This review covers fixed evaluation and selected containing outer records; unrelated non-evaluation outer74 framing is not re-reviewed here.',
                             'No full native/HF150 content, new restore, wholeHF200, final51713, Slurm terminal or resource-release acceptance.',
                             '496 local raw members are complete captured files; the runtime directory symlink_not_followed notice remains in the original snapshot summary.']}
    out = W / 'ACTUAL150_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'report': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                      'dev_correct_counts': result['dev_EM_counts'], 'latest151_batch_reward': result['latest151_random_training_batch_reward']}))


if __name__ == '__main__':
    main()
