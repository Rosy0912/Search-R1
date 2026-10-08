"""CPU postprocess of an already completed pinned protocol audit; never reruns it.

Only adds compact proof/paired statistics to that exclusive analysis directory.
No model imports, CUDA, HTTP, generation, runtime/source edits or raw text output.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

D = Path('/data/shaoshitong/search-e1-opsd-debug-20261008')
O = Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
RUN_ID = '894021c8f55f436f96bb3cb146ab05bb'
R = D / ('runs/debug-' + RUN_ID)
A = R / 'protocol-audit-full192-20261008-v1'
H = D / 'source/opsd_debug_analysis/summarize_protocol.py'
HELPER_SHA = 'a3ce95854029be735051ebbe6ab6976c15adc1f3758854fbaa051931b7c9f60a'
SUMMARY_SHA = '6ceb906987ade78f13bc0a345c63e4b802613cc4ae0e0a5d6b0c7360baaca62a'
PRIVATE_CASES_SHA = 'f0b2eb93b4a51fbb0041a8b2a4fe244f5cc7abaad749bc37057530f9cd155903'
VERSIONS = ('source100', 'update1', 'update5', 'epoch1')
CLASSES = ('strict_correct', 'valid_wrong', 'final_invalid')


def stable_read(path):
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    assert tuple(getattr(before, k) for k in keys) == tuple(getattr(after, k) for k in keys)
    return body, dict(path=str(path), sha256=hashlib.sha256(body).hexdigest(),
                      bytes=len(body), inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                      ctime_ns=before.st_ctime_ns)


def classify(v):
    assert v['status'] == 'analyzed' and v['strict_em'] in (0, 1)
    assert type(v['final_answer_protocol_valid']) is bool
    assert type(v['no_executed_search']) is bool
    assert type(v['search_calls']) is int and 0 <= v['search_calls'] <= 4
    assert v['no_executed_search'] == (v['search_calls'] == 0)
    assert not v['strict_em'] or v['final_answer_protocol_valid']
    if v['strict_em']:
        return 'strict_correct'
    return 'valid_wrong' if v['final_answer_protocol_valid'] else 'final_invalid'


def write_exclusive(name, value):
    with (A / name).open('x') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')
    return stable_read(A / name)[1]


def main():
    assert not (A / 'terminal-execution-proof.json').exists()
    assert not (A / 'paired-and-termination-summary.json').exists()
    hbody, helper = stable_read(H)
    assert helper['sha256'] == HELPER_SHA
    sb, summary_proof = stable_read(A / 'summary.json')
    summary = json.loads(sb)
    assert summary_proof['sha256'] == SUMMARY_SHA
    assert summary['run_id'] == RUN_ID
    assert summary['selected_cases'] == summary['completed_cases'] == 192
    assert all(summary['skip_counts'].get(k, 0) == 0 for k in (
        'not_complete_result', 'trailing_partial_results_line', 'selected_cases_not_yet_complete'))
    assert summary['skip_counts']['teacher_no_successful_reference'] == 64
    cb, private_proof = stable_read(A / 'cases.jsonl')
    assert private_proof['sha256'] == PRIVATE_CASES_SHA == summary['cases_output']['sha256']
    records = [json.loads(row) for row in cb.splitlines()]
    assert cb.endswith(b'\n') and len(records) == len({row['case_id'] for row in records}) == 192
    assert len({row['question_id'] for row in records}) == 192
    cells = Counter((row['dataset'], row['group']) for row in records)
    assert len(cells) == 6 and set(cells.values()) == {32}
    final_body, final_proof = stable_read(R / 'controller-final.json')
    final = json.loads(final_body)
    assert final['run_id'] == RUN_ID and final['status'] == 'WORKERS_COMPLETE'
    assert final['controller_pid'] == 33834
    assert final['child_exit_codes'] == {'33998': 0, '33999': 0, '34000': 0}
    config_body, config_proof = stable_read(R / 'config.json')
    assert json.loads(config_body)['run_id'] == RUN_ID
    assert config_proof['sha256'] == '06b110c20a248364e99c6f037e87d8da6c8fc9b6fe27506f879bd58fb4a54cd4'
    child_body, child_proof = stable_read(R / 'children-start.json')
    children = json.loads(child_body)
    assert [c['pid'] for c in children] == [33998, 33999, 34000]
    workers = []
    terminal_results = set()
    for i, child in enumerate(children):
        status_body, status_proof = stable_read(R / ('worker-%d' % i) / 'status.json')
        status = json.loads(status_body)
        assert status['status'] == 'COMPLETE'
        assert status['assigned_cases'] == status['completed_cases'] == 64
        assert status['uid'] == child['uid'] == 1005
        assert status['pid'] == child['pid']
        assert status['proc_starttick'] == child['starttick'] == 922457979
        result_body, result_proof = stable_read(R / ('worker-%d' % i) / 'case-results.jsonl')
        assert result_proof['sha256'] == status['results_sha256'] and result_body.endswith(b'\n')
        result_rows = [json.loads(row) for row in result_body.splitlines()]
        assert len(result_rows) == len({row['case_id'] for row in result_rows}) == 64
        assert all(row['status'] == 'complete' for row in result_rows)
        assert not terminal_results.intersection(row['case_id'] for row in result_rows)
        terminal_results.update(row['case_id'] for row in result_rows)
        prefix = summary['source']['completed_results_prefixes'][i]
        assert prefix['path'] == result_proof['path'] and prefix['prefix_bytes'] == len(result_body)
        assert prefix['prefix_sha256'] == result_proof['sha256'] and not prefix['trailing_partial_line_skipped']
        workers.append(dict(index=i, pid=status['pid'], uid=1005, starttick=922457979,
                            completed_cases=64, status='COMPLETE', child_exit_code=0,
                            ended_unix=status['ended_unix'], status_file=status_proof,
                            complete_final_result_file=result_proof))
    assert terminal_results == {row['case_id'] for row in records}
    tmux = subprocess.run(['tmux', 'list-panes', '-t', 'e1-opsd-debug-20261008', '-F',
                           '#{pane_id}|#{pane_dead}|#{pane_dead_status}|#{pane_pid}'],
                          capture_output=True, text=True, check=True)
    assert tmux.stdout.strip() == '%60|1|0|33832'
    # Rehash only the tiny protocol sources and actual generation trace files.
    # No model/index/NPY scans; existing M1 acceptance handles their separate scope.
    for item in summary['source']['science'].values():
        assert stable_read(Path(item['path']))[1]['sha256'] == item['sha256']
    totals = defaultdict(Counter)
    by_cell = defaultdict(Counter)
    terminations = defaultdict(Counter)
    pairs = defaultdict(Counter)
    trace_rows = []
    for row in records:
        dataset, group = row['dataset'], row['group']
        for version, v in row['versions'].items():
            trace_body, trace_proof = stable_read(Path(v['trace']['path']))
            assert trace_proof['sha256'] == v['trace']['sha256']
            trace_rows.append(dict(case_id=row['case_id'], version=version,
                                   path=trace_proof['path'], sha256=trace_proof['sha256']))
            if version == 'teacher' and group == 'all5_wrong':
                assert v['status'] == 'skipped' and v['reason'] == 'no_successful_reference'
                totals[version]['skipped_no_reference'] += 1
                continue
            cls = classify(v)
            for key in (version, (version, dataset, group)):
                counter = totals[key] if isinstance(key, str) else by_cell[key]
                counter['analyzed'] += 1
                counter[cls] += 1
                counter['no_executed_search'] += v['no_executed_search']
                counter['invalid_literal_tag_matches_gold_not_em'] += v['invalid_tag_candidate_matches_gold']
                counter['invalid_tag_unclosed_think'] += v['invalid_tag_candidate_unclosed_think']
            term = v['termination']
            assert isinstance(term, str) and term
            terminations[(version, dataset, group)][term] += 1
            terminations[(version, 'ALL', 'ALL')][term] += 1
            terminations[(version, 'class=' + cls, 'ALL')][term] += 1
            if version in VERSIONS:
                base = row['versions']['source100']
                origin = classify(base)
                for cell in (('ALL', 'ALL'), (dataset, group)):
                    p = pairs[(version,) + cell]
                    p['total'] += 1
                    p[origin + '->' + cls] += 1
                    p['no_search:%d->%d' % (base['no_executed_search'], v['no_executed_search'])] += 1
                    p['termination:' + base['termination'] + '->' + term] += 1
    assert len(trace_rows) == 960  # 768 student actual + 128 teacher actual + 64 explicit teacher-N/A.
    for version in VERSIONS:
        assert totals[version]['analyzed'] == sum(totals[version][c] for c in CLASSES) == 192
        s = summary['by_version'][version]
        assert [totals[version][c] for c in CLASSES] == [s['strict_correct'], s['protocol_valid_wrong_answer'], s['final_answer_protocol_invalid']]
        assert totals[version]['no_executed_search'] == s['no_executed_search']
        assert pairs[(version, 'ALL', 'ALL')]['total'] == 192
    assert totals['teacher']['analyzed'] == 128 and totals['teacher']['skipped_no_reference'] == 64
    assert sum(totals['teacher'][c] for c in CLASSES) == 128
    trace_body = json.dumps(trace_rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    trace_manifest = write_exclusive('classification-trace-manifest.json', trace_rows)
    proof = dict(schema='full192-terminal-bound-protocol-proof-v1', checked_unix=time.time(),
                 cpu_only=True, no_gpu_http=True, helper=helper, published_helper_commit='d1e2215',
                 run_id=RUN_ID, controller_final=final_proof, run_config=config_proof,
                 children_start=child_proof, workers=workers,
                 tmux_real_terminal=dict(pane_id='%60', pane_dead=1, pane_dead_status=0, shell_pid=33832),
                 summary=summary_proof, per_case_private=private_proof,
                 classification_trace_manifest=trace_manifest, trace_manifest_canonical_sha256=hashlib.sha256(trace_body).hexdigest(),
                 generated_trace_files_rechecked=960,
                 original_finalization_guard=dict(failed_after_successful_helper=True,
                     error="KeyError: not_complete_result (a zero Counter key absent)",
                     correction='get(key, 0); postprocess completed output without repeating or overwriting classification',
                     exact_initial_helper_elapsed_seconds_unavailable=True),
                 helper_classification_created_unix=summary['created_unix'],
                 postprocessor_source_sha256=os.environ['ANALYSIS_SOURCE_SHA256'])
    terminal_proof = write_exclusive('terminal-execution-proof.json', proof)
    paired = dict(schema='full192-protocol-paired-termination-v1', checked_unix=time.time(),
                  run_id=RUN_ID, selected_unique_cases=192, six_cells_each=32,
                  scope='Same fixed stratified diagnostic sample; not population/native512/seven-benchmark EM',
                  interpretation=dict(classes='Mutually exclusive strict_correct/valid_wrong/final_invalid; frozen QA parser and reward',
                     no_search='Separate axis grounded in search_calls and actual query turn provenance',
                     termination='Separate recorded generation termination axis, not a new failure class',
                     teacher='Own privileged autoregressive rollout for 128 eligible cases; not training q on common history',
                     literal_matches='Diagnostic normalized literal tag matches only, never strict or semantic correctness'),
                  provenance=dict(summary=summary_proof, private_cases=private_proof,
                                  terminal_proof=terminal_proof, classification_trace_manifest=trace_manifest,
                                  postprocessor_source_sha256=os.environ['ANALYSIS_SOURCE_SHA256']),
                  by_version={v: dict(c) for v, c in sorted(totals.items()) if isinstance(v, str)},
                  cells=[dict(version=v, dataset=d, group=g, **dict(c)) for (v, d, g), c in sorted(by_cell.items())],
                  paired_from_source100=[dict(target_version=v, dataset=d, group=g, counts=dict(c))
                                         for (v, d, g), c in sorted(pairs.items())],
                  termination_axis=[dict(version=v, dataset=d, group=g, counts=dict(c))
                                    for (v, d, g), c in sorted(terminations.items())])
    paired_proof = write_exclusive('paired-and-termination-summary.json', paired)
    # Final tiny source/end/analysis proof checks reject any concurrent change.
    assert stable_read(H)[1] == helper
    assert stable_read(R / 'controller-final.json')[1] == final_proof
    assert stable_read(A / 'summary.json')[1] == summary_proof
    assert stable_read(A / 'cases.jsonl')[1] == private_proof
    print(json.dumps(dict(accepted=True, terminal_proof=terminal_proof,
                          paired_summary=paired_proof, trace_manifest=trace_manifest,
                          by_version=paired['by_version'],
                          global_pairs=[p for p in paired['paired_from_source100'] if p['dataset'] == 'ALL'],
                          global_termination=[p for p in paired['termination_axis'] if p['dataset'] == 'ALL'])))


if __name__ == '__main__':
    main()
