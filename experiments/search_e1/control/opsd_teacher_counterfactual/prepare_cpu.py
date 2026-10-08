"""Prepare private M3 states from frozen192 cases; no model/tokenizer/HTTP/GPU.

Root reviews/publishes this source before invoking the command on real artifacts.
Output is CPU preparation, not executable GPU admission or teacher-effect evidence.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

try:
    from .state import reconstruct_original, reconstruct_hf, sha
    from .references import select, conditions
except ImportError:
    from state import reconstruct_original, reconstruct_hf, sha
    from references import select, conditions

CASES_SHA = '1703b15b1a1b3c5b91b13422fa314ab9a3b83f7bf7bf8a85dd7afcdaecdbf082'
CORE_SHA = '0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc'
QA_SHA = '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
MAX_BYTES = 128 << 20


def stat(path):
    s = Path(path).stat(); return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read(path, expected=None):
    path = Path(path); before = stat(path)
    if before['bytes'] > MAX_BYTES:
        raise ValueError('CPU preparation bounded file limit exceeded')
    data = path.read_bytes(); actual = hashlib.sha256(data).hexdigest()
    if stat(path) != before or expected is not None and actual != expected:
        raise ValueError('Immutable source changed/hash differs: ' + str(path))
    return data, dict(path=str(path.resolve()), stat=before, sha256=actual)


def science(runtime):
    runtime = Path(runtime).resolve(); identities = {}
    for relative, expected in [('search_e1/core.py', CORE_SHA), ('upstream/Search-R1/search_r1/qa_protocol.py', QA_SHA)]:
        _, identities[relative] = read(runtime / relative, expected)
    qa_path = runtime / 'upstream/Search-R1/search_r1/qa_protocol.py'
    name = 'search_r1.qa_protocol'
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != qa_path:
        raise ValueError('Foreign previously imported QA')
    if 'search_r1' not in sys.modules:
        package = types.ModuleType('search_r1'); package.__path__ = [str(qa_path.parent)]; sys.modules['search_r1'] = package
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, qa_path); qa = importlib.util.module_from_spec(spec); sys.modules[name] = qa; spec.loader.exec_module(qa)
    spec = importlib.util.spec_from_file_location('_m3_frozen_core', runtime / 'search_e1/core.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    return core, sys.modules[name], identities


def prepare(cases_dir, runtime, output, trace_root=None):
    output = Path(output)
    if output.exists(): raise FileExistsError('Output must be exclusive/new')
    helper_proof = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        _, helper_proof[path.name] = read(path)
    core, qa, science_proof = science(runtime)
    data, source = read(Path(cases_dir) / 'cases.jsonl', CASES_SHA)
    manifest_data, manifest_proof = read(Path(cases_dir) / 'manifest.json')
    manifest = json.loads(manifest_data)
    if manifest['cases']['sha256'] != source['sha256'] or manifest['selected_cases'] != 192 or manifest['per_cell'] != 32 or manifest['seed'] != 42 or manifest['strict_reward_checked'] is not True:
        raise ValueError('Frozen192 case manifest differs')
    if not data.endswith(b'\n'): raise ValueError('Incomplete case input')
    cases = [json.loads(line) for line in data.splitlines()]
    if len(cases) != 192 or Counter((c['dataset'], c['group']) for c in cases) != Counter({(d, g): 32 for d in ('nq', 'hotpotqa') for g in ('correct_correct', 'correct_wrong', 'all5_wrong')}):
        raise ValueError('Frozen six-cell192 coverage differs')
    for case in cases:
        if sha({k: v for k, v in case.items() if k != 'case_semantic_sha256'}) != case['case_semantic_sha256']:
            raise ValueError('Case semantic hash differs')
    selected = select(cases); refs = conditions(selected, core, qa); records = []; skips = Counter(); traces = []
    for item in selected:
        case = item['case']; pair = case['original_pair']['row']
        ordered = [x['row'] for x in sorted(case['original_siblings'], key=lambda x: x['provenance']['line'])]
        reference_sibling = refs[case['case_id']]['reference_sibling']
        candidates = [r for r in ordered if r['sibling'] != reference_sibling and r['reward'] == 0]
        if not candidates: candidates = [r for r in ordered if r['sibling'] != reference_sibling]
        source_rows = [r for r in candidates if r['response'] == pair['student'] and r['policy_spans'] == pair['student_policy_spans'] and r['reward'] == pair['student_reward']]
        if not source_rows: raise ValueError('Missing actual original student row')
        # Identical eligible text has identical distance; native max's first
        # source-order match is its shadow prediction, not unique stored origin.
        row = source_rows[0]
        record = dict(case_id=case['case_id'], question_id=case['question_id'], dataset=case['dataset'], group=case['group'],
                      source_case_sha256=case['case_semantic_sha256'], pilot=item['pilot'], selected_cell_index=item['selected_cell_index'],
                      source_train_row=case['source_train_row'], matching_student_siblings=[r['sibling'] for r in source_rows],
                      native_shadow_first_matching_student_sibling=row['sibling'], stored_pair_origin_uniqueness_not_claimed=True,
                      original_student=reconstruct_original(row), reference_interventions=refs[case['case_id']], actual_hf_states={})
        if trace_root:
            for version in ('source100', 'epoch1'):
                candidates = list(Path(trace_root).glob('worker-*/' + case['case_id'] + '/student-generation-' + version + '.json'))
                if not candidates:
                    skips['missing_existing_hf_trace:' + version] += 1; continue
                if len(candidates) != 1: raise ValueError('Duplicate existing HF trace')
                raw, proof = read(candidates[0]); trace = json.loads(raw)
                if trace['active_adapter'] != version or trace['privileged_reference'] is not False:
                    raise ValueError('Existing HF trace role differs')
                # TraceSearchAgent omits a separate question field. Bind its
                # literal user-message content now; exact template/IDs later.
                user_content = core.prompt(case['source_train_row']['row']['question'])[0]['content']
                if user_content not in trace['prompt_text']:
                    raise ValueError('Existing trace prompt belongs to another question')
                states = reconstruct_hf(trace); states['artifact'] = proof; record['actual_hf_states'][version] = states; traces.append(proof)
        for b in record['original_student']['skipped_boundaries']: skips[b['reason']] += 1
        for condition in record['reference_interventions']['conditions']:
            if condition['status'] == 'skipped': skips[condition['reason']] += 1
        records.append(record)
    # Recheck source stat continuity, including only the small consumed artifacts.
    for proof in [source, manifest_proof] + list(science_proof.values()) + list(helper_proof.values()) + traces:
        if stat(proof['path']) != proof['stat']: raise ValueError('Source changed during preparation')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = output / 'states.private.jsonl'
    with path.open('xb') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
    _, output_proof = read(path)
    summary = dict(schema='m3-cpu-preparation-v1', accepted_cpu_preparation=True, gpu_admitted=False,
                   teacher_effect_evaluated=False, selected_cases=64, pilot_cases=16, seed=42,
                   selection_rule='Same M1 seed42 SHA ordering: first16/cell, first4/cell pilot',
                   donor_rule='Within-cell deterministic cyclic first other-question donor with native-final answer not matching recipient gold',
                   source=dict(cases=source, manifest=manifest_proof, science=science_proof, helper=helper_proof, consumed_existing_traces=traces),
                   skip_counts=dict(skips), states_output=output_proof, created_unix=time.time(),
                   pending='Root publication/admission, actual tokenizer/context budget checks, GPU ordinary-base/distribution/executor null effect. No real continuation has run.')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases-dir', 'runtime', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--trace-root')
    a = p.parse_args(); result = prepare(a.cases_dir, a.runtime, a.output, a.trace_root)
    print(json.dumps(dict(selected=result['selected_cases'], pilot=result['pilot_cases'], gpu_admitted=False)))


if __name__ == '__main__': main()
