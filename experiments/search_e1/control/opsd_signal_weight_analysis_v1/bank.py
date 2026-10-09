"""Freeze the one common-zero successful reference bank before post endpoints."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time
try:
    from .common import (require, read_json, read_rows, semantic_sha, exclusive_json,
        remote_output, bind_training, checked_endpoint, cpu_only, VARIANTS)
except ImportError:
    from common import (require, read_json, read_rows, semantic_sha, exclusive_json,
        remote_output, bind_training, checked_endpoint, cpu_only, VARIANTS)


def build_bank(data, records, trace_reader, protocol_checker):
    """One row per fixed question; never drop source-zero failures from EM."""
    require(len(data) == len(records) == 512 and len({q['question_id'] for q in data}) == 512,
        'Bank requires all512 fixed validation questions')
    index = {}
    for row in records:
        i = row['validation_index']
        require(type(i) is int and i in range(512) and i not in index and row['primary'] is True
            and row['version'] == 'source100' and row['completed_outer'] == 0, 'Bank cannot select another endpoint')
        q = data[i]
        require(row['question_id'] == q['question_id'] and row['row_semantic_sha256'] == semantic_sha(q),
            'Bank validation identity differs')
        trace = trace_reader(row['trace']); result = trace['native_result']
        require(trace['validation_index'] == i and trace['question_id'] == q['question_id']
            and trace['question'] == q['question'] and trace['answers'] == q['answers']
            and semantic_sha(result) == row['native_result_sha256'], 'Common-zero native trace changed')
        decoded = protocol_checker(result, q['answers'])
        require(type(decoded['protocol_em']) is int and decoded['protocol_em'] in (0, 1)
            and decoded == trace['protocol'] and decoded['protocol_em'] == row['protocol_em'],
            'Frozen authored-final scorer disagrees with common-zero result')
        success = decoded['protocol_em'] == 1
        index[i] = dict(validation_index=i, question_id=q['question_id'], dataset=q['data_source'],
            validation_row_semantic_sha256=semantic_sha(q), source_zero_em=decoded['protocol_em'],
            source_zero_trace=row['trace'], source_zero_native_result_sha256=row['native_result_sha256'],
            available=success, missing_reason=None if success else 'source_zero_not_strict_correct',
            reference_response=result['response'] if success else None,
            reference_response_sha256=hashlib.sha256(result['response'].encode()).hexdigest() if success else None,
            reference_policy_spans=result['policy_spans'] if success else None,
            reference_gold_used_for_eligibility=True)
    require(set(index) == set(range(512)), 'Bank fixed512 coverage incomplete')
    return dict(schema='opsd-signal-fixed-common-zero-bank-v1', accepted=True,
        entries=[index[i] for i in range(512)], questions=512,
        available=sum(x['available'] for x in index.values()),
        selection='This study common source100 greedy authored-final strict correct responses only',
        gold_used_for_reference_eligibility=True, reference_generation_added=0,
        all_endpoints_share_same_frozen_reference=True,
        original_wrong_to_new_correct_teacher_signals_not_covered=True,
        baseline_bank_subset_em_constant_one=True, predictive_or_causal_claim=False)


def run(request_ref, output):
    cpu_only(); request, rp = read_json(request_ref); destination = remote_output(output)
    require(request.get('schema') == 'opsd-signal-bank-freeze-request-v1', 'Wrong bank freeze request')
    configs = {}; refs = {}
    for ref in request['configs']:
        cfg, proof = read_json(ref); cfg['_config_sha256'] = proof['sha256']
        require(cfg['variant'] not in configs, 'Duplicate bank config')
        configs[cfg['variant']] = cfg; refs[cfg['variant']] = proof
    require(set(configs) == set(VARIANTS), 'Freeze must bind all four group configurations')
    cfg = configs['G_POS']; modules = bind_training(cfg)
    for other in configs.values():
        require(other['study_id'] == cfg['study_id'] and other['source_model_manifest'] == cfg['source_model_manifest']
            and other['runtime_manifest'] == cfg['runtime_manifest'] and other['validation_identity'] == cfg['validation_identity']
            and other['common_initial_adapter_manifest'] == cfg['common_initial_adapter_manifest'],
            'Bank groups do not share one source/runtime/data/common-zero adapter')
        for t in (15, 25):
            require(not (Path(other['paths']['output_root']) / 'evaluations' / ('outer-%02d' % t)).exists(),
                'Bank freeze is late: a group post endpoint already exists; preserve inputs and report chronology')
    verified = modules['evaluation'].verify_common_zero(cfg, request['common_zero'])
    zero, zp, records, data = checked_endpoint(cfg, verified['evaluation_receipt'], modules)
    core = importlib.import_module('search_e1.core'); qa = importlib.import_module('search_r1.qa_protocol')
    bank = build_bank(data, records, lambda ref: read_json(ref)[0],
        lambda result, answers: modules['training_metrics'].protocol(core, qa, result, answers))
    bank.update(study_id=cfg['study_id'], request=rp, common_zero_wrapper=request['common_zero'],
        common_zero_evaluation=zp, configs=refs, source_model_manifest=cfg['source_model_manifest'],
        runtime_manifest=cfg['runtime_manifest'], validation_identity=cfg['validation_identity'],
        common_initial_adapter=cfg['common_initial_adapter_manifest'], protocol=zero['input_identity']['protocol'],
        frozen_before_all_group_15_25_outputs=True, frozen_unix=time.time(), parent_torch_imported=False)
    return exclusive_json(destination, bank)


def main():
    p = argparse.ArgumentParser()
    for name in ('request', 'request-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(); proof = run(dict(path=a.request, sha256=a.request_sha256), a.output)
    print(json.dumps(dict(accepted=True, bank=proof)))


if __name__ == '__main__':
    main()
