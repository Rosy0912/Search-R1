"""Reuse the accepted common7B evaluator without modifying its scientific code."""
import argparse
import json
from pathlib import Path
from types import FunctionType

from reproduction.official_eval_v2 import driver, port_recovery
from reproduction.official_eval_v2.protocol import PROFILE, SCIENCE, digest, sha, validate_rows

JOB = 'zhangjunming-7b-cgrts50-v1'
TRAIN_JOB = '139134'
TRAIN_SOURCE = '92942a32161071b85cc70906050143ed58244eb1'
TRAIN_PLAN = 'b71e43e5f979bed427dc26bf0953524cf151b37bc8f9f36a8ed4cade5c44cdf5'
CORE = ('reproduction/official_eval_v2/driver.py',
        'reproduction/official_eval_v2/protocol.py')
STARTUP_FILE = 'reproduction/official_eval_v2/port_recovery.py'
STARTUP_SHA = 'ec586e53ca75b9f2e21cd7f68001445234768e81acd967fb7dbe4238d63776d5'
MUTABLE = {'kind', 'root', 'job', 'source', 'source_commit', 'source_files', 'model',
           'model_revision', 'model_files', 'model_identity', 'milestone_checkpoint', 'repeat_count'}


def validate_proof(proof):
    assert proof['job_id'] == TRAIN_JOB and proof['source'] == TRAIN_SOURCE
    assert proof['plan_sha256'] == TRAIN_PLAN and proof['completed_native_step'] == 50
    assert proof['local_small_evidence_sha_verified']
    assert proof['external_cpu_only'] and proof['no_cuda_initialization']
    assert proof['process_rng_unchanged'] and proof['no_training_mutation']
    workers = proof['workers']
    assert len(workers) == 8 and {r['rank'] for r in workers} == set(range(8))
    assert len(proof['physical_rank_files']) == 8
    for r in workers:
        assert r['scheduler_last_epoch'] == 50 and r['optimizer_states'] > 0
        assert r['actual_adam_min'] == r['actual_adam_max'] > 0
        assert r['all_complete_state_finite'] and r['stored_cpu_numpy_python_and_cuda_rng_verified']
    assert proof['sampler']['rows'] == 169028 and proof['sampler']['cursor'] == 50 * 512
    assert len(proof['hf_identity']['files']) == 16


def validate_contract(plan, reference, proof):
    validate_proof(proof)
    assert plan['profile'] == PROFILE and plan['science'] == SCIENCE
    assert plan['kind'] == 'cgrts50' and plan['job'] == JOB
    assert plan['repeat'] == 1 and plan['repeat_count'] == 1
    assert Path(plan['root']).name == 'cgrts50-r1'
    assert (plan['gpus'], plan['cpus'], plan['memory_gib']) == (8, 64, 600)
    assert {k: v for k, v in plan.items() if k not in MUTABLE} == {
        k: v for k, v in reference.items() if k not in MUTABLE}
    for name in CORE:
        assert plan['source_files'][name] == reference['source_files'][name]
    assert plan['source_files'][STARTUP_FILE] == STARTUP_SHA
    target = plan['milestone_checkpoint']
    assert target['step'] == 50 and target['training_job'] == TRAIN_JOB
    assert target['reference_model_label_only'] is False and target['not_final200']
    assert target['accepted_scientific_repeat_count_for_milestone'] == 1
    assert plan['model_files'] == proof['hf_identity']['files']
    assert plan['model_identity'] == digest(dict(training_job=TRAIN_JOB,
        training_source=TRAIN_SOURCE, completed_step=50, hf_identity=proof['hf_identity']))
    assert plan['model'] == target['training_root'] + '/checkpoints/actor/global_step_50'


def verify(plan):
    target = plan['milestone_checkpoint']
    assert sha(target['reference_plan']) == target['reference_plan_sha256']
    assert sha(target['native_acceptance']) == target['native_acceptance_sha256']
    reference = json.loads(Path(target['reference_plan']).read_text())
    proof = json.loads(Path(target['native_acceptance']).read_text())
    validate_contract(plan, reference, proof)
    for name, expected in plan['source_files'].items():
        assert sha(Path(plan['source']) / name) == expected, name
    for name, expected in plan['model_files'].items():
        assert sha(Path(plan['model']) / name) == expected, name
    assert sha(plan['data']) == plan['data_sha256']
    rows = [json.loads(line) for line in Path(plan['data']).open()]
    validate_rows(rows)
    assert digest([{k: r[k] for k in ('final_index', 'question_id', 'source', 'gold_answers')}
                   for r in rows]) == plan['id_gold_identity']
    assert sha(plan['retriever_receipt']) == plan['retriever_receipt_sha256']
    return rows


def run(planfile):
    # A private globals dictionary preserves the frozen evaluator function bytecode
    # and leaves every imported module/global used by other entry points unchanged.
    namespace = dict(driver.run.__globals__, verify=verify,
                     worker=port_recovery.startup_worker)
    return FunctionType(driver.run.__code__, namespace)(planfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    run(parser.parse_args().plan)
