"""Real paired dev512 reporting at official0 and fresh5; accuracy is measured.

This does not add an accuracy/no-drop acceptance rule. The existing stage64
checks remain separate and retain their original canonical occurrences.
"""
import json,math,os,time
from pathlib import Path
from reproduction.common8.formal_contract import SAFETY_MODE,assert_formal_run
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.verify_data import digest

ROLE='formal_inline_development_evaluation'
PATHS={0:'logs/formal-inline-development-before.json',5:'logs/formal-inline-development-after.json'}


def score_counts(rows):
    correct={source:sum(row['em'] for row in rows if row['data_source']==source)
             for source in ('nq','hotpotqa')}
    return dict(correct_count=sum(row['em'] for row in rows),f1_sum=math.fsum(row['f1'] for row in rows),
                source_correct=correct,source_em={source:value/256 for source,value in correct.items()})


def active_run():
    if os.environ.get('SEARCH_G1_PROFILE')!='common8':return None
    base=Path(os.environ['SEARCH_G1_BASE']).resolve()
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run.get('policy_safety_mode')!=SAFETY_MODE:return None
    assert_formal_run(run)
    return base,run


def configure_development(trainer):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn
    from reproduction.common8 import validation_identity as validation
    import copy
    base,run=active_run();assert int(trainer.global_steps) in (0,5)
    record,frame,_=validation.expected(base,'validation')
    dataset=copy.copy(trainer.val_dataset)
    dataset.dataframe=validation.bind_validation_dataframe(frame,base,'validation')
    trainer.val_dataloader=DataLoader(dataset,batch_size=64,shuffle=False,drop_last=False,collate_fn=collate_fn)
    validation._ACTIVE_SPLIT='validation';validation._ACTIVE_SOURCE_COUNTS=dict(record['source_counts'])
    print('SEARCH_G1_EVALUATION_SPLIT=validation',flush=True)


def record_development(step,rows,original_rewards):
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.formal_inline_safety import verify_plan
    assert type(step) is int and step in PATHS
    base,run=active_run();record,frame,ids=expected(base,'validation')
    result=aggregate(rows,ids,frame['data_source'].tolist())
    result.update(score_counts(rows))
    assert [row['em'] for row in rows]==list(original_rewards), 'Measured strict answers must match actual EM reward'
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text())
    _,plan_sha=verify_plan(base,artifact,run)
    result.update(schema_version=1,role=ROLE,run_id=run['run_id'],policy_safety_mode=SAFETY_MODE,step=step,
        plan_sha256=plan_sha,validation_sha256=record['sha256'],source_map_sha256=document_sha256(run['source_sha256']),
        checked_unix=time.time(),semantics='paired real dev512 EM/F1/actions; accuracy change is reported without a zero-drop gate')
    write_once(base/PATHS[step],result)
    return result


def evaluate_development(trainer):
    """Same live trainer and weights; preserve normal stage/dev loader selection."""
    from reproduction.common8 import validation_identity as validation
    assert active_run() is not None and int(trainer.global_steps) in PATHS
    original_loader=trainer.val_dataloader
    original_split=validation._ACTIVE_SPLIT;original_sources=validation._ACTIVE_SOURCE_COUNTS
    try:
        return trainer._validate(inline_development=True)
    finally:
        trainer.val_dataloader=original_loader
        validation._ACTIVE_SPLIT=original_split;validation._ACTIVE_SOURCE_COUNTS=original_sources


def verify_pair(base,run):
    """Rebuild ordered scores/actions from actual scalar rows; do not gate gains."""
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    from reproduction.common8.formal_inline_safety import verify_plan
    base=Path(base).resolve();assert_formal_run(run)
    artifact=json.loads((base/'logs/policy-safety-calibration.json').read_text());_,plan_sha=verify_plan(base,artifact,run)
    record,frame,ids=expected(base,'validation');proofs={};hashes={}
    for step,path in PATHS.items():
        target=base/path
        assert target.is_file() and not target.is_symlink() and target.resolve()==target
        proof=json.loads(target.read_text());measured=aggregate(proof['rows'],ids,frame['data_source'].tolist())
        measured.update(score_counts(proof['rows']))
        assert proof['schema_version']==1 and proof['role']==ROLE and proof['run_id']==run['run_id']
        assert proof['policy_safety_mode']==SAFETY_MODE and proof['step']==step and proof['plan_sha256']==plan_sha
        assert proof['source_map_sha256']==document_sha256(run['source_sha256']) and proof['validation_sha256']==record['sha256']
        assert all(proof[name]==value for name,value in measured.items() if name not in ('role','semantics'))
        proofs[step]=proof;hashes[str(step)]=digest(target)
    before,after=proofs[0],proofs[5]
    assert before['question_ids']==after['question_ids']==ids and before['data_sources']==after['data_sources']==frame['data_source'].tolist()
    return dict(schema_version=1,role='formal_inline_paired_development_report',run_id=run['run_id'],plan_sha256=plan_sha,
        before=before,after=after,file_sha256=hashes,
        changes={name:after[name]-before[name] for name in ('weighted_em','weighted_f1','valid_action_rate',
            'executed_search_question_rate','executed_search_calls_per_question')},
        accuracy_health_gate=False,semantics='Actual paired512 scores/actions, including any decline; no invented zero-drop requirement')
