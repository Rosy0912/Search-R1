"""Audit the full fresh common8 run; structural completion and accuracy are separate."""
import argparse
import json
import math
import time
from pathlib import Path
from collections import Counter
from reproduction.common8.runtime import budget_rows
from reproduction.common8.worker_evidence import read_console_form
from reproduction.common8.verify_data import digest, verify
from reproduction.common8.validation_identity import expected


def verify_full(base):
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    base=Path(base);run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8' and run['completed_updates']==0 and run['target_updates']==300
    data=verify(base/'data')
    records=[json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==list(range(1,301))
    assert all(r['run_id']==run['run_id'] and all(math.isfinite(v) for v in r['metrics'].values()) for r in records)
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    behavior_gates = {str(r['step']): read_gate_receipt(base, run, r['step'], update_record=r) for r in records}
    verify_write_receipt_links(base, behavior_gates, 1, 300)
    budgets=budget_rows(read_console_form(base,run['run_id']),1,300)
    five=json.loads((base/'logs/five-updates-acceptance.json').read_text())
    assert five['accepted'] is True and five['run_id']==run['run_id']
    snapshots={}
    for step in (5,50,100,150,200,250,300):
        checkpoint=Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
        proof=json.loads((checkpoint/'complete.json').read_text())
        assert proof['run_id']==run['run_id'] and proof['step']==step and proof['finite_all_saved_tensors'] is True
        assert Path(proof['checkpoint']).resolve()==checkpoint.resolve()
        assert all(digest(checkpoint/name)==sha for name,sha in proof['files_sha256'].items())
        snapshots[str(step)]=proof
    final_checkpoint=Path(snapshots['300']['checkpoint'])
    count=validate_saved_checkpoint(final_checkpoint,Path(run['initial_model']),run['started_unix'])
    assert all(digest(Path(run['initial_model'])/name)==sha for name,sha in run['initial_model_sha256'].items())
    # Match every frozen evaluator interval to its genuinely archived calibration.
    archived={}
    for path in (base/'artifacts').glob('round_*-attempt-*/version-manifest.json'):
        live=json.loads(path.read_text())
        if live['run_id']==run['run_id']:
            assert live['version'] not in archived, 'Duplicate archived UUID'
            archived[live['version']]=live
    versions={}
    for round_id in (0,50,100,150,200,250):
        adopted={r['probe_version'] for r in records if round_id<r['step']<=round_id+50}
        assert len(adopted)==1
        live=archived[adopted.pop()];assert live['round']==round_id
        versions[round_id]=live
    for round_id,live in versions.items():
        encoder=Path(live['encoder'])
        if round_id==0:
            expected_encoder={n:h for n,h in run['initial_model_sha256'].items()
                              if n not in ('vocab.json','merges.txt','generation_config.json')}
            assert encoder.resolve()==Path(run['initial_model']).resolve()
        else:
            snapshot=snapshots[str(round_id)]
            expected_encoder=dict(snapshot['files_sha256'])
            expected_encoder.update({n:run['initial_model_sha256'][n] for n in ('tokenizer.json','tokenizer_config.json')})
            assert encoder.resolve()==Path(snapshot['checkpoint']).resolve()
        assert live['encoder_sha256']==expected_encoder
        assert all(digest(encoder/name)==sha for name,sha in live['encoder_sha256'].items())
        heads=Path(live['commit']).parent;dump=heads.parent/'dumps'
        assert all(digest(heads/name)==sha for name,sha in live['head_sha256'].items())
        assert digest(heads/'calibration-report.json')==live['calibration_report_sha256']
        assert {p.name:digest(p) for p in dump.iterdir() if p.is_file()}==live['dump_sha256']
        assert json.loads((heads/'calibration-report.json').read_text())==live['calibration_report']
        assert json.loads((dump/'calibration-label-proof.json').read_text())==live['label_proof']
        assert set(live['label_proof']['question_ids'])==set(run['calibration_question_ids'])
        report=live['calibration_report']
        assert report['require_all_cv_folds'] is True and report['configured_cv_folds']==2
        assert all(math.isfinite(report[key+'_cv_auc']) and 0<=report[key+'_cv_auc']<=1 for key in ('commit','boundary'))
    for r in records:
        round_id=((r['step']-1)//50)*50
        assert r['probe_version']==versions[round_id]['version']
    evaluation=json.loads((base/'logs/final-identity.json').read_text())
    assert evaluation['run_id']==run['run_id'] and evaluation['step']==300 and evaluation['final'] is True
    canonical,frame,ids=expected(base,'final')
    assert evaluation['validation_sha256']==canonical['sha256']
    assert evaluation['question_ids']==ids and evaluation['data_sources']==frame['data_source'].tolist()
    scores=evaluation['em'];assert len(scores)==51713 and all(x in (0,1) for x in scores)
    assert evaluation['counts']==canonical['source_counts']
    assert evaluation['correct']==sum(scores) and evaluation['weighted_em']==sum(scores)/len(scores)
    counts=Counter(evaluation['data_sources']);correct={s:0 for s in counts}
    for source,score in zip(evaluation['data_sources'],scores):correct[source]+=score
    proof=dict(run_id=run['run_id'],structural_accepted=True,actual_outer_updates=300,
        per_rank_optimizer_writes=6000,final_saved_tensor_count=count,
        shared_data=data,checkpoint_proofs=snapshots,final_counts=dict(counts),
        behavior_probability_gates=behavior_gates,
        final_correct=correct,final_weighted_em=evaluation['weighted_em'],
        semantics='full training/data/numerical structure verified; accuracy is the measured result, not automatically reproduction success',
        checked_unix=time.time())
    with (base/'logs/full-run-acceptance.json').open('x') as f:json.dump(proof,f,indent=2)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(verify_full(a.base)))
