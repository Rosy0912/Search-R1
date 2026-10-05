"""Verify interrupted G1 logical1..300 across separately bound run segments.

Does not equate weights-only recovery with uninterrupted optimizer/RNG state.
Old failed calibration and its original source remain immutable evidence.
"""
import json
import math
import os
import re
import time
from pathlib import Path
from reproduction.a800.prepare_run import digest
from reproduction.a800.owned_processes import publish
from reproduction.a800.verify_full import actual_updates,final_validation
from verl.utils.checkpoint_integrity import validate_saved_checkpoint


def segment_rows(text,run,first,last):
    assert text.splitlines()[0]=='SEARCH_G1_RUN '+run['run_id']
    return actual_updates(text,first,last)


def bind_segment_versions(rows,run,initial_round,adopted,expected_rounds):
    assert initial_round in (0,200,250)
    assert len(adopted)==len(expected_rounds)
    assert [int(v.split(':')[1]) for v in adopted]==expected_rounds
    assert all(v.startswith(run['run_id']+':') for v in adopted)
    initial=run['initial_probe_version']
    assert initial.startswith(run['run_id']+':'+str(initial_round)+':')
    versions={initial_round:initial,**{int(v.split(':')[1]):v for v in adopted}}
    for row in rows:
        boundary=((row['step']-1)//50)*50
        assert row['probe_version']==versions[boundary], 'Wrong evaluator identity at '+str(row['step'])


def fixed64_identity(evidence,run,validation):
    assert evidence['run_id']==run['run_id'] and evidence['step']==300 and evidence['final'] is True
    assert evidence['validation_sha256']==run['input_sha256']['validation-64.parquet']
    ids=evidence['question_ids'];sources=evidence['data_sources'];scores=evidence['em']
    assert ids==run['validation_question_ids'] and len(set(ids))==64
    assert len(ids)==len(sources)==len(scores)==64
    assert all(type(x) in (int,float) and math.isfinite(x) and x in (0.,1.) for x in scores)
    counts={s:sources.count(s) for s in set(sources)}
    assert counts==run['validation_source_counts']==validation['counts']
    for s in counts:
        mean=sum(score for source,score in zip(sources,scores) if source==s)/counts[s]
        assert abs(mean-validation['source_em'][s])<1e-12, 'Row EM and final metrics disagree'
    assert evidence['correct']==sum(scores) and abs(evidence['weighted_em']-sum(scores)/64)<1e-12
    return dict(actual_unique_questions=64,correct=sum(scores),weighted_em=sum(scores)/64,
        per_source_correct={s:sum(x for source,x in zip(sources,scores) if source==s) for s in counts})


def verify_segment_inputs(base,run):
    for name,value in run['input_sha256'].items():assert digest(base/'data'/name)==value
    for name,value in run['source_sha256'].items():assert digest(base/'repo'/name)==value


def verify_five(base,run,first):
    initial=json.loads((base/'logs/initial-probes-accepted.json').read_text())
    five=json.loads((base/'logs/five-updates-acceptance.json').read_text())
    assert initial['accepted'] is True and five['accepted'] is True
    assert initial['run_id']==five['run_id']==run['run_id']
    assert initial['version']==run['initial_probe_version']
    assert [r['step'] for r in five['actual_updates']]==list(range(first,first+5))
    assert all(r['probe_version']==run['initial_probe_version'] for r in five['actual_updates'])
    assert five['changed_parameters'] and five['validation_counts']==run['validation_source_counts']
    return digest(base/'logs/five-updates-acceptance.json')


def saved_boundary(run,step,full=False):
    folder=Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
    proof=json.loads((folder/'complete.json').read_text())
    assert proof['run_id']==run['run_id'] and proof['step']==step and proof['finite_all_saved_tensors'] is True
    for name,value in proof['files_sha256'].items():assert digest(folder/name)==value
    if full:
        count=validate_saved_checkpoint(folder,Path(run['initial_model']),run['started_unix'])
        assert count==proof['validated_tensor_count']
    return proof


def audit_chain(base,run,last):
    """Audit immutable run segments back to the original source, without requiring
    a failed parent's terminal300 acceptance. Every segment has its own five.
    """
    chain=[];seen=set();end=last
    while True:
        assert run['run_id'] not in seen, 'Cyclic/repeated ancestry'
        seen.add(run['run_id'])
        start=run.get('completed_updates',0)
        assert start in (0,200,250) and start<end<=300
        assert run['target_updates']==300 and run['acceptance_update']==start+5
        verify_segment_inputs(base,run)
        text=Path(run['train_log']).read_text(errors='replace')
        rows=segment_rows(text,run,start+1,end)
        adopted=re.findall(r'SEARCH_G1_REFIT_ADOPTED=([^\s]+)',text)
        bind_segment_versions(rows,run,start,adopted,list(range(start+50,end,50)))
        five=verify_five(base,run,start+1)
        initial=json.loads((base/'logs/initial-probes-accepted.json').read_text())
        if start:
            archived_path=Path(initial['evaluator_manifest'])
            assert digest(archived_path)==initial['live_manifest_sha256']
            archived=json.loads(archived_path.read_text())
            from reproduction.a800.probe_admission import calibrated_version
            assert calibrated_version(run,archived)==run['initial_probe_version']
            from reproduction.a800.recovery_binding import verify_interventions
            assert archived['recovery']['dump_sha256']==run['recovery_dump_sha256']
            verify_interventions(Path(run['recovery_dumps']),run['recovery_dump_sha256'])
            assert set(archived['label_proof']['question_ids'])==set(json.loads((base/'data/audit.json').read_text())['calibration_question_ids'])
            if start==250:
                from reproduction.a800.recovery_binding import verify_repair_calibration
                assert digest(Path(run['calibration_acceptance']))==run['calibration_acceptance_sha256']
                verify_repair_calibration(Path(run['recovery_dumps']),Path(run['calibration_acceptance']))
        saves=sorted(set([start+5]+list(range(start+50,end+1,50))))
        proofs={str(step):saved_boundary(run,step,full=step==300) for step in saves}
        chain.append(dict(base=str(base),run_id=run['run_id'],first=start+1,last=end,rows=rows,
            five_sha256=five,adopted=adopted,saves=proofs,source_commit=run['source_commit']))
        if not start:break
        binding=run['parent'];parent=Path(binding['parent_base'])
        old=json.loads((parent/'logs/run-manifest.json').read_text())
        assert old['run_id']==binding['parent_run_id']
        assert digest(parent/'logs/run-manifest.json')==binding['parent_manifest_sha256']
        assert digest(Path(old['train_log']))==binding['parent_train_log_sha256']
        assert run['input_sha256']==old['input_sha256'] and run['initial_model_sha256']==old['initial_model_sha256']
        assert run['validation_question_ids']==old['validation_question_ids']
        checkpoint=Path(old['checkpoint_dir'])/'actor'/f'global_step_{start}'
        assert Path(run['actor_checkpoint'])==checkpoint
        assert digest(checkpoint/'complete.json')==binding['parent_checkpoint_marker_sha256']
        assert run['actor_checkpoint_sha256']==binding['parent_checkpoint_sha256']
        for name,value in run['actor_checkpoint_sha256'].items():assert digest(checkpoint/name)==value
        base,run,end=parent,old,start
    chain.reverse()
    assert chain[0]['first']==1 and chain[-1]['last']==last
    assert all(a['last']+1==b['first'] for a,b in zip(chain,chain[1:]))
    return chain


def verify(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run.get('completed_updates') in (200,250)
    chain=audit_chain(base,run,300)
    text=Path(run['train_log']).read_text(errors='replace')
    validation=final_validation(text,run['validation_source_counts'])
    final_path=base/'logs/final-fixed64-identity.json'
    rows=fixed64_identity(json.loads(final_path.read_text()),run,validation)
    proofs={k:v for part in chain for k,v in part['saves'].items()}
    result=dict(run_id=run['run_id'],accepted=True,actual_update_count=sum(len(p['rows']) for p in chain),
        actual_segments=[[p['first'],p['last']] for p in chain],ancestry=chain,
        continuation_semantics=run['continuation_semantics'],saved_steps=sorted(map(int,proofs)),
        fresh_final_checkpoint=proofs['300'],fixed64_final_validation=validation,actual_fixed64_identity=rows,
        final_identity_sha256=digest(final_path),five_update_acceptance_sha256={p['run_id']:p['five_sha256'] for p in chain},
        adopted_refit_versions={p['run_id']:p['adopted'] for p in chain},
        training_console_precision='three decimal places; final JSON EM uses actual binary rows',checked_unix=time.time())
    assert result['actual_update_count']==300
    publish(base/'logs/full-run-acceptance.json',result)
    print('SEARCH_G1_INTERRUPTED_FULL300_ACCEPTED='+run['run_id'],flush=True)
    return result


if __name__=='__main__':verify(Path(os.environ['SEARCH_G1_BASE']))
