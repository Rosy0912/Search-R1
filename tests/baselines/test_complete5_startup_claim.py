"""Startup claim CPU regressions; never a real job authorization."""
import json
import pytest
from search_r1.baselines.aligned_startup import claim_new_run
from search_r1.baselines import aligned_complete5_resume as resume


def fixture_manifest(run):
    ranks=[dict(rank=i,baseline_aligned=True,actual_outer=5,attempt_id=5,status='COMPLETE',optimizer_steps=20,
        expected_optimizer_steps=20,optimizer_opportunities_total=20,processed_opportunities=20,skipped_opportunities=0,
        unprocessed_opportunities=0,skipped_indices=[],skipped_reasons=[],postabort=False,update_complete=True,attempt_complete=True,
        outer_commit_phase='STAGE_COMMITTED',accepted_complete_outer=True)for i in range(8)]
    return dict(old_run_root=str(resume.OLD_RUN),old_source_commit=resume.OLD_SOURCE,target_run_root=str(run),target_job_id='9999',target_source_commit='b'*40,ranks=[dict(rank=i)for i in range(8)],rank_write_ledger=ranks)


def test_complete5_claim_has_resume_scope_and_cannot_be_reclaimed(tmp_path):
    run=tmp_path/'runs/continued-9999'
    assert claim_new_run(run,'b'*40,complete5_resume=fixture_manifest(run))==run
    record=json.loads((run/'.aligned-startup-claim.json').read_text())
    assert record['scope']=='same-experiment-complete5-verify-only-resume'
    assert record['source_commit']=='b'*40 and type(record['launcher_pid'])is int
    with pytest.raises(ValueError,match='prior experiment'):claim_new_run(run,'b'*40,complete5_resume=fixture_manifest(run))


def test_pending_and_complete_modes_cannot_share_a_claim(tmp_path):
    run=tmp_path/'runs/continued-9999'
    with pytest.raises(ValueError,match='Cannot mix'):claim_new_run(run,'b'*40,partial_resume={},complete5_resume=fixture_manifest(run))
    assert not (run/'.aligned-startup-claim.json').exists()


@pytest.mark.parametrize('kind',['source','run','old_job'])
def test_complete5_claim_rejects_wrong_source_run_or_original_job_before_claim(tmp_path,kind):
    run=tmp_path/'runs/continued-9999';manifest=fixture_manifest(run)
    if kind=='source':manifest['target_source_commit']='c'*40
    elif kind=='run':manifest['target_run_root']=str(tmp_path/'runs/foreign-9999')
    else:manifest['target_job_id']='3357'
    with pytest.raises(ValueError):claim_new_run(run,'b'*40,complete5_resume=manifest)
    assert not (run/'.aligned-startup-claim.json').exists()
