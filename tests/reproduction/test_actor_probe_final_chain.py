"""Counter composition and mandatory-reader orchestration, not native acceptance."""
import copy
import unittest
from unittest.mock import patch

from reproduction.common8.verify_actor_probe_chain import compose
from reproduction.common8.actor_probe_fault_review import verify_documents, ROLE
from reproduction.common8.policy_calibration_plan import document_sha256
from search_g1.actor_probe_recovery import identity, binding_digest
from tests.reproduction.test_actor_probe_calibration_acceptance import retained_chain


def fixture():
    entries,segments=retained_chain()
    for segment in segments:
        steps=(5,50) if segment['first']==1 else (100,150,200)
        segment['checkpoint_proofs']={str(s):dict(receipt={'fixture':s},finite_tensor_count=435) for s in steps}
        segment['full_states']={str(s):{'fixture':'full-state-'+str(s)} for s in steps}
    calibration=dict(original_calibration_provenance_verified=True,retained_version_intervals_verified=True)
    five=dict(accepted=True,identity=identity(entries[0]['run']))
    faults={entries[1]['run']['run_id']:dict(accepted=True,actual_failure_and_candidate_review_verified=True)}
    predecessors=[dict(actual_predecessor_completion_reverified=True) for _ in entries]
    counts=dict(nq=3610,hotpotqa=7000,triviaqa=10000,popqa=10000,webqa=8000,musique=10000,bamboogle=3103)
    final=dict(run_id=entries[-1]['run']['run_id'],step=200,question_count=51713,counts=counts,
        correct=12345,weighted_em=12345/51713,weighted_f1=.3,source_em={},source_f1={})
    return entries,segments,calibration,five,faults,predecessors,final


class FinalChainCompositionTests(unittest.TestCase):
    def test_retained200_and_lost_tail_keep_separate_counters_and_do_not_claim_cleanup(self):
        args=fixture();before=copy.deepcopy(args);proof=compose(*args)
        self.assertEqual(args,before)
        self.assertEqual(proof['actual_outer_updates'],200)
        self.assertEqual(proof['per_rank_optimizer_opportunities'],4000)
        self.assertEqual(set(proof['per_rank_actual_optimizer_writes'].values()),{3993})
        self.assertEqual(proof['retained_skipped_opportunities'],7)
        self.assertEqual(proof['retained_counter_chain']['discarded_by_namespace'][0]['successful_adam'],60)
        self.assertEqual(proof['retained_counter_chain']['discarded_by_namespace'][0]['complete_outers'],[51,52,53])
        self.assertFalse(proof['owned_cleanup_verified']);self.assertFalse(proof['goal_complete'])

    def test_missing_milestone_review_or_original_calibration_is_not_complete(self):
        for kind in ('milestone','fault','calibration','root_five','final_identity','nq'):
            args=list(fixture());entries,segments,calibration,five,faults,parents,final=args
            if kind=='milestone':segments[1]['checkpoint_proofs'].pop('100')
            elif kind=='fault':faults.clear()
            elif kind=='calibration':calibration['original_calibration_provenance_verified']=False
            elif kind=='root_five':five['identity']['source_commit']='0'*40
            elif kind=='final_identity':final['run_id']='0'*32
            else:final['counts']['nq']+=1
            with self.subTest(kind=kind),self.assertRaises(AssertionError):compose(*args)

    def test_all_skip_retry_opportunities_are_not_hidden_by_200_complete_outers(self):
        from tests.reproduction.test_actor_probe_chain import segment
        entries,segments,calibration,five,faults,parents,final=fixture()
        first=segment(entries[0]['run']['run_id'],0,53,50,zero_guard_at=3)
        last=segment(entries[1]['run']['run_id'],50,200,200,first['checkpoint_writes'],first['checkpoint_attempt_index'])
        segments[0]['counter_segment']=first;segments[1]['counter_segment']=last
        proof=compose(entries,segments,calibration,five,faults,parents,final)
        self.assertEqual(proof['per_rank_optimizer_opportunities'],4020)
        self.assertEqual(proof['retained_skipped_opportunities'],27)
        self.assertEqual(proof['actual_outer_updates'],200)

    def test_review_is_bound_to_actual_parent_bytes_candidate_source_and_deployment(self):
        from tests.reproduction.test_actor_probe_recovery import documents
        parent,child,binding,_=documents();child['source_sha256']={'fixture.py':'1'*64}
        evidence=dict(retained_through=5,fixture='actual scalar evidence stand-in')
        review=dict(role=ROLE,accepted=True,identity=identity(child),parent_identity=identity(parent),
            recovery_binding_sha256=binding_digest(binding),retained_through=5,
            parent_evidence_sha256=document_sha256(evidence),reviewed_source_commit=child['source_commit'],
            reviewed_source_map_sha256=document_sha256(child['source_sha256']),
            independent_review_initialized_without_parent_context=True,actual_failure_reviewed=True,
            actual_candidate_deployment_reviewed=True,original_failure_source_and_logs_preserved=True,
            no_retained_failure_ignored=True,resolution_kind='full_state_restore_after_external_exit',
            reviewer_task='/root/independent_scalar_fixture',resolution_summary='Synthetic schema mechanics only')
        self.assertEqual(verify_documents(review,child,parent,evidence),review)
        for key,value in [('parent_evidence_sha256','0'*64),('reviewed_source_commit','0'*40),
            ('actual_candidate_deployment_reviewed',False),('independent_review_initialized_without_parent_context',False)]:
            wrong=copy.deepcopy(review);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):verify_documents(wrong,child,parent,evidence)


if __name__=='__main__':unittest.main()
