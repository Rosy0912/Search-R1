import copy
import math
import unittest

from reproduction.common8.final_evaluation_acceptance import verify_final_rows


def fixture():
    counts=dict(nq=3610,hotpotqa=8000,triviaqa=8000,popqa=8000,musique=8000,twowikimultihopqa=8000,bamboogle=8103)
    sources=[source for source,count in counts.items() for _ in range(count)]
    ids=['canonical-question-'+str(index//2) for index in range(51713)]
    em=[float(index%3==0) for index in range(51713)];f1=[1. if value else .25 for value in em]
    canonical=dict(rows=51713,source_counts=counts,sha256='a'*64);run=dict(run_id='b'*32)
    evaluation=dict(run_id=run['run_id'],step=200,split='final',final=True,validation_sha256=canonical['sha256'],
        question_ids=ids,data_sources=sources,em=em,f1=f1,counts=counts,correct=sum(em),weighted_em=sum(em)/51713,
        weighted_f1=math.fsum(f1)/51713)
    return evaluation,run,canonical,ids,sources


class FinalEvaluationTests(unittest.TestCase):
    def test_full_ordered_occurrences_and_seven_aggregates_rebuilt_in_memory(self):
        evaluation,run,canonical,ids,sources=fixture()
        proof=verify_final_rows(evaluation,run,canonical,ids,sources)
        self.assertEqual(proof['question_count'],51713);self.assertEqual(proof['counts']['nq'],3610)
        self.assertEqual(sum(proof['source_correct'].values()),proof['correct'])
        self.assertEqual(proof['correct'],evaluation['correct'])
        self.assertEqual(proof['source_f1']['nq'],math.fsum(evaluation['f1'][:3610])/3610)
        self.assertLess(len(set(ids)),len(ids))

    def test_wrong_identity_order_binary_scores_f1_and_aggregates_rejected(self):
        original,run,canonical,ids,sources=fixture()
        for mode in ('order','dedup','bool_em','nan_f1','em_f1','correct','weighted_f1','source_counts','split','float_step'):
            with self.subTest(mode=mode):
                evaluation=copy.deepcopy(original)
                if mode=='order':evaluation['question_ids'][1],evaluation['question_ids'][2]=evaluation['question_ids'][2],evaluation['question_ids'][1]
                if mode=='dedup':evaluation['question_ids']=list(dict.fromkeys(evaluation['question_ids']))
                if mode=='bool_em':evaluation['em'][0]=True
                if mode=='nan_f1':evaluation['f1'][0]=float('nan')
                if mode=='em_f1':evaluation['f1'][0]=.5
                if mode=='correct':evaluation['correct']+=1
                if mode=='weighted_f1':evaluation['weighted_f1']+=.001
                if mode=='source_counts':evaluation['counts']['nq']+=1
                if mode=='split':evaluation['split']='validation'
                if mode=='float_step':evaluation['step']=200.
                with self.assertRaises(AssertionError):verify_final_rows(evaluation,run,canonical,ids,sources)


if __name__=='__main__':unittest.main()
