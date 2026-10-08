"""Completed opportunity accounting tests; no fixture counts as actual native training."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from reproduction.common8.test_configuration import ConfigurationTests
from reproduction.common8.update_evidence import METRICS,record
from reproduction.common8.runtime import budget_rows


class InlineUpdateEvidenceTests(unittest.TestCase):
    def attempt(self):
        return dict(status='complete',logical_update=1,completed_outer=1,complete_outer=True,
            scheduler_advances=1,attempt_id='a'*32,attempt_index=2,all_ranks_agreed=True,real_rank_count=8,
            optimizer_writes_before=4,optimizer_writes_after=17,optimizer_writes_this_attempt=13,
            opportunities_processed=20)

    def metrics(self):
        result={key:0. for key in METRICS}
        result.update({'actor/optimizer_steps_this_update':13.,'actor/update_opportunities_this_attempt':20.,
                       'actor/skipped_update_opportunities':7.})
        return result

    def test_actual_fewer_writes_complete_and_retain_previous_attempt_cumulative_count(self):
        with tempfile.TemporaryDirectory() as root:
            base,run=ConfigurationTests().fixture(root)
            gate=dict(proof_sha256='b'*64,private_trace_sha256='c'*64)
            with patch.dict(os.environ,SEARCH_G1_BASE=str(base)), \
                 patch('reproduction.common8.behavior_runtime.read_gate_receipt',return_value=gate) as read_gate:
                result=record(1,self.metrics(),'actual-frozen-probe',attempt_result=self.attempt())
            self.assertEqual(result['actual_optimizer_writes'],13)
            self.assertEqual((result['optimizer_writes_before'],result['optimizer_writes_after']),(4,17))
            self.assertEqual(result['opportunities_processed'],20)
            self.assertEqual(read_gate.call_args.kwargs,dict(attempt_id='a'*32,attempt_index=2))
            self.assertEqual(json.loads((base/'logs/actual-updates.jsonl').read_text()),result)

    def test_zero_write_incomplete_or_unagreed_attempt_never_counts_as_complete(self):
        for field,value in (('opportunities_processed',19),('optimizer_writes_this_attempt',0),
                            ('all_ranks_agreed',False),('status','aborted_attempt'),('complete_outer',False)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as root:
                base,_=ConfigurationTests().fixture(root);attempt=self.attempt();attempt[field]=value
                with patch.dict(os.environ,SEARCH_G1_BASE=str(base)):
                    with self.assertRaises(AssertionError):record(1,self.metrics(),'probe',attempt_result=attempt)
                self.assertFalse((base/'logs/actual-updates.jsonl').exists())

    def evidence(self):
        lines=[];record=dict(step=1,attempt_id='a'*32,attempt_index=2,
                           optimizer_writes_before=4,optimizer_writes_after=17,opportunities_processed=20)
        for rank in range(8):
            cumulative=4
            for mini in range(20):
                success=mini<13;context=dict(logical_update=1,attempt_id='a'*32,attempt_index=2,minibatch_index=mini)
                opportunity=dict(rank=rank,context=context,opportunity_index=mini,processed=True,
                    optimizer_write_succeeded=success,status='written' if success else 'skipped_guard',
                    optimizer_writes_before=cumulative,optimizer_writes_after=cumulative+int(success))
                lines.append('SEARCH_G1_OPPORTUNITY='+json.dumps(opportunity))
                if success:
                    cumulative+=1
                    lines.append('SEARCH_G1_OPTIMIZER_WRITE='+json.dumps(dict(rank=rank,context=context,
                        write_count=cumulative,learning_rates=[0.])))
            lines.append('SEARCH_G1_OUTER_BUDGET='+json.dumps(dict(rank=rank,logical_update=1,
                attempt_id='a'*32,attempt_index=2,local_trajectories=320,local_mini=16,local_micro=2,
                opportunities_processed=20,optimizer_writes=13,cumulative_writes=17)))
            lines.append('SEARCH_G1_OPTIMIZER_WRITE='+json.dumps(dict(rank=rank,
                context=dict(logical_update=1,attempt_id='b'*32,attempt_index=1,minibatch_index=0),
                write_count=1,learning_rates=[0.])))
        return lines,[record]

    def test_real_opportunities_and_write_subset_are_bound_to_completed_attempt(self):
        lines,records=self.evidence()
        proof=budget_rows('\n'.join(lines),1,1,complete_records=records)
        self.assertEqual(len(proof),8)
        self.assertTrue(all(row['optimizer_writes']==13 for row in proof))
        for mode in ('missing_opportunity','forged_write_status','wrong_cumulative','wrong_lr'):
            changed=list(lines)
            if mode=='missing_opportunity':changed.pop(0)
            elif mode=='forged_write_status':changed[0]=changed[0].replace('"written"','"skipped_guard"')
            elif mode=='wrong_cumulative':changed[0]=changed[0].replace('"optimizer_writes_after": 5','"optimizer_writes_after": 4')
            else:changed[1]=changed[1].replace('[0.0]','[1e-06]')
            with self.subTest(mode=mode),self.assertRaises(AssertionError):
                budget_rows('\n'.join(changed),1,1,complete_records=records)


if __name__=='__main__':unittest.main()
