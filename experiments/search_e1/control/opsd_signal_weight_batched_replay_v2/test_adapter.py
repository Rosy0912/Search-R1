import copy,unittest
from adapter import canonical_rows,same_ref
from common import semantic_sha

class Tests(unittest.TestCase):
    def setUp(self):
        self.data=[dict(question_id=str(i),question='q'+str(i),answers=['gold'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
        self.records=[];self.traces={}
        for i,q in enumerate(self.data):
            hit=int(i%2==0);decoded=dict(protocol_em=hit,protocol_valid_final_action=True,failure_class=None if hit else 'valid_gold_mismatch')
            result=dict(response='gold' if hit else 'wrong',reward=hit,policy_spans=[[0,4]],search_calls=0,termination='answer')
            self.traces[i]=dict(validation_index=i,question_id=q['question_id'],question=q['question'],answers=q['answers'],native_result=result,protocol=decoded)
            self.records.append(dict(validation_index=i,question_id=q['question_id'],primary=True,version='G_NEG',completed_outer=25,
                row_semantic_sha256=semantic_sha(q),native_result_sha256=semantic_sha(result),protocol_em=hit,trace={'fixture':i}))
    def reader(self,ref):return self.traces[ref['fixture']]
    def scorer(self,result,answers):
        hit=int(result['response']==answers[0]);return dict(protocol_em=hit,protocol_valid_final_action=True,failure_class=None if hit else 'valid_gold_mismatch')
    def test_all512_wrong_included_sorted_and_ordinary_scope(self):
        rows=canonical_rows(self.data,list(reversed(self.records)),self.reader,self.scorer)
        self.assertEqual([r['validation_index'] for r in rows],list(range(512)))
        self.assertEqual(sum(r['protocol_em'] for r in rows),256)
        self.assertEqual(rows[1]['failure_class'],'valid_gold_mismatch')
        self.assertEqual(rows[1]['dataset'],'nq')
        self.assertTrue(same_ref({'path':'p','sha256':'s','ctime':1},{'path':'p','sha256':'s','ctime':2}))
    def test_incomplete_duplicate_or_changed_response_rejected(self):
        with self.assertRaises(ValueError):canonical_rows(self.data,self.records[:-1],self.reader,self.scorer)
        with self.assertRaises(ValueError):canonical_rows(self.data,self.records[:-1]+[self.records[0]],self.reader,self.scorer)
        self.traces[3]['native_result']['response']='gold'
        with self.assertRaises(ValueError):canonical_rows(self.data,self.records,self.reader,self.scorer)
    def test_score_or_mixed_endpoint_rejected(self):
        rows=copy.deepcopy(self.records);rows[0]['protocol_em']=0
        with self.assertRaises(ValueError):canonical_rows(self.data,rows,self.reader,self.scorer)
        rows=copy.deepcopy(self.records);rows[0]['version']='G_POS'
        with self.assertRaises(ValueError):canonical_rows(self.data,rows,self.reader,self.scorer)
    def test_common_zero_and_raw_endpoint_identity_are_not_manifest_aliases(self):
        from summary import validate_endpoint_identity
        for v,t in [('source100',0),('G_NEG',25)]:
            raw=dict(kind='common_zero_adapter' if t==0 else 'raw_end_of_outer',manifest=dict(path='raw',sha256='r'*64),optimizer_updates=t*10)
            identity=dict(expected_adapter_tensor_sha256='a'*64,raw_checkpoint=raw)
            ep=dict(accepted=True,schema='opsd-signal-batched-canonical-endpoint-v1',version=v,completed_outer=t,primary_slots=512,
                actual_worker_wait_returncodes=[0]*8,input_identity=identity,raw_checkpoint=raw)
            req=dict(endpoint={'sha256':'e'*64},input_identity=identity)
            self.assertEqual(validate_endpoint_identity(ep,req,{'sha256':'e'*64}),v+'/'+str(t))
            bad=copy.deepcopy(ep);bad['raw_checkpoint']=dict(manifest=raw['manifest'])
            with self.assertRaises(ValueError):validate_endpoint_identity(bad,req,{'sha256':'e'*64})
            bad=copy.deepcopy(ep);bad['actual_worker_wait_returncodes'][7]=1
            with self.assertRaises(ValueError):validate_endpoint_identity(bad,req,{'sha256':'e'*64})
if __name__=='__main__':unittest.main()
