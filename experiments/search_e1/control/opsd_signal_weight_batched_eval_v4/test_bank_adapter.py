import tempfile,pathlib,json,unittest,copy
from bank_adapter import export_endpoint
from evaluate import write,proof
class BankTests(unittest.TestCase):
    def setUp(self):
        self.data=[dict(question_id=str(i),question='q'+str(i),answers=['a'],data_source='nq' if i<256 else 'hotpotqa') for i in range(512)]
        self.results=[dict(q,response='<answer>a</answer>',policy_spans=[[0,18]],reward=int(i%2==0),protocol={'protocol_em':int(i%2==0)}) for i,q in enumerate(self.data)]
        self.request=dict(config={'path':'cfg','sha256':'a'*64},checkpoint=None,completed_outer=0,data={'path':'data','sha256':'b'*64})
    def test_full512_correct_only_reference_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);directory=root/'common-zero';directory.mkdir()
            inp,bank=export_endpoint(root,directory,'common-zero',self.data,list(reversed(self.results)),{'path':'summary','sha256':'c'*64},self.request,lambda r,a:r['protocol'],write,proof)
            value=json.loads(pathlib.Path(bank['path']).read_text());self.assertEqual(value['questions'],512);self.assertEqual(value['available'],256);self.assertTrue(value['fixed_before_eight_new_endpoint_outputs'])
            self.assertIsNone(value['entries'][1]['reference_response']);self.assertEqual(value['entries'][0]['question_id'],'0');self.assertTrue(value['baseline_bank_subset_em_constant_one'])
            self.assertTrue(json.loads(pathlib.Path(inp['path']).read_text())['posthoc_probabilities_not_yet_computed'])
    def test_late_bank_or_duplicate_questions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);directory=root/'common-zero';directory.mkdir();(root/'G_POS-15').mkdir()
            with self.assertRaises(ValueError):export_endpoint(root,directory,'common-zero',self.data,self.results,{},self.request,lambda r,a:r['protocol'],write,proof)
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);directory=root/'common-zero';directory.mkdir();rows=self.results[:-1]+[self.results[0]]
            with self.assertRaises(ValueError):export_endpoint(root,directory,'common-zero',self.data,rows,{},self.request,lambda r,a:r['protocol'],write,proof)
if __name__=='__main__':unittest.main()
