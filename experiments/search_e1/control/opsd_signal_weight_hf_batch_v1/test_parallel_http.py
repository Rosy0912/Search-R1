"""Independent requests overlap, but each body/executor state stays native."""
import threading,unittest
from search_batch import run_questions

class Tokenizer:
    def apply_chat_template(self,prompt,**kwargs):return prompt
    def encode(self,text,**kwargs):return list(text.encode())
    def decode(self,ids,**kwargs):return bytes(ids).decode()
class Response:
    def __init__(self,q):self.q=q
    def raise_for_status(self):pass
    def json(self):return {'result':[[{'document':{'contents':self.q}}]]}

class Tests(unittest.TestCase):
    def test_four_requests_overlap_without_batching_e5_or_extra_queries(self):
        barrier=threading.Barrier(4);calls=[];lock=threading.Lock()
        class HTTP:
            def post(self,url,json,timeout):
                selftest.assertEqual(json['topk'],3);selftest.assertTrue(json['return_scores'])
                selftest.assertEqual(len(json['queries']),1);selftest.assertEqual(timeout,120)
                with lock:calls.append(json['queries'][0])
                barrier.wait(timeout=2)
                return Response(json['queries'][0])
        selftest=self
        def generate(texts):
            return [{'text':'<answer>yes</answer>' if '<information>' in s else '<search>'+s+'</search>'} for s in texts]
        rr=run_questions(['a','b','c','d'],Tokenizer(),lambda q:q,generate,HTTP(),'oracle')
        self.assertEqual(sorted(calls),['a','b','c','d'])
        self.assertEqual([r['search_calls'] for r in rr],[1]*4)
        for q,r in zip(['a','b','c','d'],rr):
            self.assertIn('Doc 1: '+q,r['response']);self.assertEqual(r['termination'],'answer')
        self.assertEqual(len(rr),4)
if __name__=='__main__':unittest.main()
