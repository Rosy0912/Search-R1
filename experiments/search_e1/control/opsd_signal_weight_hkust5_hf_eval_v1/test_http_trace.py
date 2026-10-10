import io,json,threading,unittest
from concurrent.futures import ThreadPoolExecutor
from http_trace import ConcurrentHTTP

class Tests(unittest.TestCase):
    def test_parallel_original_bodies_unique_atomic_records_and_same_response_objects(self):
        barrier=threading.Barrier(4);responses={}
        class Response:
            status_code=200
            def __init__(self,q):self.q=q;self.text=json.dumps({'result':self.q})
            def json(self):return {'result':self.q}
        class Requests:
            def post(self,*args,**kwargs):
                q=kwargs['json']['queries'];r=Response(q);responses[q[0]]=r;barrier.wait(timeout=2);return r
        stream=io.StringIO();proxy=ConcurrentHTTP(Requests(),stream,0,'model',lambda:{'batch':0})
        with ThreadPoolExecutor(max_workers=4) as pool:
            ff=[pool.submit(proxy.for_case(i,1).post,'oracle',json={'queries':[str(i)],'topk':3,'return_scores':True},timeout=120) for i in range(4)]
            rr=[f.result() for f in ff]
        rows=[json.loads(l) for l in stream.getvalue().splitlines()]
        self.assertEqual(sorted(r['request_id'] for r in rows),list(range(4)))
        self.assertEqual(sorted(r['executor_case']['question_slot'] for r in rows),list(range(4)))
        for i,r in enumerate(rr):self.assertIs(r,responses[str(i)])
        self.assertTrue(all(len(r['keyword_args']['json']['queries'])==1 and r['keyword_args']['timeout']==120 for r in rows))
        self.assertEqual(proxy.logging_errors,[])
if __name__=='__main__':unittest.main()
