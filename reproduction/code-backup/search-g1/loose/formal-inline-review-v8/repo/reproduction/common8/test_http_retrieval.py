import copy,hashlib,json,unittest
from reproduction.common8.http_retrieval import compare_response

class HttpRetrievalTests(unittest.TestCase):
    def fixture(self):
        docs=[{'id':i,'contents':'document '+str(i)} for i in range(3)]
        row=[{'document':doc,'score':.9-i*.1} for i,doc in enumerate(docs)]
        hashes=[hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest() for d in docs]
        return {'result':[copy.deepcopy(row) for _ in range(64)]},{'gpu_document_sha256':[hashes]*64,'gpu_document_scores':[[.9,.8,.7]]*64}
    def test_actual_format_and_document_hashes_match(self):
        response,proof=self.fixture();self.assertEqual(compare_response(response,proof),proof['gpu_document_sha256'])
    def test_wrong_document_score_or_missing_question_rejected(self):
        for case in ('document','score','missing','nan'):
            response,proof=self.fixture()
            if case=='document':response['result'][1][0]['document']['contents']='wrong'
            if case=='score':response['result'][1][0]['score']=.5
            if case=='missing':response['result'].pop()
            if case=='nan':response['result'][1][0]['score']=float('nan')
            with self.assertRaises(AssertionError):compare_response(response,proof)
