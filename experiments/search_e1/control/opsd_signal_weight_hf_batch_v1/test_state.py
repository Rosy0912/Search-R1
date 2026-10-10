import ast,copy,pathlib,types,unittest,os,hashlib
from search_batch import run_questions,FORCED

class IDs:
    def __init__(self,data):self.data=data;self.shape=(1,len(data))
    def to(self,device):return self
    def __getitem__(self,key):return self.data[key[1]] if isinstance(key,tuple) else self.data
class Tokenizer:
    eos_token_id=0
    def apply_chat_template(self,messages,**kwargs):return 'prefix('+messages[0]+')'
    def encode(self,text,**kwargs):return list(text)
    def decode(self,ids,**kwargs):return ''.join(ids)
    def __call__(self,text,**kwargs):return types.SimpleNamespace(input_ids=IDs(list(text)))
class HTTP:
    def __init__(self):self.requests=[]
    def post(self,url,json,timeout):
        self.requests.append((url,copy.deepcopy(json),timeout))
        self.value={'result':[[{'document':{'text':'doc-text'}}]]}
        return self
    def raise_for_status(self):pass
    def json(self):return self.value
def output(text):
    q=text.split('prefix(',1)[1].split(')',1)[0]
    turns=text.count('<search>')
    if q=='answer':return '<answer>gold</answer>trailing'
    if q=='invalid':return 'no action'
    if q=='search' and turns==0:return '<search> request </search>'
    if q=='four' and turns<4:return '<search>request</search>'
    return '<answer>gold</answer>'
class Model:
    def eval(self):return self
    def parameters(self):yield types.SimpleNamespace(device='cpu')
    def generate(self,ids,**kwargs):return IDs(ids.data+list(output(''.join(ids.data))))

def original_class():
    p=pathlib.Path(os.environ.get('OPSD_HF_BATCH_FROZEN_ROLLOUT',str(pathlib.Path(__file__).resolve().parent.parent/'opsd_debug_frozen_runtime_20261008/runtime/search_e1/rollout.py')))
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='a88506797ea7fe24e61e1ad53ba4487aef87bdd69ec395aef26c94973b3d629a'
    source=p.read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SearchAgent')
    # Execute the actual published class body, no rewritten reference algorithm.
    namespace=dict(torch=types.SimpleNamespace(no_grad=lambda:lambda f:f),
        prompt=lambda q:[q],StoppingCriteriaList=list,StopTags=lambda *a:object(),re=__import__('re'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),namespace)
    return namespace['SearchAgent'],namespace

class Tests(unittest.TestCase):
    def compare(self,questions,max_context=8192):
        cls,namespace=original_class();native_http=HTTP();namespace['requests']=native_http;tokenizer=Tokenizer()
        native=[cls(Model(),tokenizer,'local',max_context=max_context).run(q,temperature=0) for q in questions]
        batched_http=HTTP();calls=[]
        def generate(texts):calls.append(len(texts));return [dict(text=output(t),token_ids=list(output(t))) for t in texts]
        actual=run_questions(questions,tokenizer,lambda q:[q],generate,batched_http,'local',max_context=max_context)
        self.assertEqual(actual,native)
        self.assertEqual(sorted(native_http.requests,key=repr),sorted(batched_http.requests,key=repr))
        return actual,calls
    def test_mixed_termination_and_independent_four_search_budget(self):
        result,calls=self.compare(['answer','search','invalid','four'])
        self.assertEqual(calls[0],4);self.assertLess(calls[1],4)
        self.assertEqual(result[3]['search_calls'],4);self.assertIn(FORCED,result[3]['response'])
        self.assertEqual(result[2]['termination'],'invalid_or_budget')
    def test_context_limit_does_not_insert_forced_text(self):
        result,_=self.compare(['four'],max_context=900)
        self.assertEqual(result[0]['termination'],'context_limit');self.assertNotIn(FORCED,result[0]['response'])
    def test_initial_context_boundary_has_no_policy_or_request(self):
        result,calls=self.compare(['answer'],max_context=501)
        self.assertEqual(calls,[]);self.assertEqual(result[0]['policy_spans'],[])
if __name__=='__main__':unittest.main()
