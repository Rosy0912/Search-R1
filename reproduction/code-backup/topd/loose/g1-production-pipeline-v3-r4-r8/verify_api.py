"""Actual local GPU retriever protocol check, not a surrogate resource proof."""
import hashlib,json,pathlib,time

def check_payload(payload,queries,scored):
 if not isinstance(payload,dict) or set(payload)!={'result'} or len(payload['result'])!=len(queries):raise ValueError('Retrieval row schema/ordering count')
 for row in payload['result']:
  if not isinstance(row,list) or len(row)!=3:raise ValueError('Actual top3 absent')
  for doc in row:
   if scored:
    if not isinstance(doc,dict) or set(doc)!={'document','score'}:raise ValueError('Scored response schema')
    import math
    if isinstance(doc['score'],bool) or not isinstance(doc['score'],(float,int)) or not math.isfinite(doc['score']):raise ValueError('Nonfinite retrieval score')
    doc=doc['document']
   if not isinstance(doc,dict) or not isinstance(doc.get('contents'),str) or not doc['contents'].strip():raise ValueError('Empty retrieval content')
 return payload

def verify(url,session,persist=None,original_dump=None):
 queries=['What is the capital of France?','Who wrote Hamlet?','What is the capital of Japan?']
 records=[]
 for scored,queries in ((False,queries[:1]),(False,queries),(True,queries)):
  request={'queries':queries,'topk':3}
  if scored:request['return_scores']=True
  started=time.time()
  response=session.post(url,json=request,timeout=120)
  raw={'index':len(records),'request':request,'status_code':response.status_code,'elapsed_seconds':time.time()-started}
  if response.status_code!=200:
   raw['response_text']=getattr(response,'text','')
   if persist:persist(raw)
   raise ValueError('Actual retrieval HTTP '+str(response.status_code))
  try:payload=response.json()
  except Exception as parse_failure:
   raw['response_text']=getattr(response,'text','');raw['json_error']=repr(parse_failure)
   if persist:persist(raw)
   raise ValueError('Actual retrieval response is not JSON') from parse_failure
  raw['response']=payload
  if persist:persist(raw)
  payload=check_payload(payload,queries,scored)
  records.append({'request':request,'response':payload,'status_code':response.status_code,'elapsed_seconds':time.time()-started})
 # Original dump's false-response parser reads contents, not document wrappers.
 joined=['\n'.join(d['contents'] for d in row) for record in records if not record['request'].get('return_scores',False) for row in record['response']['result']]
 if not all(s.strip() for s in joined):raise ValueError('Original dump parsed empty information')
 parser_evidence=None
 if original_dump is not None:
  import ast,os,types
  source=pathlib.Path(original_dump).read_bytes();expected='816052e763ff52f4322932372a58ba0b438710af50db5f21a82631241f246719'
  if hashlib.sha256(source).hexdigest()!=expected:raise ValueError('Original dump parser source drift')
  tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='retrieve_batch');namespace={'os':os,'RETRIEVER_URL':url}
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(original_dump),'exec'),namespace)
  parsed=[]
  for record in records:
   if record['request'].get('return_scores',False):continue
   payload=record['response'];response=types.SimpleNamespace(status_code=200,json=lambda payload=payload:payload)
   namespace['_SESSION']=types.SimpleNamespace(post=lambda *a,response=response,**k:response)
   result=namespace['retrieve_batch'](record['request']['queries'],topk=3)
   expected_join=['\n'.join(d['contents'] for d in row) for row in payload['result']]
   if result!=expected_join or not all(x.strip() for x in result):raise ValueError('Actual original dump parser mismatch/empty')
   parsed.extend(result)
  parser_evidence={'original_dump_sha256':expected,'original_function':'retrieve_batch','joined':parsed,'response_replay_verified':True}
 return {'scope':'ACTUAL_LOCAL_RETRIEVAL_PROTOCOL_FALSE_TRUE_TOP3_NOT_CALIBRATION_APPROVAL','status':'VERIFIED','url':url,'records':records,'original_dump_nonempty_joined':joined,'original_dump_function_replay':parser_evidence}
