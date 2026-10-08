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
    if not isinstance(doc['score'],(float,int)) or not math.isfinite(doc['score']):raise ValueError('Nonfinite retrieval score')
    doc=doc['document']
   if not isinstance(doc,dict) or not isinstance(doc.get('contents'),str) or not doc['contents'].strip():raise ValueError('Empty retrieval content')
 return payload

def verify(url,session):
 queries=['What is the capital of France?','Who wrote Hamlet?','What is the capital of Japan?']
 records=[]
 for scored in (False,True):
  request={'queries':queries,'topk':3,'return_scores':scored};started=time.time()
  response=session.post(url,json=request,timeout=120)
  if response.status_code!=200:raise ValueError('Actual retrieval HTTP '+str(response.status_code))
  payload=check_payload(response.json(),queries,scored)
  records.append({'request':request,'response':payload,'status_code':response.status_code,'elapsed_seconds':time.time()-started})
 # Original dump's false-response parser reads contents, not document wrappers.
 joined=['\n'.join(d['contents'] for d in row) for row in records[0]['response']['result']]
 if not all(s.strip() for s in joined):raise ValueError('Original dump parsed empty information')
 return {'scope':'ACTUAL_LOCAL_RETRIEVAL_PROTOCOL_FALSE_TRUE_TOP3_NOT_CALIBRATION_APPROVAL','status':'VERIFIED','url':url,'records':records,'original_dump_nonempty_joined':joined}
