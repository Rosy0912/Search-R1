"""Supervise original dump's silent HTTP fallback without altering its math."""
import hashlib,pathlib,re
class RetrievalHTTPFailure(RuntimeError):pass

def inspect_log(path):
 p=pathlib.Path(path)
 if not p.is_file():return {'scope':'ACTUAL_RETRIEVAL_HTTP_LOG_AUDIT','path':str(p),'status':'LOG_NOT_YET_CREATED','requests':0}
 raw=p.read_bytes();text=raw.decode('utf-8',errors='replace');records=[];failures=[]
 for line in text.splitlines():
  match=re.search(r'"POST /retrieve(?:\?[^ ]*)? HTTP/[^"]+"\s+(\d{3})\b',line)
  if match:
   code=int(match.group(1));records.append({'status_code':code,'raw':line})
   if code!=200:failures.append(line)
  if 'ERROR:    Exception in ASGI application' in line or 'Traceback (most recent call last):' in line:failures.append(line)
 receipt={'scope':'ACTUAL_RETRIEVAL_HTTP_LOG_AUDIT_NOT_GENERATION_QUALITY','path':str(p),'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'requests':len(records),'http_status_counts':{str(c):sum(r['status_code']==c for r in records) for c in sorted({r['status_code'] for r in records})},'records':records,'failures':failures,'status':'PASS_NO_HTTP_OR_ASGI_FAILURE' if not failures else 'FAIL_RETRIEVAL_HTTP_OR_ASGI'}
 return receipt

def require_healthy_log(path,persist=None):
 receipt=inspect_log(path)
 if receipt.get('status')=='LOG_NOT_YET_CREATED':raise RetrievalHTTPFailure('Actual retriever log disappeared/missing')
 if receipt.get('failures'):
  if persist:persist(receipt)
  raise RetrievalHTTPFailure('Actual retrieval HTTP/ASGI failed; original dump fallback cannot authorize calibration')
 return receipt
