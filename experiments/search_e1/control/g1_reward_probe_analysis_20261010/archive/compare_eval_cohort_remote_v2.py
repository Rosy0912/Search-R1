"""Archive dev512 exact G1-normalization cohort hash, CPU-only read-only."""
from collections import Counter
from pathlib import Path
import datetime, hashlib, json, re, socket
import pyarrow.parquet as pq
B=Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
P=B/'data/validation.parquet'
raw=P.read_bytes(); before=hashlib.sha256(raw).hexdigest()
assert before=='30d8a6175f3f3f7cc6d4e106521035be672cf9b35e98d371a8e81ac487859e71'
table=pq.read_table(P,columns=['prompt','data_source'])
ids=[]
for chat in table['prompt'].to_pylist():
 text=chat[0]['content']; matches=list(re.finditer(r'Question:\s*',text));assert matches
 question=re.sub(r'\s+',' ',text[matches[-1].end():]).strip().casefold()
 ids.append(hashlib.sha256(question.encode()).hexdigest())
assert len(ids)==len(set(ids))==512
after=hashlib.sha256(P.read_bytes()).hexdigest();assert after==before
order=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
sethash=hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()
print(json.dumps({'read_only':True,'CPU_only':True,'questions_transferred':False,'observed_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'hostname':socket.gethostname(),'path':str(P),'filebytes':len(raw),'parquet_sha256_before_after':before,'source_counts':dict(Counter(table['data_source'].to_pylist())),'question_count':len(ids),'unique_questions':len(set(ids)),'normalization_definition':'G1exact regexlastQuestion:\\s* suffix; whitespacecollapse;strip;casefold;UTF8SHA256;compactJSONlistSHA256','question_order_compact_sha256':order,'question_set_compact_sha256':sethash,'G1_comparison':{'expected_order':'b8557f89a232e9f0cf4532222e679ceb141d0d398ed4d9ca91906cf67355bbe6','expected_set':'256b08cff964dbccaef4eab5dd98feaf9726465ab8963748a4c8d850f3510e65','ordered_equal':order=='b8557f89a232e9f0cf4532222e679ceb141d0d398ed4d9ca91906cf67355bbe6','set_equal':sethash=='256b08cff964dbccaef4eab5dd98feaf9726465ab8963748a4c8d850f3510e65'}}))
