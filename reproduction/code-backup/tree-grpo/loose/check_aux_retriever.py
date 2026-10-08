"""Read-only binding to one independently accepted own auxiliary instance."""
import argparse,json,pathlib,hashlib,re,requests
p=argparse.ArgumentParser();p.add_argument('--acceptance',required=True,type=pathlib.Path);p.add_argument('--expected-acceptance-sha256',required=True);p.add_argument('--output',type=pathlib.Path);p.add_argument('--timeout',type=float,default=180);a=p.parse_args()
raw=a.acceptance.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.expected_acceptance_sha256,'Acceptance changed since own stage start'
r=json.loads(raw);assert r['status']=='ACCEPTED_AFTER_REAL_ENCODER_HTTP_DIFFERENCE_REVIEW'
assert r['url']=='http://127.0.0.1:18082/retrieve' and a.timeout>0
required=['encoder_report_file','http_report_file','source_file','baseline_file','independent_review_file']
evidence=r['evidence_file_sha256'];assert isinstance(evidence,dict) and len(evidence)>=len(required)
for role in required:assert r[role] in evidence and re.fullmatch('[0-9a-f]{64}',evidence[r[role]]),'Missing required reviewed evidence'
for file,expected in evidence.items():assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==expected,'Accepted evidence changed'
expected=r['provenance'];assert expected['gpu_uuid']=='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
assert isinstance(expected['pid'],int) and not isinstance(expected['pid'],bool) and expected['pid']>0
remote_run=pathlib.PurePosixPath(expected['run']);assert remote_run.parent==pathlib.PurePosixPath('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval/runs') and '..' not in remote_run.parts
assert expected['difference_report_sha256']==evidence[r['encoder_report_file']]
assert expected['source_sha256']==evidence[r['source_file']]
assert expected['encoder_device']=='cpu' and expected['encoder_fp16'] is True and expected['index_float16'] is False
encoder=json.loads(pathlib.Path(r['encoder_report_file']).read_text());http=json.loads(pathlib.Path(r['http_report_file']).read_text())
assert encoder['source_sha256']==expected['source_sha256'] and encoder['pid']==expected['pid'] and encoder['gpu_uuid']==expected['gpu_uuid']
assert encoder['baseline_sha256']==evidence[r['baseline_file']] and http['baseline_sha256']==evidence[r['baseline_file']]
assert http['provenance']==expected
response=requests.get('http://127.0.0.1:18082/provenance',timeout=a.timeout);response.raise_for_status();provenance=response.json()
assert provenance==expected,'Different auxiliary service instance or configuration'
if a.output:
 with a.output.open('x') as f:json.dump(provenance,f,indent=2);f.write('\n')
print('AUX_RETRIEVER_BOUND_TO_ACCEPTED_INSTANCE')
