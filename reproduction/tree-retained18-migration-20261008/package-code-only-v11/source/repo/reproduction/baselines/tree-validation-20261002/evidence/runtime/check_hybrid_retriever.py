"""Bind one stage CPU service to reviewed immutable hybrid/index evidence."""
import argparse,pathlib,json,hashlib,re,math,os,requests

OWN='/dataset/shaoshitong/search-r1-rl/tree-grpo'
AUX='/data/shaoshitong/search-r1-rl/tree-grpo-retrieval'
UUID='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
INDEX_SHA='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'
ROLES=['binary_report_file','hybrid_http_report_file','index_source_file','hybrid_source_file','binary_probe_source_file','http_probe_source_file','baseline_file','independent_review_file']
INDEX_KEYS={'gpu_uuid','pid','run','mode','index_float16','index_sha256','ntotal','dimension','scratch_bytes','source_sha256'}
HYBRID_KEYS={'mode','pid','run','source_sha256','index_instance','encoder_device','encoder_fp16','index_float16','baseline_sha256','model_sha256','asset_identity','retrieval_module_sha256'}
IGNORED={'model.safetensors.partial','model.safetensors.source.partial','model.safetensors.source.partial.aria2'}

def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha_value(value):assert isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),'Invalid SHA256'
def pid(value):assert type(value) is int and value>0,'Invalid CPU/GPU PID'
def own_path(value,parent=None):
 assert isinstance(value,str);p=pathlib.PurePosixPath(value)
 assert p.is_absolute() and str(p)==value and '..' not in p.parts,'Noncanonical own run'
 if parent:assert p.parent==pathlib.PurePosixPath(parent),'Foreign run'
 return p

def validate_rows(report,kind):
 assert report['status']=='DIFFERENCE_REPORT_ONLY'
 expected={(14,3),(14,50),(512,3),(512,50)} if kind=='binary' else {(14,3),(14,50),(512,3)}
 observed=[]
 for row in report['rows']:
  n=row['queries'] if kind=='binary' else row['query_count'];k=row['k']
  assert type(n) is int and type(k) is int and (n,k) in expected
  observed.append((n,k));assert row['unique_queries']==14
  reps=row['repeats'];assert isinstance(reps,list) and len(reps)==2 and row['repeat_equal'] is True and reps[0]==reps[1],'Unstable or missing candidate repetitions'
  for rep in reps:
   assert isinstance(rep,list) and len(rep)==n
   for i,entry in enumerate(rep):
    assert entry['row']==i and entry['query_index']==i%14
    ids=entry['ids'];scores=entry['scores'];assert isinstance(ids,list) and isinstance(scores,list) and len(ids)==len(scores)==k
    assert all(type(v) is int and 0<=v<21015324 for v in ids) and len(set(ids))==k
    assert all(type(v) in (int,float) and math.isfinite(v) for v in scores)
    if kind=='http':
     hashes=entry['contents_sha256'];assert isinstance(hashes,list) and len(hashes)==k
     for h in hashes:sha_value(h)
 assert len(observed)==len(expected) and set(observed)==expected,'Incomplete/duplicate gate coverage'

def main():
 p=argparse.ArgumentParser();p.add_argument('--acceptance',type=pathlib.Path,required=True);p.add_argument('--expected-acceptance-sha256',required=True);p.add_argument('--stage-provenance',type=pathlib.Path,required=True);p.add_argument('--expected-stage-provenance-sha256',required=True);p.add_argument('--output',type=pathlib.Path);p.add_argument('--timeout',type=float,default=60);a=p.parse_args()
 for h in (a.expected_acceptance_sha256,a.expected_stage_provenance_sha256):sha_value(h)
 raw=a.acceptance.read_bytes();assert digest(raw)==a.expected_acceptance_sha256,'Stage acceptance changed';r=json.loads(raw)
 assert r['status']=='ACCEPTED_AFTER_HYBRID_BINARY_HTTP_DIFFERENCE_REVIEW'
 assert r['url']=='http://127.0.0.1:18082/retrieve' and r['index_url']=='http://127.0.0.1:18083' and math.isfinite(a.timeout) and 0<a.timeout<=180
 evidence=r['evidence_file_sha256'];assert isinstance(evidence,dict) and len(evidence)>=len(ROLES)
 assert len({r[role] for role in ROLES})==len(ROLES),'Evidence roles must identify distinct files'
 for role in ROLES:assert r[role] in evidence;sha_value(evidence[r[role]])
 for file,wanted in evidence.items():sha_value(wanted);assert digest(pathlib.Path(file).read_bytes())==wanted,'Reviewed evidence changed'
 index=r['index_instance'];assert set(index)==INDEX_KEYS
 assert index['gpu_uuid']==UUID and index['mode']=='fp32-index-only' and index['index_float16'] is False
 pid(index['pid']);own_path(index['run'],AUX+'/runs')
 assert index['source_sha256']==evidence[r['index_source_file']] and type(index['ntotal']) is int and index['ntotal']==21015324 and type(index['dimension']) is int and index['dimension']==768
 assert index['index_sha256']==INDEX_SHA and type(index['scratch_bytes']) is int and index['scratch_bytes']==512*1024*1024
 gate=r['provenance'];assert set(gate)==HYBRID_KEYS and gate['mode']=='h100-cpu-encoder-remote-fp32-index' and gate['index_instance']==index
 pid(gate['pid']);gate_run=own_path(gate['run']);assert pathlib.PurePosixPath(OWN+'/runs') in gate_run.parents
 assert gate['encoder_device']=='cpu' and gate['encoder_fp16'] is True and gate['index_float16'] is False
 assert gate['source_sha256']==evidence[r['hybrid_source_file']] and gate['baseline_sha256']==evidence[r['baseline_file']];sha_value(gate['retrieval_module_sha256'])
 baseline=json.loads(pathlib.Path(r['baseline_file']).read_text());assert baseline['mode']=='cpu' and baseline['asset_identity_verified'] is True
 assert gate['asset_identity']==baseline['asset_identity'] and baseline['asset_identity']['index']['sha256']==INDEX_SHA
 assert gate['model_sha256']=={k:v for k,v in baseline['asset_identity']['model'].items() if k not in IGNORED}
 for h in gate['model_sha256'].values():sha_value(h)
 binary=json.loads(pathlib.Path(r['binary_report_file']).read_text());http=json.loads(pathlib.Path(r['hybrid_http_report_file']).read_text())
 assert binary['instance']==index and binary['baseline_sha256']==evidence[r['baseline_file']] and binary['source_sha256']==evidence[r['binary_probe_source_file']] and binary['vectors_sha256']==baseline['embeddings_sha256'];sha_value(binary['embedding_file_sha256'])
 assert http['url']=='http://127.0.0.1:18082' and http['provenance']==gate and http['baseline_sha256']==evidence[r['baseline_file']] and http['source_sha256']==evidence[r['http_probe_source_file']] and http['queries_sha256']==baseline['queries_sha256']
 validate_rows(binary,'binary');validate_rows(http,'http')
 stage_raw=a.stage_provenance.read_bytes();assert digest(stage_raw)==a.expected_stage_provenance_sha256,'Stage CPU startup provenance changed';stage=json.loads(stage_raw)
 assert set(stage)==HYBRID_KEYS;pid(stage['pid']);assert stage['pid']!=gate['pid'],'Do not bind the retired gate CPU PID'
 stage_run=own_path(stage['run']);match=re.fullmatch(re.escape(OWN)+r'/runs/tree-(two|five|formal)-([1-9][0-9]*)/hybrid',str(stage_run));assert match and stage_run!=gate_run,'Invalid production stage CPU run'
 if os.environ.get('SLURM_JOB_ID'):assert match.group(2)==os.environ['SLURM_JOB_ID'],'Different own Slurm stage'
 assert {k:v for k,v in stage.items() if k not in ('pid','run')}=={k:v for k,v in gate.items() if k not in ('pid','run')},'Stage configuration/assets/index differs from reviewed gate'
 for url,wanted in [(r['index_url']+'/provenance',index),('http://127.0.0.1:18082/provenance',stage)]:
  response=requests.get(url,timeout=a.timeout);response.raise_for_status();assert response.json()==wanted,'Different or unhealthy pinned hybrid/index instance'
 if a.output:
  with a.output.open('x') as f:json.dump(stage,f,indent=2);f.write('\n')
 print('HYBRID_RETRIEVER_BOUND_TO_ACCEPTED_INSTANCES_AND_STAGE_CPU')

if __name__=='__main__':main()
