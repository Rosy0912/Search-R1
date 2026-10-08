import pathlib,tempfile,json,hashlib,subprocess,sys,os,copy
root=pathlib.Path(__file__).resolve().parents[2];checker=root/'work/check_aux_retriever.py';rows=[]
with tempfile.TemporaryDirectory() as tmp:
 t=pathlib.Path(tmp);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 files={name:t/(name+'.json') for name in ('encoder','http','source','baseline','review')};files['source'].write_text('fixture-source');files['baseline'].write_text('{"baseline":2854}');files['review'].write_text('independent fixture review only')
 encoder={'status':'DIFFERENCE_REPORT_ONLY','source_sha256':sha(files['source']),'pid':1234,'gpu_uuid':'GPU-c63282c6-1d05-224e-234b-2a8c794b1f04','baseline_sha256':sha(files['baseline'])};files['encoder'].write_text(json.dumps(encoder))
 prov={'gpu_uuid':encoder['gpu_uuid'],'pid':1234,'run':'/data/shaoshitong/search-r1-rl/tree-grpo-retrieval/runs/fixture','encoder_device':'cpu','encoder_fp16':True,'index_float16':False,'difference_report_sha256':sha(files['encoder']),'source_sha256':sha(files['source'])}
 http={'status':'DIFFERENCE_REPORT_ONLY','baseline_sha256':sha(files['baseline']),'provenance':prov};files['http'].write_text(json.dumps(http))
 r={'status':'ACCEPTED_AFTER_REAL_ENCODER_HTTP_DIFFERENCE_REVIEW','url':'http://127.0.0.1:18082/retrieve','evidence_file_sha256':{str(p):sha(p) for p in files.values()},'provenance':prov}
 for role,key in [('encoder_report_file','encoder'),('http_report_file','http'),('source_file','source'),('baseline_file','baseline'),('independent_review_file','review')]:r[role]=str(files[key])
 original=t/'original.json';original.write_text(json.dumps(r));pin=sha(original)
 (t/'requests.py').write_text('''import os,json
class R:
 def raise_for_status(self): pass
 def json(self): return json.loads(os.environ['FAKE_PROVENANCE'])
def get(url,timeout):
 assert url=='http://127.0.0.1:18082/provenance' and 0<timeout<=180
 if os.environ.get('FAKE_TIMEOUT')=='1': raise TimeoutError('fixture timeout')
 return R()
''')
 def case(name,record=None,actual=None,timeout=False,pinned=None,expected=False):
  rec=r if record is None else record;p=t/(name+'.json');p.write_text(json.dumps(rec));env=dict(os.environ,PYTHONPATH=str(t),FAKE_PROVENANCE=json.dumps(prov if actual is None else actual),FAKE_TIMEOUT=str(int(timeout)))
  run=subprocess.run([sys.executable,str(checker),'--acceptance',str(p),'--expected-acceptance-sha256',pinned or sha(p),'--timeout','0.001'],env=env,text=True,capture_output=True)
  accepted=run.returncode==0;assert accepted==expected,(name,run.stderr)
  rows.append({'case':name,'accepted':accepted,'expected_accepted':expected,'exit_code':run.returncode,'stderr_tail':run.stderr[-240:]})
 case('positive',expected=True)
 case('not_accepted_status',dict(r,status='DIFFERENCE_REPORT_ONLY'))
 case('changed_instance',actual=dict(prov,pid=9999))
 case('bounded_read_timeout_mock',timeout=True)
 case('empty_evidence',dict(r,evidence_file_sha256={}))
 minimal={k:prov[k] for k in ('encoder_device','encoder_fp16','index_float16')};case('missing_identity',dict(r,provenance=minimal),minimal)
 for key,value in [('gpu_uuid','OTHER-GPU'),('pid',-1),('pid',True),('run','/foreign/run'),('run','/data/shaoshitong/search-r1-rl/tree-grpo-retrieval/runs/../foreign')]:
  wrong=dict(prov);wrong[key]=value;case('bad_'+key+'_'+str(value).replace('/','_'),dict(r,provenance=wrong),wrong)
 case('wrong_encoder_sha_binding',dict(r,provenance=dict(prov,difference_report_sha256='a'*64)))
 case('wrong_source_sha_binding',dict(r,provenance=dict(prov,source_sha256='a'*64)))
 removed=copy.deepcopy(r);del removed['evidence_file_sha256'][removed['http_report_file']];case('missing_http_evidence',removed)
 before=files['review'].read_bytes();files['review'].write_text('changed');case('changed_evidence');files['review'].write_bytes(before)
 case('acceptance_replacement_after_start',dict(r,status='REPLACED'),pinned=pin)
 original_http=files['http'].read_bytes();files['http'].write_text(json.dumps(dict(http,baseline_sha256='a'*64)));bad=copy.deepcopy(r);bad['evidence_file_sha256'][str(files['http'])]=sha(files['http']);case('http_wrong_baseline_even_rehashed',bad);files['http'].write_bytes(original_http)
 files['http'].write_text(json.dumps(dict(http,provenance=dict(prov,pid=5555))));bad=copy.deepcopy(r);bad['evidence_file_sha256'][str(files['http'])]=sha(files['http']);case('http_other_instance_even_rehashed',bad);files['http'].write_bytes(original_http)
 original_encoder=files['encoder'].read_bytes();files['encoder'].write_text(json.dumps(dict(encoder,pid=5555)));bad=copy.deepcopy(r);bad['evidence_file_sha256'][str(files['encoder'])]=sha(files['encoder']);bad['provenance']['difference_report_sha256']=sha(files['encoder']);case('encoder_other_instance_even_rehashed',bad);files['encoder'].write_bytes(original_encoder)
(root/'work/aux-gate-review/fixtures-results-fixed.json').write_text(json.dumps(rows,indent=2)+'\n');print('fixed checker: %d fixture cases passed'%len(rows))
