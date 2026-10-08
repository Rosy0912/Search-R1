import pathlib,tempfile,json,hashlib,copy,types,importlib.util,sys,os
root=pathlib.Path(__file__).resolve().parents[2];source=root/'work/check_hybrid_retriever.py';spec=importlib.util.spec_from_file_location('checker',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
 t=pathlib.Path(tmp);f={r:t/(r+'.json') for r in m.ROLES}
 for role in m.ROLES:f[role].write_text('fixture '+role)
 base={'mode':'cpu','asset_identity_verified':True,'queries_sha256':'a'*64,'embeddings_sha256':'b'*64,'asset_identity':{'index':{'sha256':m.INDEX_SHA},'corpus':{'sha256':'c'*64},'model':{'model.safetensors':'d'*64,'config.json':'e'*64,'model.safetensors.partial':'f'*64}}};f['baseline_file'].write_text(json.dumps(base))
 index={'gpu_uuid':m.UUID,'pid':774407,'run':m.AUX+'/runs/index-fixture','mode':'fp32-index-only','index_float16':False,'index_sha256':m.INDEX_SHA,'ntotal':21015324,'dimension':768,'scratch_bytes':512*1024*1024,'source_sha256':sha(f['index_source_file'])}
 gate={'mode':'h100-cpu-encoder-remote-fp32-index','pid':100,'run':m.OWN+'/runs/hybrid-cpu-gate-41/service','source_sha256':sha(f['hybrid_source_file']),'index_instance':index,'encoder_device':'cpu','encoder_fp16':True,'index_float16':False,'baseline_sha256':sha(f['baseline_file']),'model_sha256':{k:v for k,v in base['asset_identity']['model'].items() if k not in m.IGNORED},'asset_identity':base['asset_identity'],'retrieval_module_sha256':'1'*64}
 def report_rows(kind):
  data=[]
  for n,k in ([(14,3),(14,50),(512,3),(512,50)] if kind=='binary' else [(14,3),(14,50),(512,3)]):
   entries=[dict(row=i,query_index=i%14,ids=list(range(k)),scores=list(map(float,range(k)))) for i in range(n)]
   if kind=='http':
    for e in entries:e['contents_sha256']=['2'*64]*k
   data.append({('queries' if kind=='binary' else 'query_count'):n,'k':k,'unique_queries':14,'repeat_equal':True,'repeats':[entries,copy.deepcopy(entries)]})
  return data
 binary={'status':'DIFFERENCE_REPORT_ONLY','instance':index,'baseline_sha256':sha(f['baseline_file']),'source_sha256':sha(f['binary_probe_source_file']),'vectors_sha256':base['embeddings_sha256'],'embedding_file_sha256':'3'*64,'rows':report_rows('binary')}
 http={'status':'DIFFERENCE_REPORT_ONLY','url':'http://127.0.0.1:18082','provenance':gate,'baseline_sha256':sha(f['baseline_file']),'source_sha256':sha(f['http_probe_source_file']),'queries_sha256':base['queries_sha256'],'rows':report_rows('http')}
 stage=dict(copy.deepcopy(gate),pid=101,run=m.OWN+'/runs/tree-two-42/hybrid');os.environ['SLURM_JOB_ID']='42'
 def case(name,change=None,network=None,bad_stage_sha=False,bad_acceptance_sha=False,change_evidence=False,expected=False):
  b,h,s,g,ix=copy.deepcopy((binary,http,stage,gate,index));r={'status':'ACCEPTED_AFTER_HYBRID_BINARY_HTTP_DIFFERENCE_REVIEW','url':'http://127.0.0.1:18082/retrieve','index_url':'http://127.0.0.1:18083','index_instance':ix,'provenance':g,**{role:str(p) for role,p in f.items()}}
  if change:change(r,s,b,h)
  f['binary_report_file'].write_text(json.dumps(b));f['hybrid_http_report_file'].write_text(json.dumps(h));r['evidence_file_sha256']={str(p):sha(p) for p in f.values()}
  acceptance=t/'acceptance.json';acceptance.write_text(json.dumps(r));startup=t/'stage.json';startup.write_text(json.dumps(s))
  acceptance_sha=sha(acceptance);stage_sha=sha(startup)
  if bad_acceptance_sha:acceptance_sha='0'*64
  if bad_stage_sha:stage_sha='0'*64
  original_review=f['independent_review_file'].read_bytes()
  if change_evidence:f['independent_review_file'].write_text('mutated')
  calls=[]
  class Response:
   def __init__(self,value,status=200):self.value=value;self.status=status
   def raise_for_status(self):
    if self.status!=200:raise RuntimeError('HTTP503 poisoned')
   def json(self):return self.value
  def get(url,timeout):
   calls.append((url,timeout));assert 0<timeout<=180
   value=copy.deepcopy(ix if ':18083/' in url else s)
   if network=='503' or (network=='cpu503' and ':18082/' in url):return Response(value,503)
   if network=='timeout':raise TimeoutError('fixture timeout')
   if network=='index_changed' and ':18083/' in url:value['pid']+=1
   if network=='cpu_changed' and ':18082/' in url:value['pid']+=1
   return Response(value)
  m.requests=types.SimpleNamespace(get=get);sys.argv=[str(source),'--acceptance',str(acceptance),'--expected-acceptance-sha256',acceptance_sha,'--stage-provenance',str(startup),'--expected-stage-provenance-sha256',stage_sha]
  try:m.main();passed=True;error=None
  except (Exception,SystemExit) as e:passed=False;error=repr(e)
  finally:f['independent_review_file'].write_bytes(original_review)
  assert passed==expected,(name,error)
  rows.append({'case':name,'accepted':passed,'expected_accepted':expected,'error':error,'network_calls':calls})
 case('success_new_cpu_pid_run',expected=True)
 for stage_name in ('five','formal'):case('success_'+stage_name,lambda r,s,b,h:s.update(run=m.OWN+'/runs/tree-'+stage_name+'-42/hybrid'),expected=True)
 for value in (0,-1,True,100):case('bad_or_retired_pid_'+str(value),lambda r,s,b,h,v=value:s.update(pid=v))
 for value in ('/foreign/hybrid',m.OWN+'/runs/tree-two-43/hybrid',m.OWN+'/runs/tree-two-42/service',m.OWN+'/runs/tree-two-42/../hybrid',m.OWN+'/runs/hybrid-cpu-gate-41/service'):
  case('bad_run_'+value,lambda r,s,b,h,v=value:s.update(run=v))
 case('stage_sha_changed',bad_stage_sha=True);case('acceptance_sha_changed',bad_acceptance_sha=True);case('evidence_changed',change_evidence=True)
 case('source_changed',lambda r,s,b,h:s.update(source_sha256='0'*64))
 case('assets_changed',lambda r,s,b,h:s['asset_identity']['corpus'].update(sha256='0'*64))
 case('model_changed',lambda r,s,b,h:s['model_sha256'].update(**{'model.safetensors':'0'*64}))
 case('stage_index_changed',lambda r,s,b,h:s['index_instance'].update(pid=999))
 case('returned_index_changed',network='index_changed');case('returned_cpu_changed',network='cpu_changed');case('index503',network='503');case('hybrid_poison503',network='cpu503');case('network_timeout',network='timeout')
 case('unstable_binary_flag',lambda r,s,b,h:b['rows'][0].update(repeat_equal=False))
 case('unstable_http_even_flag_true',lambda r,s,b,h:h['rows'][0]['repeats'][1][0]['scores'].__setitem__(0,1.0))
 case('incomplete_binary_rows',lambda r,s,b,h:b.update(rows=[]))
 case('duplicate_http_rows',lambda r,s,b,h:h['rows'].__setitem__(1,copy.deepcopy(h['rows'][0])))
 case('binary_wrong_source',lambda r,s,b,h:b.update(source_sha256='0'*64))
 case('wrong_http_gate_url',lambda r,s,b,h:h.update(url='http://foreign:18082'))
 case('http_wrong_baseline',lambda r,s,b,h:h.update(baseline_sha256='0'*64))
 case('literal_extra_stage_field',lambda r,s,b,h:s.update(unreviewed='x'))
 case('nonfinite_result',lambda r,s,b,h:b['rows'][0]['repeats'][0][0]['scores'].__setitem__(0,float('nan')))
 (root/'work/aux-gate-review/hybrid-checker-fixtures.json').write_text(json.dumps({'scope':'actual checker main with mocked HTTP and generated reviewed evidence; no actual model/asset/GPU/Slurm acceptance','source_sha256':sha(source),'cases':rows},indent=2)+'\n');print('%d cases passed'%len(rows))
