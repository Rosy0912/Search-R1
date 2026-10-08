import pathlib,tempfile,json,hashlib,time,types,unittest.mock as mock,sys
root=pathlib.Path.cwd();src=root/'work/dispatch_a800_index_formal.py';code=src.read_text();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();results=[]
cases=['positive','changed-source','acceptance-sha','not-five','wrong-actual','review-sha','bad-cost-json','nan-cost','future-end','wrong-gpu','wrong-old-pid','old-alive','port-busy','existing-runner','dangling-marker']
for case in cases:
 with tempfile.TemporaryDirectory() as td:
  r=pathlib.Path(td);[(r/n).mkdir(parents=True) for n in ['tools','inputs','logs','runs/fp32-index-http-20261003']];(r/'tools/index_gpu_server.py').write_bytes((root/'work/index_gpu_server.py').read_bytes());review=r/'inputs/five-real-stage-independent-review.md';review.write_text('independently accepted five fixture')
  accept=r/'inputs/five-stage-acceptance-reviewed.json';record={'status':'REAL_TREE_STAGE_ACCEPTED','stage':'five','actual_outer_updates':5,'independent_review_file':'/remote/review.md','evidence_file_sha256':{'/remote/review.md':digest(review)}}
  old=r/'runs/fp32-index-http-20261003';prov={'pid':774407,'run':str(old),'source_sha256':digest(r/'tools/index_gpu_server.py'),'gpu_uuid':'GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'};cost={'allocated_auxiliary_gpus':1,'gpu_uuid':prov['gpu_uuid'],'start_epoch':time.time()-15000,'end_epoch':time.time()-20,'gpu_peak_used_bytes':67000000000,'self_cpu_seconds':300,'reaped_child_cpu_seconds':0,'self_peak_rss_bytes':5000000000,'violations':[],'scope':'complete own process lifetime'}
  if case=='changed-source':(r/'tools/index_gpu_server.py').write_text('polluted')
  if case=='not-five':record['stage']='two'
  if case=='wrong-actual':record['actual_outer_updates']=2
  if case=='review-sha':review.write_text('changed review')
  if case=='nan-cost':cost['self_cpu_seconds']=float('nan')
  if case=='future-end':cost['end_epoch']=time.time()+5000
  if case=='wrong-gpu':cost['gpu_uuid']='GPU-foreign'
  if case=='wrong-old-pid':prov['pid']=123
  (old/'cost.json').write_text('{' if case=='bad-cost-json' else json.dumps(cost));(old/'provenance.json').write_text(json.dumps(prov));(r/'logs/fp32-index-http-20261003.exit').write_text('143\n');accept.write_text(json.dumps(record));pin=digest(accept)
  if case=='acceptance-sha':pin='0'*64
  if case=='existing-runner':(r/'tools/fp32-index-formal-20261003-runner.sh').write_text('do not overwrite')
  if case=='dangling-marker':(r/'logs/fp32-index-formal-20261003-dispatched.json').symlink_to(r/'absent')
  launch=[]
  def popen(*a,**kw):launch.append(a);return types.SimpleNamespace(pid=222)
  def kill(*a):
   if case=='old-alive':return
   raise ProcessLookupError()
  sock=mock.MagicMock();sock.__enter__.return_value=sock
  if case=='port-busy':sock.bind.side_effect=OSError('address in use')
  failure=None
  with mock.patch.object(sys,'argv',[str(src),'--five-acceptance-sha256',pin]),mock.patch('os.kill',side_effect=kill),mock.patch('socket.socket',return_value=sock),mock.patch('subprocess.Popen',side_effect=popen):
   try:exec(compile(code.replace("'/data/shaoshitong/search-r1-rl/tree-grpo-retrieval'",repr(str(r))),str(src),'exec'),{})
   except Exception as exc:failure=type(exc).__name__
  good=case=='positive';assert (failure is None)==good,(case,failure);assert bool(launch)==good,(case,launch)
  results.append({'case':case,'accepted':good,'failure':failure,'mock_launch_count':len(launch)})
out={'scope':'actual full dispatcher with isolated root and mocked process/socket primitives; no real service/GPU/Slurm','dispatcher_sha256':digest(src),'cases':results};(root/'work/aux-gate-review/formal-index-dispatch-fixtures.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
