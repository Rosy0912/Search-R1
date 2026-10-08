import pathlib,sys,tempfile,types,runpy,json,time,hashlib
from unittest.mock import patch
r=pathlib.Path(__file__).resolve().parents[2];source=r/'work/probe_busy_hybrid.py';rows=[];sleep=time.sleep
for mode in ('success','concurrent_health_error','post_error','final_health_error'):
 with tempfile.TemporaryDirectory() as temp:
  t=pathlib.Path(temp);base=t/'base.json';base.write_text(json.dumps({'queries':['public'+str(i) for i in range(14)]}));out=t/'report.json';state={'calls':0,'failure_saved_before_worker_end':False};expected={'pid':42}
  class Reply:
   def __init__(self,value):self.value=value;self.content=json.dumps(value).encode()
   def raise_for_status(self):pass
   def json(self):return self.value
  def get(url,timeout):
   state['calls']+=1
   if mode=='concurrent_health_error' and state['calls']==2:raise TimeoutError('fixture health timeout')
   frame=sys._getframe(1)
   if mode=='final_health_error' and frame.f_globals.get('phase')=='retrieve-result':raise RuntimeError('fixture final provenance changed')
   return Reply(expected)
  def post(url,json,timeout):
   if mode=='post_error':raise TimeoutError('fixture POST timeout')
   sleep(.12)
   if mode=='concurrent_health_error':state['failure_saved_before_worker_end']=out.exists()
   return Reply({'result':[[{'document':{'id':str(i)},'score':float(i)} for i in range(3)] for q in json['queries']]})
  sys.argv=[str(source),'--baseline',str(base),'--output',str(out)]
  error=None
  with patch.dict(sys.modules,{'requests':types.SimpleNamespace(get=get,post=post)}),patch('time.sleep',lambda seconds:sleep(.02)):
   try:runpy.run_path(str(source),run_name='__main__')
   except Exception as e:error=repr(e)
  report=json.loads(out.read_text()) if out.exists() else None;samples=[json.loads(l) for l in out.with_suffix('.samples.jsonl').read_text().splitlines()]
  if mode=='success':assert report['status']=='BUSY_HEALTH_MEASUREMENT_ONLY' and len(samples)>=2
  if mode in ('concurrent_health_error','post_error'):assert report['status']=='FAILED_BUSY_HEALTH_MEASUREMENT'
  if mode=='concurrent_health_error':assert state['failure_saved_before_worker_end']
  rows.append({'case':mode,'error':error,'report_status':report['status'] if report else None,'phase':report.get('phase') if report else None,'samples_persisted':len(samples),'failure_saved_before_worker_end':state['failure_saved_before_worker_end']})
(r/'work/aux-gate-review/busy-persistence-fixtures.json').write_text(json.dumps({'scope':'actual script run with mocked HTTP/public synthetic fixture, short real threads/sleeps; no actual service/model/GPU','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
