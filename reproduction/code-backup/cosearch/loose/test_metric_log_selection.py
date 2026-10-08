import ast, hashlib, json, tempfile, unittest
from pathlib import Path
source=Path(__file__).with_name('metric_log_source.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='validated_metric_log')
namespace=dict(json=json,hashlib=hashlib,Path=Path)
exec(compile(ast.Module(body=[node],type_ignores=[]),'validated_metric_log','exec'),namespace)
select=namespace['validated_metric_log']
class MetricLogTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.run=self.root/'runs/n';self.run.mkdir(parents=True)
  (self.run/'training.log').write_text('COSEARCH_OUTER_METRICS_JSON={"step":1}\n')
  self.worker=self.run/'worker-training.log';self.worker.write_text(''.join('COSEARCH_OUTER_METRICS_JSON='+json.dumps({'step':i})+'\n' for i in range(1,6)))
  session=self.root/'session_bound';logs=session/'logs';logs.mkdir(parents=True);original=logs/'worker-owned-11.out';original.write_bytes(self.worker.read_bytes());st=original.stat()
  (self.run/'launch.json').write_text(json.dumps({'job_id':'3002'}))
  self.meta=dict(job=3002,source=str(original),worker_pid=11,destination=str(self.worker),bytes=self.worker.stat().st_size,sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest(),process_cgroup_verified=True,process_start_ticks=123,cgroup='/job_3002/',session=str(session),source_inode=st.st_ino,source_device=st.st_dev,live_binding_at=1,captured_at=2)
  self.save()
 def save(self): (self.run/'worker-log-provenance.json').write_text(json.dumps(self.meta))
 def tearDown(self): self.temp.cleanup()
 def test_partial_forwarded_log_does_not_hide_complete_bound_source(self):
  text,label=select(self.run,self.root);self.assertEqual(label,'worker-training.log');self.assertEqual([json.loads(l.split('=',1)[1])['step'] for l in text.splitlines()],[1,2,3,4,5])
 def test_tampered_copy_refused_without_silent_driver_fallback(self):
  self.worker.write_bytes(self.worker.read_bytes()+b'forged');self.assertRaises(AssertionError,select,self.run,self.root)
 def test_other_job_refused(self):
  self.meta['job']=999;self.save();self.assertRaises(AssertionError,select,self.run,self.root)
 def test_no_worker_uses_driver(self):
  self.worker.unlink();text,label=select(self.run,self.root);self.assertEqual(label,'training.log');self.assertIn('"step":1',text)
if __name__=='__main__': unittest.main()
