import unittest,importlib.util,pathlib,json,base64,hashlib,copy,tempfile
p=pathlib.Path(__file__).with_name('publish-reviewed-hardware3287.py');s=importlib.util.spec_from_file_location('pub',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def binding(path,raw):return dict(path=str(path),base64=base64.b64encode(raw).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
class Tests(unittest.TestCase):
 def payload(self):
  report=b'Actual hardware report\n';path=m.ROOT/'logs/HARDWARE3287-FINAL.md';r=dict(publication_ready=True,status='PASS_INDEPENDENT_E1_HARDWARE_REVIEW',source_commit=m.SOURCE,root=str(m.ROOT),job_id=3287,run_root=str(m.ROOT/'runs/bapo-instruct300-hardware-3287'),actual_gpu_accepted=True,production_training_permission_signed=False,independent_review=dict(path=str(path),bytes=len(report),sha256=hashlib.sha256(report).hexdigest()),evidence=[dict(path=str(m.ROOT/'logs/x'),bytes=0,sha256=hashlib.sha256(b'').hexdigest())]);return dict(receipt=binding(m.ROOT/'logs/hardware-acceptance.json',json.dumps(r).encode()),report=binding(path,report))
 def mutate(self,key,value):
  p=self.payload();r=json.loads(base64.b64decode(p['receipt']['base64']));r[key]=value;p['receipt']=binding(p['receipt']['path'],json.dumps(r).encode());return p
 def test_original_valid(self):self.assertEqual(m.validate_payload(self.payload())[0]['job_id'],3287)
 def test_not_ready(self):
  with self.assertRaises(ValueError):m.validate_payload(self.mutate('publication_ready',False))
 def test_production_permission(self):
  with self.assertRaises(ValueError):m.validate_payload(self.mutate('production_training_permission_signed',True))
 def test_wrong_job_source_root(self):
  for k,v in [('job_id',True),('job_id',3284),('source_commit','old'),('root','/old')]:
   with self.subTest(k=k,v=v),self.assertRaises(ValueError):m.validate_payload(self.mutate(k,v))
 def test_report_corrupt(self):
  p=self.payload();p['report']['base64']=base64.b64encode(b'bad').decode()
  with self.assertRaises(ValueError):m.validate_payload(p)
 def test_rebound_wrong_report(self):
  p=self.payload();p['report']=binding(p['report']['path'],b'rebound')
  with self.assertRaises(ValueError):m.validate_payload(p)
 def test_external_evidence(self):
  with self.assertRaises(ValueError):m.validate_payload(self.mutate('evidence',[dict(path='/foreign/x',bytes=0,sha256='0'*64)]))
 def test_duplicate_evidence(self):
  p=self.payload();r=json.loads(base64.b64decode(p['receipt']['base64']));r['evidence']*=2;p['receipt']=binding(p['receipt']['path'],json.dumps(r).encode())
  with self.assertRaises(ValueError):m.validate_payload(p)
 def test_exclusive_existing_receipt(self):
  old=m.ROOT
  try:
   with tempfile.TemporaryDirectory()as d:
    m.ROOT=pathlib.Path(d);(m.ROOT/'logs').mkdir();(m.ROOT/'logs/hardware-acceptance.json').write_bytes(b'existing');p=self.payload()
    with self.assertRaises(ValueError):m.publish(p)
    self.assertEqual((m.ROOT/'logs/hardware-acceptance.json').read_bytes(),b'existing')
  finally:m.ROOT=old
if __name__=='__main__':unittest.main()
