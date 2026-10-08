"""Regression against actual 3279 originals; local only, no GPU or signals."""
import base64,copy,hashlib,importlib.util,json,pathlib,tempfile,unittest
B=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('checker',B/'check-hardware-allocation.py');k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
CAP=pathlib.Path('work/evidence/ray-titlefix-hardware3279-allocation-transfer.json')
CP='work/evidence/formal3231-checkpoint25-fullhash-cpu-candidate.json';REPO='work/bapo-e1-300-ray-titlefix'
def rebound(row,raw):
 row.update(base64=base64.b64encode(raw).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())
class Actual(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.original=json.loads(CAP.read_bytes())
 def check(self,d):
  with tempfile.TemporaryDirectory()as td:
   p=pathlib.Path(td)/'capture.json';p.write_text(json.dumps(d));return k.validate(p,3279,CP,REPO)
 def reject(self,change):
  d=copy.deepcopy(self.original);change(d)
  with self.assertRaises((ValueError,KeyError)):self.check(d)
 def test_original_actual_binding_not_permission(self):
  r=self.check(self.original);self.assertEqual(r['legacy_checkpoint_bound_files'],31);self.assertFalse(r['actual_gpu_accepted']);self.assertFalse(r['permission_signed']);self.assertFalse(r['training_authorized'])
 def test_other_job_rejected(self):self.reject(lambda d:d.update(job_id=3280))
 def test_old_source_rejected(self):self.reject(lambda d:d.update(source_commit='b7119791b86951727cea36a1881e8a525c932494'))
 def test_old_root_rejected(self):self.reject(lambda d:d.update(root='/old/root'))
 def test_runtime_semantic_rebind_rejected(self):
  def change(d):
   row=d['root_inputs']['runtime-executable.json'];j=json.loads(row['text']);j['executable']['inode']+=1;rebound(row,json.dumps(j).encode())
  self.reject(change)
 def test_manifest_legacy_rank_file_rebind_rejected(self):
  def change(d):
   row=d['files']['recovery-manifest.json'];j=json.loads(row['text']);j['ranks'][0]['state']['sha256']='0'*64;rebound(row,json.dumps(j).encode())
  self.reject(change)
 def test_actual_tool_rebind_rejected(self):
  self.reject(lambda d:rebound(d['actual_tools']['resource_watch.py'],b'# arbitrary replacement\n'))
 def test_slurm_restart_even_with_new_raw_sha_rejected(self):
  def change(d):
   row=d['slurm_live'];row['text']=row['text'].replace('Restarts=0','Restarts=1');row['sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
  self.reject(change)
 def test_durable_exit_rebound_nonzero_rejected(self):
  def change(d):
   row=d['root_inputs']['cpu-gate-durable-exit.json'];j=json.loads(row['text']);j['exit_code']=42;rebound(row,json.dumps(j).encode())
  self.reject(change)
if __name__=='__main__':unittest.main(verbosity=2)
