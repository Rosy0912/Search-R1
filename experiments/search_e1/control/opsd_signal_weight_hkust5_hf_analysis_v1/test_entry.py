import copy,json,hashlib,unittest
from pathlib import Path
import entry

class ResourceTests(unittest.TestCase):
 def setUp(self):
  self.cfg={'owner':{'gpu_uuids':['g0','g1','g2','g3']},'resources':{'startup_minimum_free_mib':20480,'training_minimum_free_mib':6144}}
  self.devices=[dict(index=i,uuid='g'+str(i),actual_free_mib=24000) for i in range(4)]
  self.owner={'pid':10,'uid':1001,'run_id':'new'}
 def test_empty_released_and_active_own_leaf(self):
  self.assertEqual(entry.validate_devices(self.cfg,self.devices,[],self.owner,'new',None,None,True),[])
  callback=lambda pid,run,owner,children:dict(pid=pid,uid=1001,run_id=run)
  result=entry.validate_devices(self.cfg,self.devices,[dict(pid=20)],self.owner,'new',None,callback,False)
  self.assertEqual(result[0]['pid'],20)
  with self.assertRaises(ValueError):entry.validate_devices(self.cfg,self.devices,[dict(pid=20)],self.owner,'new',None,callback,True)
 def test_remote_oracle_pid_not_a_local_exemption_and_mapping_floor(self):
  def foreign(*a):raise ValueError('foreign GPU worker')
  with self.assertRaises(ValueError):entry.validate_devices(self.cfg,self.devices,[dict(pid=1749337)],self.owner,'new',None,foreign,False)
  with self.assertRaises(ValueError):entry.validate_devices(self.cfg,self.devices,[dict(pid=10)],self.owner,'new',None,foreign,False)
  for change in ['wronguuid','lowfree','missingcard']:
   dd=copy.deepcopy(self.devices)
   if change=='wronguuid':dd[0]['uuid']='foreign'
   if change=='lowfree':dd[0]['actual_free_mib']=20479
   if change=='missingcard':dd.pop()
   with self.assertRaises(ValueError):entry.validate_devices(self.cfg,dd,[],self.owner,'new',None,None,True)
 def test_scientific_files_byte_frozen(self):
  root=Path(__file__).parent;frozen=json.loads((root/'frozen-files.json').read_text())
  self.assertEqual(len(frozen['original_source_commit']),40)
  for name,sha in frozen['files'].items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),sha)
if __name__=='__main__':unittest.main()
