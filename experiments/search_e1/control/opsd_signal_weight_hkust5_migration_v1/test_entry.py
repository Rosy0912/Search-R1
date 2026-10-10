import copy,importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('migration',Path(__file__).with_name('entry.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_only_owner_changes(self):
  x=dict(variant='H_POS',owner=dict(uid=1001,physical_gpus=[4,5,6,7],gpu_uuids=['old']*4),scientific=dict(global_batch=32,gradient_accumulation_full_batch_per_rank=8,learning_rate=1e-6),signal=dict(amplitude=.9),run_id='same',paths=dict(model='source100'))
  before=copy.deepcopy(x);m.effective(x);before['owner']=x['owner'];self.assertEqual(x,before);self.assertEqual(x['owner']['physical_gpus'],[0,1,2,3])
 def test_inventory_unknown_owned_rejected(self):
  gpu='\n'.join('%s,%s,40000,49140'%(i,u) for i,u in enumerate(m.UUIDS))
  with patch.object(m.subprocess,'check_output',side_effect=[gpu,m.UUIDS[0]+', [N/A], [N/A]']):
   with self.assertRaises(ValueError):m.inventory()
 def test_actual_mapping_checked(self):
  gpu='\n'.join('%s,%s,40000,49140'%(i,u) for i,u in enumerate(m.UUIDS))
  with patch.object(m.subprocess,'check_output',side_effect=[gpu,'']):
   ds,apps=m.inventory();self.assertEqual(len(ds),4);self.assertFalse(apps)
 def test_exact_native_dispatch_and_static_port(self):
  captured=[]
  def original(command,*args,**kwargs):captured.append((command,kwargs));return 'actual_handle'
  with patch.object(m.subprocess,'Popen',original):
   with m.dispatch_context(dict(path='/binding',sha256='a'*64)):
    cmd=['/python','-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(m.NATIVE/'trainer.py'),'--config','/sameconfig']
    result=m.subprocess.Popen(cmd,env={'CUDA_VISIBLE_DEVICES':'4,5,6,7','E1_RUN_ID':'same'})
    self.assertEqual(result,'actual_handle');self.assertIn('--rdzv_backend=static',captured[0][0]);self.assertEqual(captured[0][1]['env']['CUDA_VISIBLE_DEVICES'],'0,1,2,3');self.assertEqual(captured[0][1]['env']['E1_RUN_ID'],'same')
 def test_foreign_dispatch_unchanged(self):
  captured=[]
  with patch.object(m.subprocess,'Popen',lambda command,**kwargs:captured.append((command,kwargs))):
   with m.dispatch_context(dict(path='/binding',sha256='a'*64)):m.subprocess.Popen(['/foreign','arg'],env={'CUDA_VISIBLE_DEVICES':'4,5,6,7'})
  self.assertEqual(captured,[(['/foreign','arg'],dict(env={'CUDA_VISIBLE_DEVICES':'4,5,6,7'}))])
if __name__=='__main__':unittest.main()
