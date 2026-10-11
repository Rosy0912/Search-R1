import unittest
from unittest.mock import patch
from search_r1.baselines import recursive_admission_hkust5 as a
class Response:
 def read(self,*args):
  import json
  return json.dumps(a.EXPECTED_HEALTH).encode()
class Admission(unittest.TestCase):
 def run_gate(self,rows='',host='xie-lab-4',selected=None,inventory=None):
  selected=selected or ['0','1','2','3']; inventory=inventory or {str(i):{'uuid':u} for i,u in enumerate(a.UUIDS)}
  with patch.object(a.socket,'gethostname',return_value=host),patch.object(a.os,'getuid',return_value=1001),patch.object(a.subprocess,'check_output',return_value=rows),patch.object(a.urllib.request,'urlopen',return_value=Response()):return a.exact_gpu_gate(selected,inventory)
 def test_empty_owned_cards_remote_oracle(self):self.assertTrue(self.run_gate()['ready'])
 def test_foreign_card4_unknown_ignored(self):self.assertTrue(self.run_gate('OTHER, [N/A], [N/A]')['ready'])
 def test_any_selected_process_refused(self):
  with self.assertRaises(RuntimeError):self.run_gate(a.UUIDS[0]+', 1749337, 1')
 def test_wrong_host_refused(self):
  with self.assertRaises(RuntimeError):self.run_gate(host='xielab5')
 def test_wrong_scope_refused(self):
  with self.assertRaises(RuntimeError):self.run_gate(selected=['4','5','6','7'])
 def test_uuid_mismatch_refused(self):
  inv={str(i):{'uuid':u} for i,u in enumerate(a.UUIDS)};inv['0']['uuid']='OTHER'
  with self.assertRaises(RuntimeError):self.run_gate(inventory=inv)
if __name__=='__main__':unittest.main()
