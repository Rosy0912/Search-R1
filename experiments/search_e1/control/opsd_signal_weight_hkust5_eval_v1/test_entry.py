import copy,unittest
from pathlib import Path
import entry
class HostPortTests(unittest.TestCase):
 def test_port_pinned_body(self):
  source=entry.port_source((Path(__file__).parent/'evaluate.py').read_text())
  compile(source,'overlay','exec');self.assertEqual(source.count('str(rank%4)'),2)
  with self.assertRaises(ValueError):entry.port_source(source)
 def test_terminal_positive_and_rejections(self):
  c=dict(accepted=True,training_outer_total=100,groups=list(entry.SHAS),binding=dict(sha256=entry.BINDING_SHA))
  t={g:dict(variant=g,actual_wait_returncode=0,binding=c['binding']) for g in ['H_POS','H_NEG']}
  l={g:dict(accepted=True,outer_iterations=25,rows=[dict(outer_index=i,train_denominator=2560,train_correct=1000) for i in range(25)]) for g in entry.SHAS}
  p='3550866|1|0|$2|%2';s=dict(pid=3535788,uid=1001,starttick=643725164)
  self.assertTrue(entry.check_terminal_values(c,t,l,p,s))
  for scenario in range(7):
   cc,tt,ll,pp,ss=copy.deepcopy((c,t,l,p,s))
   if scenario==0:cc['training_outer_total']=99
   if scenario==1:tt['H_NEG']['actual_wait_returncode']=1
   if scenario==2:ll['H_POS']['rows'][0]['train_denominator']=1280
   if scenario==3:pp='3550866|0||$2|%2'
   if scenario==4:ss['starttick']+=1
   if scenario==5:cc['binding']['sha256']='different'
   if scenario==6:ll['G_POS']['rows'][0]['outer_index']=1
   with self.assertRaises(ValueError):entry.check_terminal_values(cc,tt,ll,pp,ss)
if __name__=='__main__':unittest.main()
