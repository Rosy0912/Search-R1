"""CPU regression with synthetic arrays; never published as fitted experiment heads."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
import ts_probe_train as training
from sklearn.model_selection import GroupShuffleSplit

class CalibrationData(unittest.TestCase):
 def test_empty_searched_file_still_preserves_boundary_questions(self):
  with tempfile.TemporaryDirectory() as directory:
   common=dict(boundary_q_cb=np.array([0,1]),boundary_question_id=np.array(['q0','q1']),Xp_question_21=np.ones((2,3)))
   np.savez(Path(directory)/'empty.npz',y_em=np.array([]),**common)
   full=dict(common);full.update(y_em=np.array([1]),y_flip=np.array([1]),q_cb=np.array([0]),question_id=np.array(['q2']),ds=np.array(['nq']),boundary_q_cb=np.array([0]),boundary_question_id=np.array(['q2']),Xp_question_21=np.ones((1,3)))
   for name in ('Xp','Xa','Xc'):full[name+'_21']=np.ones((1,3))
   np.savez(Path(directory)/'full.npz',**full)
   result=training.load_dumps([directory],[21]);self.assertEqual(len(result[1]),1)
   self.assertEqual(set(result[-1][2]),{'q0','q1','q2'});self.assertEqual(len(result[-1][1]),3)
 def test_fitting_cv_keeps_same_question_trajectories_together(self):
  groups=np.repeat(np.arange(60),5);y=groups%2;X=np.stack([y,np.sin(groups)],axis=1).astype(float)
  checked=[]
  class CheckedSplitter:
   def __init__(self,*args,**kwargs):self.delegate=GroupShuffleSplit(*args,**kwargs)
   def split(self,X,y,groups):
    for a,b in self.delegate.split(X,y,groups):
     assert not set(groups[a])&set(groups[b]);checked.append(True);yield a,b
  with patch.object(training,'GroupShuffleSplit',CheckedSplitter):auc,_=training.cv_auc(X,y,groups,n=2)
  self.assertEqual(len(checked),2);self.assertTrue(np.isfinite(auc))
  degenerate,_=training.cv_auc(X,np.zeros_like(y),groups);self.assertTrue(np.isnan(degenerate))
if __name__=='__main__':unittest.main()
