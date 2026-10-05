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
 def test_small_mlp_fold_is_ineligible_without_discarding_lr(self):
  X=np.arange(24,dtype=float).reshape(12,2);y=np.array([0]+[1]*8+[0,1,1]);groups=np.arange(12)
  class FixedSplitter:
   def __init__(self,*args,**kwargs):pass
   def split(self,*args):yield np.arange(9),np.arange(9,12)
  with patch.object(training,'GroupShuffleSplit',FixedSplitter):
   auc,folds=training.cv_auc(X,y,groups,model='lr');self.assertTrue(np.isfinite(auc));self.assertEqual(len(folds),1)
   with patch.object(training,'make_clf',side_effect=AssertionError('ineligible MLP must not fit')):
    auc,folds=training.cv_auc(X,y,groups,model='mlp64');self.assertTrue(np.isnan(auc));self.assertEqual(folds,[])
 def test_mlp_too_small_internal_validation_is_ineligible(self):
  X=np.arange(24,dtype=float).reshape(12,2);y=np.arange(12)%2;groups=np.arange(12)
  class FixedSplitter:
   def __init__(self,*args,**kwargs):pass
   def split(self,*args):yield np.arange(8),np.arange(8,12)
  with patch.object(training,'GroupShuffleSplit',FixedSplitter),patch.object(training,'make_clf',side_effect=AssertionError('must not fit')):
   auc,_=training.cv_auc(X,y,groups,model='mlp128');self.assertTrue(np.isnan(auc))
 def test_feasible_candidate_programming_error_is_not_hidden(self):
  groups=np.arange(100);y=groups%2;X=np.stack([y,np.sin(groups)],axis=1).astype(float)
  with patch.object(training,'make_clf',side_effect=RuntimeError('unexpected fit bug')):
   with self.assertRaisesRegex(RuntimeError,'unexpected fit bug'):training.cv_auc(X,y,groups,model='mlp64')
 def test_any_infeasible_fold_rejects_whole_mlp_candidate(self):
  X=np.arange(60,dtype=float).reshape(30,2);y=np.arange(30)%2;groups=np.arange(30)
  class FixedSplitter:
   def __init__(self,*args,**kwargs):pass
   def split(self,*args):
    yield np.arange(20),np.arange(20,30)
    yield np.arange(8),np.arange(8,30)
  with patch.object(training,'GroupShuffleSplit',FixedSplitter),patch.object(training,'make_clf',side_effect=AssertionError('must not fit even first fold')):
   auc,folds=training.cv_auc(X,y,groups,model='mlp256');self.assertTrue(np.isnan(auc));self.assertEqual(folds,[])
 def test_strict_cv_rejects_one_degenerate_fold_before_any_fit(self):
  X=np.arange(24,dtype=float).reshape(12,2);y=np.arange(12)%2;groups=np.arange(12)
  class FixedSplitter:
   def __init__(self,*a,**k):pass
   def split(self,*a):
    yield np.arange(8),np.arange(8,12)
    yield np.array([0,1,2,3,5,7,9,11]),np.array([4,6,8,10])
  with patch.object(training,'GroupShuffleSplit',FixedSplitter),patch.object(training,'make_clf',side_effect=AssertionError('must not fit')):
   auc,folds=training.cv_auc(X,y,groups,model='lr',require_all_folds=True)
   self.assertTrue(np.isnan(auc));self.assertEqual(folds,[])
if __name__=='__main__':unittest.main()
