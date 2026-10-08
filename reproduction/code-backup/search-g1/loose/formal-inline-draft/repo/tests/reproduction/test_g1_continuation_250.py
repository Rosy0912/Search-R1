import json,tempfile,unittest
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from reproduction.a800.recovery_binding import finite_fitted_values
from reproduction.a800.prepare_continuation import lineage_lock_root
from reproduction.a800.prepare_run import digest
from reproduction.a800.verify_continuation import bind_segment_versions

class ThirdSegment(unittest.TestCase):
    def test_pipeline_nonfinite_weights_rejected(self):
        clf=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(2,),max_iter=3,random_state=0)).fit(np.array([[0.,0.],[1.,2.],[2.,1.],[3.,3.]]),[0,0,1,1])
        finite_fitted_values(vars(clf))
        clf[-1].coefs_[0][0,0]=np.nan
        with self.assertRaises(AssertionError):finite_fitted_values(vars(clf))
    def test_pipeline_nonfinite_scaler_rejected(self):
        clf=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(2,),max_iter=3,random_state=0)).fit(np.array([[0.,0.],[1.,2.],[2.,1.],[3.,3.]]),[0,0,1,1])
        clf[0].scale_[0]=np.inf
        with self.assertRaises(AssertionError):finite_fitted_values(vars(clf))
    def test_historical_chain_uses_root_lock_and_binds_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'root';middle=Path(t)/'middle'
            for b in (root,middle):(b/'logs').mkdir(parents=True)
            (root/'logs/run-manifest.json').write_text(json.dumps({'run_id':'old'}))
            (middle/'logs/run-manifest.json').write_text(json.dumps({'run_id':'middle','completed_updates':200,'parent':{'parent_base':str(root),'parent_run_id':'old','parent_manifest_sha256':digest(root/'logs/run-manifest.json')}}))
            self.assertEqual(lineage_lock_root(middle),root)
            (root/'logs/run-manifest.json').write_text(json.dumps({'run_id':'tampered'}))
            with self.assertRaises(AssertionError):lineage_lock_root(middle)
    def test_251_to_300_evaluator_and_failed_boundary(self):
        run={'run_id':'last','initial_probe_version':'last:250:real'}
        rows=[{'step':x,'probe_version':'last:250:real'} for x in range(251,301)]
        bind_segment_versions(rows,run,250,[],[])
        rows[4]['probe_version']='other:250:fake'
        with self.assertRaises(AssertionError):bind_segment_versions(rows,run,250,[],[])
        with self.assertRaises(AssertionError):bind_segment_versions([],run,250,['last:300:fake'],[])

if __name__=='__main__':unittest.main()
