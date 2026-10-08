import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from reproduction.a800.run_pipeline import r1_finished

RUN='cc437e7315c94fccba5750433ff850b1'
class HandoffTests(unittest.TestCase):
    def test_both_terminal_paths_require_preserved_files_and_actual_updates(self):
        for rejected in (False,True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);ck=root/'checkpoint';ck.mkdir();(ck/'weights').write_bytes(b'weights')
                evidence=dict(checkpoint=str(ck),file_sha256={'weights':hashlib.sha256(b'weights').hexdigest()})
                proof=dict(run_id=RUN,target_update=300,completed_updates=list(range(101,301)))
                if rejected:proof['checkpoint_evidence']=evidence
                else:proof.update(evidence)
                state='STOPPED_AFTER_REJECTED_NONFINITE_CHECKPOINT' if rejected else 'STOPPED_AFTER_VERIFIED_CHECKPOINT'
                name='r1-update300-failed-snapshot.json' if rejected else 'r1-update300-acceptance.json'
                (root/'r1-stop300-status.json').write_text(json.dumps(dict(run_id=RUN,state=state)))
                (root/name).write_text(json.dumps(proof));self.assertTrue(r1_finished(root))
                proof['completed_updates'].pop();(root/name).write_text(json.dumps(proof))
                with self.assertRaises(AssertionError):r1_finished(root)
                proof['completed_updates'].append(300);(root/name).write_text(json.dumps(proof));(ck/'weights').write_bytes(b'changed')
                with self.assertRaises(AssertionError):r1_finished(root)
    def test_nonterminal_markers_cannot_launch_g1(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'r1-stop300-status.json').write_text(json.dumps(dict(run_id=RUN,state='CHECKPOINT_REJECTED_COORDINATOR_LEFT_SUSPENDED')))
            self.assertFalse(r1_finished(root))

if __name__=='__main__':unittest.main()
