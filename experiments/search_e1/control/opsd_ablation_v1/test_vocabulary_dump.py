"""Capacity survives an interrupted original-forward evidence write."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vocabulary_dump import VocabularyDump


class VocabularyCapacity(unittest.TestCase):
    def payload(self):
        import torch
        logp=torch.log_softmax(torch.tensor([.2,-.7,1.3],dtype=torch.float32),0)
        return dict(token_index=3,**{k:logp for k in ('student_logp','teacher_logp','outer_old_logp','target_logp')})

    def test_actual_sha_and_cap_skip(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';rows=[]
            VocabularyDump(root,Path(tmp)/'case',dict(run='test'),rows.append)(self.payload())
            complete=rows[0];self.assertEqual(complete['full_vocab_dump'],'complete')
            self.assertEqual(len(complete['files']),4)
            for proof in complete['files']:
                self.assertEqual(hashlib.sha256(Path(proof['path']).read_bytes()).hexdigest(),proof['sha256'])
            ledger=json.loads((root/'ledger.json').read_text());ledger['bytes']=24*(1<<30)
            (root/'ledger.json').write_text(json.dumps(ledger))
            VocabularyDump(root,Path(tmp)/'second',dict(run='test'),rows.append)(self.payload())
            self.assertEqual(rows[-1]['full_vocab_dump'],'cap_skip')
            self.assertFalse((Path(tmp)/'second').exists())

    def test_interrupted_write_keeps_conservative_reservation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study';real_save=np.save;calls=[0]
            def interrupted(*args,**kwargs):
                calls[0]+=1
                if calls[0]==2:raise OSError('simulated interrupted evidence write')
                return real_save(*args,**kwargs)
            with patch.object(np,'save',side_effect=interrupted):
                with self.assertRaises(OSError):
                    VocabularyDump(root,Path(tmp)/'case',dict(run='test'),lambda x:None)(self.payload())
            ledger=json.loads((root/'ledger.json').read_text())
            self.assertEqual(ledger['reservations'][0]['status'],'reserved')
            self.assertGreaterEqual(ledger['bytes'],sum(p.stat().st_size for p in (Path(tmp)/'case').glob('*.npy')))


if __name__=='__main__':unittest.main()
