import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import formal_audit_failure718650 as audit


class FailurePreservationTests(unittest.TestCase):
    def test_reject_nonterminal_or_other_owner(self):
        audit.terminal('|'.join(audit.TERMINAL)+'\n')
        for index,value in ((1,'RUNNING'),(2,'0:0'),(3,'mou176'),(4,'21930'),(10,'ACD1-31')):
            row=list(audit.TERMINAL);row[index]=value
            with self.assertRaisesRegex(ValueError,'EXACT_FAILED_OWNER_ALLOCATION_REQUIRED'):
                audit.terminal('|'.join(row)+'\n')

    def preserve(self, corrupt=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        base=Path(temp.name);checkpoint=base/'checkpoints/actor/global_step_5'
        checkpoint.mkdir(parents=True)
        payload=checkpoint/'state.bin';payload.write_bytes(b'actual-saved-state')
        sha=hashlib.sha256(payload.read_bytes()).hexdigest()
        proof=dict(files_sha256={'state.bin':'0'*64 if corrupt else sha})
        (checkpoint/'complete.json').write_text(json.dumps(proof))
        original={str(p):p.read_bytes() for p in checkpoint.iterdir()}
        with patch.multiple(audit,BASE=base,CHECKPOINT=checkpoint,ARCHIVE=base/'failed-job',
                            OWNER=payload.stat().st_uid), \
                patch.object(audit,'metadata',return_value={'completed_pair_available':False}), \
                patch.object(audit,'verify_saved_prefix',return_value={'actual_cuda_restore_accepted':False}):
            if corrupt:
                with self.assertRaisesRegex(ValueError,'ORIGINAL_CHECKPOINT_SHA_DIFFERS'):audit.archive()
                self.assertTrue((base/'failed-job/terminal.json').is_file())
                self.assertFalse((base/'failed-job/terminal-and-files.json').exists())
            else:
                value=audit.archive()
                self.assertEqual(value['checkpoint_file_count'],2)
                saved=audit.read(base/'failed-job/terminal-and-files.json')
                self.assertFalse(saved['completed_pair_available'])
                self.assertFalse(saved['cpu_state_validation_pending'])
                self.assertFalse(saved['saved_prefix_cpu_validation']['actual_cuda_restore_accepted'])
                self.assertEqual(saved['full_read_bytes'],sum(len(x) for x in original.values()))
                self.assertEqual((base/'failed-job/small-evidence/checkpoints/actor/global_step_5/state.bin').read_bytes(),
                                 payload.read_bytes())
                with self.assertRaisesRegex(ValueError,'ALREADY_ATTEMPTED_INSPECT_DO_NOT_REPLAY'):
                    audit.archive()
        for name,data in original.items():self.assertEqual(Path(name).read_bytes(),data)

    def test_full_sha_independent_copy_without_pair_claim_or_replay(self):
        self.preserve()

    def test_changed_checkpoint_preserves_failure_but_cannot_complete(self):
        self.preserve(corrupt=True)


if __name__=='__main__':unittest.main()
