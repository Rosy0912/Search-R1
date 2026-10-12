"""Small scalar byte seals; no actual model/training acceptance is claimed."""
import tempfile
import unittest
from pathlib import Path

from reproduction.common8.actor_probe_evidence_seals import fixed_logs,verify_fixed_logs,sealed_digest


class EvidenceSealTests(unittest.TestCase):
    def test_original_prefix_allows_append_but_detects_replacement_or_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();path=base/'gpu.jsonl';prefix=b'{"fixture":1}\n'
            path.write_bytes(prefix);saved=sealed_digest(base,path,prefix_bytes=len(prefix))
            with path.open('ab') as stream:stream.write(b'{"fixture":2}\n')
            self.assertEqual(sealed_digest(base,path,prefix_bytes=len(prefix)),saved)
            path.write_bytes(b'{"fixture":9}\n'+b'{"fixture":2}\n')
            self.assertNotEqual(sealed_digest(base,path,prefix_bytes=len(prefix)),saved)
            path.write_bytes(prefix[:5])
            with self.assertRaises(AssertionError):sealed_digest(base,path,prefix_bytes=len(prefix))

    def test_completed_evidence_is_sealed_while_pipeline_inventory_may_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();logs=base/'logs';logs.mkdir()
            training=logs/'training.log';training.write_text('completed scalar stand-in\n')
            gate=logs/'gates'/'attempt.json';gate.parent.mkdir();gate.write_text('{"fixture":1}\n')
            dynamic=logs/'process-ledger.json';dynamic.write_text('{"fixture":"verifier live"}\n')
            gpu=logs/'gpu-samples-fixture.jsonl';gpu.write_text('{"fixture":1}\n')
            run={'train_log':str(training)};saved=fixed_logs(base,run)
            self.assertEqual(set(saved),{'logs/training.log','logs/gates/attempt.json'})
            dynamic.write_text('{"fixture":"cleanup finished"}\n');gpu.write_text('{"fixture":2}\n')
            self.assertEqual(verify_fixed_logs(base,run,saved),2)
            gate.write_text('{"fixture":2}\n')
            with self.assertRaises(AssertionError):verify_fixed_logs(base,run,saved)

    def test_added_evidence_and_outside_or_symlink_bytes_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'logs').mkdir()
            training=base/'logs/training.log';training.write_text('scalar fixture')
            run={'train_log':str(training)};saved=fixed_logs(base,run)
            added=base/'logs/failure.json';added.write_text('{"fixture":"late evidence"}')
            with self.assertRaises(AssertionError):verify_fixed_logs(base,run,saved)
            link=base/'logs/link.json';link.symlink_to(added)
            with self.assertRaises(AssertionError):sealed_digest(base,link)
            with self.assertRaises(AssertionError):sealed_digest(base/'logs/gates',added)


if __name__=='__main__':unittest.main()
