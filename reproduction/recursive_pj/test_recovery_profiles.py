import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import resume_transport as transport
from .resume_profiles import profile
from .test_resume_transport import NativeTransport
from .migration_job import storage_admission


class RecoveryProfiles(unittest.TestCase):
    def test_registered_profiles_and_unknown_rejected(self):
        self.assertEqual(profile('depth2-v1')['step'], 56)
        self.assertEqual(profile('budget5-v2')['step'], 2)
        self.assertNotEqual(profile('budget5-v2')['target_job'], profile('budget5-v1')['target_job'])
        with self.assertRaises(ValueError): profile('position-early')

    def test_depth56_rejects_reset_and_incomplete_state(self):
        p=profile('depth2-v1')
        with patch.multiple(transport, GROUP=p['group'], STEP=p['step'],
                PARENT_JOB=p['parent_job'], PARENT_SOURCE=p['parent_source'],
                RUN_ID=p['run_id'], CONFIG_SHA=p['config_sha']):
            b,r,m=NativeTransport().fixture()
            r.update(group=p['group'],run_id=p['run_id'],source_commit=p['parent_source'],config_sha256=p['config_sha'])
            b.update(parent_job=p['parent_job'],completed_outer=56)
            m['identity'].update(job_id=p['parent_job'],code_commit=p['parent_source'],config_sha256=p['config_sha'],completed_step=56)
            for w in m['worker_states']:
                w['identity'].update(m['identity'])
                w.update(actual_total_optimizer_calls=1120,scheduler_last_epoch=56)
            m['driver']['next_step']=57
            self.assertTrue(transport.validate_inputs(b,r,m))
            m['worker_states'][4]['actual_total_optimizer_calls']=0
            with self.assertRaises(ValueError):transport.validate_inputs(b,r,m)
            m['worker_states'][4]['actual_total_optimizer_calls']=1120
            m['driver']['next_step']=1
            with self.assertRaises(ValueError):transport.validate_inputs(b,r,m)

    def test_reentry_must_match_parent_and_cannot_overwrite_archive(self):
        p=profile('depth2-v1')
        with TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'receipts').mkdir()
            previous=base/'receipts/current-entry.json'
            previous.write_text(json.dumps(dict(allocation=p['parent_job'],run_id=p['run_id'])))
            binding=dict(parent_entry_sha256=transport.sha(previous))
            (base/'cutover.json').write_text('{}')
            (base/('result-'+p['target_job']+'.json')).write_text('{}')
            record=dict(fresh_authorized=False,allocation=p['target_job'])
            with patch.multiple(transport, GROUP=p['group'],PARENT_JOB=p['parent_job'],
                    RUN_ID=p['run_id'],TARGET_JOB=p['target_job'],STAGE=base), \
                    patch.object(transport,'load_inputs',return_value=(binding,{},{})), \
                    patch.object(transport,'validate_gate'):
                with self.assertRaises(ValueError):transport.validate_resume_entry(dict(record,fresh_authorized=True),base)
                transport.validate_resume_entry(record,base)
                self.assertTrue((base/'receipts'/('previous-entry-'+p['parent_job']+'.json')).exists())
                with self.assertRaises(FileExistsError):transport.validate_resume_entry(record,base)
                previous.write_text('{}')
                with self.assertRaises(ValueError):transport.validate_resume_entry(record,base)

    def test_storage_admission_uses_available_not_total(self):
        with patch('os.statvfs',return_value=SimpleNamespace(f_bavail=0,f_frsize=4096)):
            with self.assertRaises(RuntimeError):storage_admission('/unused')
        with patch('os.statvfs',return_value=SimpleNamespace(f_bavail=10**9,f_frsize=4096)):
            self.assertEqual(storage_admission('/unused'),4096*10**9)

    def test_reject_duplicate_native_member_names(self):
        b,r,m=NativeTransport().fixture()
        b['members'][1]=dict(b['members'][0])
        with self.assertRaises(ValueError):transport.validate_inputs(b,r,m)


if __name__=='__main__':unittest.main()
