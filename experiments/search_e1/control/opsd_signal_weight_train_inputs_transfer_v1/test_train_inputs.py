import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import train_inputs as m


class Contracts(unittest.TestCase):
    def ledger(self):
        cfg = {'run_id': 'fixed'}
        rows = [dict(accepted=True, variant='G_POS', run_id='fixed', outer_index=i,
                     config_sha256='sha', train_denominator=2560, train_correct=1000,
                     train_em=1000/2560, train_post_update=False) for i in range(25)]
        return cfg, dict(accepted=True, variant='G_POS', run_id='fixed', rows=rows)

    def test_complete_preupdate_cohort(self):
        cfg, ledger = self.ledger()
        self.assertEqual(len(m.completed_rows(ledger, 'G_POS', cfg, 'sha')), 25)

    def test_wrong_sampling_denominator(self):
        cfg, ledger = self.ledger(); ledger['rows'][2]['train_denominator'] = 1280
        with self.assertRaises(ValueError): m.completed_rows(ledger, 'G_POS', cfg, 'sha')

    def test_postupdate_must_not_masquerade(self):
        cfg, ledger = self.ledger(); ledger['rows'][2]['train_post_update'] = True
        with self.assertRaises(ValueError): m.completed_rows(ledger, 'G_POS', cfg, 'sha')

    def test_missing_or_repeated_outer(self):
        cfg, ledger = self.ledger(); ledger['rows'][2]['outer_index'] = 1
        with self.assertRaises(ValueError): m.completed_rows(ledger, 'G_POS', cfg, 'sha')

    def test_path_traversal_and_non_G(self):
        for name in [str(m.S/'group-H_POS/foo'), str(m.S/'group-G_POS/../group-G_NEG/foo'), str(m.S/'group-G_POS/a\nb')]:
            with self.assertRaises(ValueError): m.safe(name)


class Installation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); base = Path(self.tmp.name).resolve()
        self.source, self.target, self.stage = base/'source', base/'target', base/'stage'
        self.source.mkdir(); self.target.mkdir(); self.stage.mkdir()
        self.patch_s=patch.object(m, 'S', self.source); self.patch_t=patch.object(m, 'T', self.target)
        self.patch_s.start(); self.patch_t.start()
        files=[]
        for group in m.CONFIGS:
            p=self.source/('group-'+group)/'runs/tokens.jsonl'; p.parent.mkdir(parents=True); p.write_bytes(b'fixed-token-record\n')
            ref=m.fresh_ref(p); files.append(ref)
            src=self.stage/str(p).lstrip('/'); src.parent.mkdir(parents=True); src.write_bytes(p.read_bytes())
        self.value=dict(schema=m.SCHEMA,accepted=True,groups=[dict(variant=g,completed_outers=25,config={'sha256':sha}) for g,sha in m.CONFIGS.items()],files=files,member_bytes=sum(r['bytes'] for r in files),scope='test')

    def tearDown(self):
        self.patch_s.stop(); self.patch_t.stop(); self.tmp.cleanup()

    def dest(self, ref): return self.target/str(ref['path']).lstrip('/')

    def test_verified_install_and_identical_reuse(self):
        self.assertTrue(m.install(self.value,self.stage)['accepted'])
        self.assertTrue(m.install(self.value,self.stage)['accepted'])
        for ref in self.value['files']: self.assertEqual(m.digest(self.dest(ref)),ref['sha256'])

    def test_existing_difference_rejects_before_any_write(self):
        last=self.dest(self.value['files'][-1]); last.parent.mkdir(parents=True);last.write_bytes(b'different')
        with self.assertRaises(ValueError): m.install(self.value,self.stage)
        self.assertFalse(self.dest(self.value['files'][0]).exists());self.assertEqual(last.read_bytes(),b'different')

    def test_corrupt_stage_rejects(self):
        p=self.stage/str(self.value['files'][0]['path']).lstrip('/');p.write_bytes(b'corrupt')
        with self.assertRaises(ValueError): m.install(self.value,self.stage)

    def test_duplicate_inventory_rejects(self):
        value=copy.deepcopy(self.value);value['files'].append(value['files'][0]);value['member_bytes']+=value['files'][0]['bytes']
        with self.assertRaises(ValueError):m.install(value,self.stage)

    def test_symlink_stage_rejects(self):
        p=self.stage/str(self.value['files'][0]['path']).lstrip('/');p.unlink();p.symlink_to(Path(self.value['files'][0]['path']))
        with self.assertRaises(ValueError):m.install(self.value,self.stage)


if __name__ == '__main__': unittest.main()
