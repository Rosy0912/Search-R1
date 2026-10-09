"""One sandbox-only check of new multi-target orchestration; original unlink engine reused."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('_compact_test', Path(__file__).with_name('compact.py'))
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)


class MutationBoundary(unittest.TestCase):
    def test_reject_changed_or_open_before_unlink_and_restart_exact_plan(self):
        e = C.engines(); p, t = e['p'], e['t']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); out = root/'plan'; out.mkdir()
            files = [root/n for n in ('loose-a','loose-b','original.tar')]
            for index, f in enumerate(files): f.write_bytes(bytes([index])*8)
            keep = root/'keep.tar.xz'; keep.write_bytes(b'exact retained representation')
            preserve = [p.fileproof(keep)]; targets = [p.fileproof(f) for f in files]
            pref = p.write_json(out/'plan.json', dict(targets=targets, preserve=preserve))
            files[1].write_bytes(b'changed')
            with patch.object(p, 'no_open_files'):
                with self.assertRaises(RuntimeError): C.reclaim_targets(p,t,out,pref,targets,preserve)
            self.assertTrue(all(f.exists() for f in files)); self.assertFalse((out/'execution-intent.json').exists())
            targets = [p.fileproof(f) for f in files]
            pref = p.write_json(out/'accepted-plan.json', dict(targets=targets, preserve=preserve))
            with patch.object(p, 'no_open_files', side_effect=RuntimeError('open/uncertain')):
                with self.assertRaises(RuntimeError): C.reclaim_targets(p,t,out,pref,targets,preserve)
            self.assertTrue(all(f.exists() for f in files)); self.assertFalse((out/'execution-intent.json').exists())
            # no-open capability is mocked only in this sandbox; real CLI never bypasses lsof.
            with patch.object(p, 'no_open_files'):
                C.reclaim_targets(p,t,out,pref,targets,preserve)
                C.reclaim_targets(p,t,out,pref,targets,preserve)
            self.assertTrue(all(not f.exists() for f in files)); p.stable(preserve[0])
            self.assertEqual(len(list((out/'journal').glob('*-done.json'))), 3)
            self.assertTrue(out.is_dir()); self.assertTrue(keep.exists())


if __name__ == '__main__': unittest.main()
