import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


reader = load('read_parallel_progress'); builder = load('build_progress_request')


class Progress(unittest.TestCase):
    def test_current_native_cohort_request_binds_exact_six_sources(self):
        r = builder.build(HERE.parent / 'parallel_selected/cohort-config.json')
        self.assertEqual(r['global_selected_steps'], [1,20,40,60,80,100,120,140])
        self.assertEqual([p['gpu_index'] for p in r['producers']], [0,1,2,3,5,6])
        self.assertEqual(sum(len(p['selected_steps']) for p in r['producers']), 7)

    def test_mixed_keeps_old100and20and40_once(self):
        r = builder.mixed(HERE.parent / 'parallel_selected/cohort-config.json', HERE.parent / 'parallel_recovery_v1/cohort-config.json')
        self.assertEqual(sorted(s for p in r['producers'] for s in p['selected_steps']), [20,40,60,80,100,120,140])
        self.assertEqual([p['selected_steps'] for p in r['producers'][-2:]], [[100],[20,40]])
        self.assertEqual(len({p['producer_id'] for p in r['producers']}), 6)
        self.assertTrue(all('logs_directory' in p for p in r['producers']))

    def test_log_tail_returns_only_markers_hash_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'stderr'; p.write_text('Traceback\nprivate raw question\nHold: failure\n')
            r = reader.compact_log(p)
            self.assertEqual(r['error_markers'], ['Traceback', 'Hold:'])
            self.assertNotIn('private', json.dumps(r))

    def test_partial_last_json_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'log'; p.write_bytes(b'{"status":"ACCEPTED","step":20}\n{"partial":')
            row, size = reader.json_tail(p)
            self.assertEqual(row['step'], 20); self.assertGreater(size, 20)

    def test_stdout_unknown_text_never_exported(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'log'; p.write_text(json.dumps(dict(status='HOLD', reason='raw secret\nnewline', prompt='private')) + '\n')
            r = reader.compact_stdout(p)
            self.assertEqual(r['last_safe_summary'], {'status':'HOLD'})

    def test_missing_own_is_not_success(self):
        self.assertEqual(reader.current_process(None)['state'], 'NO_PERSISTED_OWN_IDENTITY')

    def test_accepted_rows_sha_is_fresh_and_wrong_sha_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'rows'; p.write_text('fixture')
            proof = reader.stable_rows_sha(p, reader.sha_file(p)); self.assertTrue(proof['fresh_SHA_pass'])
            with self.assertRaises(reader.Hold): reader.stable_rows_sha(p, '0'*64)

    def test_metadata_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d).resolve() / 'data'; p.write_text('{}'); link = Path(d).resolve() / 'link'; link.symlink_to(p)
            with self.assertRaises(reader.Hold): reader.read_json(link)


if __name__ == '__main__': unittest.main()
