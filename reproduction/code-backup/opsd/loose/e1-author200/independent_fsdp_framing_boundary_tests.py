"""Independent local CPU checks of exact new framing helper bytes and originals."""
import argparse
import hashlib
import json
from pathlib import Path
import types
import unittest

CLI = argparse.ArgumentParser()
CLI.add_argument('--common-sha256', required=True)
ARGS = CLI.parse_args()
B = Path(__file__).resolve().parent
P = B/'final-acceptance-fsdp-framing-candidate/author200_observation.py'
RAW = B/'raw-log-framing75-20261005T235349402397Z'
SRC = B/'actual-fsdp-warning-source-20261005T235824197037Z'
PRIOR = B/'actual-snapshots/3261-20261005T232501896059Z/original-files'
code = P.read_bytes()
assert hashlib.sha256(code).hexdigest() == ARGS.common_sha256
m = types.ModuleType('independent_exact_fsdp_consumer')
m.__file__ = str(P)
exec(compile(code, str(P), 'exec'), m.__dict__)
gate = m.frozen_gate(B.parent/'opsd-e1-300')
closure = m.loads((PRIOR/'bindings/import_closure.json').read_bytes())
source = m.loads((SRC/'summary.json').read_bytes())
workers = []
for rank in range(8):
    p = PRIOR/f'checkpoints/actor/global_step_5/training_state/rank_{rank}.json'
    rec = m.loads(p.read_bytes())
    workers.append(dict(rank=rank,pid=rec['pid'],object_ids=rec['object_ids'],report_path=str(p),report_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
context = dict(gate=gate,workers=workers,
    checkpoint_warning_path=Path(closure['modules']['torch.utils._contextlib']['path']).parent/'checkpoint.py',
    fsdp_warning_source=dict(kind='EXTRA_ACTUAL_FSDP_SOURCE_NOT_SIGNED_NATIVE210_MODULE',
        native_closure_sha256=m.CLOSURE_SHA,torch_anchor=closure['modules']['torch.utils._contextlib'],
        source_record={k:source['files']['fully_sharded_data_parallel.py'][k] for k in ('physical_path','sha256','bytes')},warning_line=689))
failure = m.loads((RAW/'source-parser-framing-failures.json').read_bytes())[0]
record = (RAW/'train.log').read_bytes().splitlines(keepends=True)[failure['line']-1]
assert hashlib.sha256(record).hexdigest() == failure['record_sha256']
marker = 'OPSD_E1_OUTER_JSON='

class IndependentBoundaryTests(unittest.TestCase):
    def rejects(self, raw):
        with self.assertRaises((ValueError, KeyError)):
            rows=m.parse_records(raw,context)
            # Parser and exact clock-validation are deliberately separate layers.
            m.check_outer_rows(rows['outers'],context)

    def test_complete_actual_prefixes_and_raw_source_pin(self):
        draw=(RAW/'driver.out').read_bytes();araw=(RAW/'train.log').read_bytes()
        d=m.parse_records(draw[:draw.rfind(b'\n')+1],context)
        a=m.parse_records(araw[:araw.rfind(b'\n')+1],context)
        j=m.join_log_prefixes(d,a,context,75)
        self.assertEqual((j['authoritative_last_outer'],j['aggregate_last_outer'],j['aggregate_forwarding_lag']),(76,75,1))
        self.assertEqual(a['outers'][73],d['outers'][73])
        frames=a['interleaved_frames'];self.assertEqual(len(frames),1)
        f=frames[0]
        self.assertEqual((f['line'],f['source_offset'],f['record_sha256']),(984,3002462,failure['record_sha256']))
        self.assertEqual(f['raw'].encode(),record)
        self.assertEqual(f['raw_suffix']+f['raw_line_ending'],failure['suffix'])
        self.assertEqual(f['source_binding']['source_record']['sha256'],'ca839d30fda12e0b12dea934e1f1bf998bd80ed716e8e4c88927691a9bb1c6a3')
        self.assertEqual(len(a['native_console']),sum(b'FULL_STATE_RANK_JSON=' in line for line in araw[:araw.rfind(b'\n')+1].splitlines()))
        self.assertEqual(len(a['native_transport']),len(a['native_console'])+1)
        self.assertEqual(m.check_evaluation_sequence(d,76)['per_question_raw_evidence_available'],False)
        self.assertEqual(len(m.check_evaluation_sequence(d,76)['events']),4)

    def test_exact_original_gate_rejection_is_preserved(self):
        with self.assertRaisesRegex(ValueError,'Unrecognized/unowned'):
            gate.parse_log_marker(record.decode(),marker,driver_pid=m.DRIVER_PID,job_id=m.JOB,
                worker_pids={w['pid'] for w in workers},checkpoint_warning_path=context['checkpoint_warning_path'])

    def test_two_colour_or_spacing_variants_rejected(self):
        for part,replacement in [(b'\x1b[36m(WorkerDict pid=1533910)',b'\x1b[32m(WorkerDict pid=1533910)'),
                                 (b'\x1b[0m /mnt/data',b'\x1b[0m  /mnt/data')]:
            with self.subTest(part=part):self.rejects(record.replace(part,replacement))

    def test_hidden_second_warning_rejected(self):
        self.rejects(record[:-1]+b'\t'+failure['suffix'].encode())

    def test_extra_json_and_nonwarning_pids_rejected(self):
        for r in [record[:-1]+b' {"x":1}\n',record.replace(b'WorkerDict pid=1533910',b'main_task pid=1533312')]:
            with self.subTest(raw=r[-100:]):self.rejects(r)

    def test_body_duplicate_key_still_rejected_before_clock_use(self):
        self.rejects(record.replace(b'"global_step": 74',b'"global_step": 74, "global_step": 74',1))

    def test_invalid_body_syntax_not_rescued_by_warning(self):
        self.rejects(record.replace(b'"global_step": 74',b'"global_step": [74,',1))

    def test_finite_invalid_actual_counter_not_accepted_as_full_prefix(self):
        self.rejects(record.replace(b'"actual_outer_updates": 74',b'"actual_outer_updates": true',1))

    def test_native_tag_corner_exact_actual_old_frame(self):
        old=B/'actual-snapshots/3261-20261005T210622178545Z/original-files/train.log'
        r=old.read_bytes().splitlines(keepends=True)[444]
        self.assertIn(b'FULL_STATE_RANK_JSON=',r)
        parsed=m.parse_records(r,context)
        self.assertEqual(len(parsed['native_console']),1)
        self.assertEqual(parsed['native_transport'][0]['observed_repeat_tag'],7)
        self.assertEqual(parsed['interleaved_frames'],[])

    def test_new_warning_source_provenance_is_not_native210(self):
        self.assertFalse(any(v['path']==context['fsdp_warning_source']['source_record']['physical_path'] for v in closure['modules'].values()))
        self.assertEqual(len(closure['modules']),210)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundaryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert hashlib.sha256(P.read_bytes()).hexdigest()==ARGS.common_sha256
    raise SystemExit(0 if result.wasSuccessful() else 1)
