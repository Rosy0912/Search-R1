"""CPU-only actual raw-frame replay plus explicitly synthetic rejection fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author200_observation as m

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
WORK = PARENT.parent
RAW = PARENT / 'raw-log-framing75-20261005T235349402397Z'
SOURCE = PARENT / 'actual-fsdp-warning-source-20261005T235824197037Z'
PRIOR = PARENT / 'actual-snapshots/3261-20261005T232501896059Z/original-files'
GATE = m.frozen_gate(WORK / 'opsd-e1-300')
CLOSURE = m.loads((PRIOR / 'bindings/import_closure.json').read_bytes())
SUMMARY = m.loads((SOURCE / 'summary.json').read_bytes())
SOURCE_REC = SUMMARY['files']['fully_sharded_data_parallel.py']
SOURCE_RAW = (SOURCE / 'fully_sharded_data_parallel.py').read_bytes()
assert hashlib.sha256(SOURCE_RAW).hexdigest() == SOURCE_REC['sha256']
assert len(SOURCE_RAW) == SOURCE_REC['bytes'] == 99345
WORKERS = []
for rank in range(8):
    p = PRIOR / f'checkpoints/actor/global_step_5/training_state/rank_{rank}.json'
    row = m.loads(p.read_bytes())
    WORKERS.append(dict(rank=rank, pid=row['pid'], object_ids=row['object_ids'],
                        report_path=str(p), report_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
CONTEXT = dict(gate=GATE, workers=WORKERS,
    checkpoint_warning_path=Path(CLOSURE['modules']['torch.utils._contextlib']['path']).parent / 'checkpoint.py',
    fsdp_warning_source=dict(kind='EXTRA_ACTUAL_FSDP_SOURCE_NOT_SIGNED_NATIVE210_MODULE',
        native_closure_sha256=m.CLOSURE_SHA,
        torch_anchor=CLOSURE['modules']['torch.utils._contextlib'],
        source_record={k: SOURCE_REC[k] for k in ('physical_path', 'sha256', 'bytes')},
        warning_line=689, warning_source_line=SOURCE_RAW.decode().splitlines()[688],
        cpu_replay_scope='ACTUAL_CAPTURED_SOURCE_PIN_NOT_A_FINALIZED_FUTURE_TOOL_MANIFEST'))
FAILURE = m.loads((RAW / 'source-parser-framing-failures.json').read_bytes())[0]
RECORD = (RAW / 'train.log').read_bytes().splitlines(keepends=True)[FAILURE['line'] - 1]
assert hashlib.sha256(RECORD).hexdigest() == FAILURE['record_sha256']


def clean_record():
    line = RECORD.decode()
    body = line.split('OPSD_E1_OUTER_JSON=', 1)[1]
    _, end = json.JSONDecoder().raw_decode(body)
    return line[:line.index('OPSD_E1_OUTER_JSON=')] + 'OPSD_E1_OUTER_JSON=' + body[:end] + '\n'


def clone_context():
    return dict(copy.deepcopy({k: v for k, v in CONTEXT.items() if k != 'gate'}), gate=GATE)


class ActualFramingTests(unittest.TestCase):
    def reject(self, raw, context=CONTEXT):
        with self.assertRaises((ValueError, KeyError)):
            m.parse_records(raw, context)

    def test_actual_one_outer74_raw_frame(self):
        parsed = m.parse_records(RECORD, CONTEXT)
        self.assertEqual(len(parsed['outers']), 1)
        self.assertEqual(parsed['outers'][0]['actual_outer_updates'], 74)
        self.assertEqual(len(parsed['native_console']), 0)
        self.assertEqual(parsed['native_transport'], parsed['interleaved_frames'])
        frame = parsed['interleaved_frames'][0]
        self.assertEqual(frame['raw'].encode(), RECORD)
        self.assertEqual(frame['record_sha256'], FAILURE['record_sha256'])
        self.assertEqual(frame['raw_suffix'] + frame['raw_line_ending'], FAILURE['suffix'])
        self.assertEqual(frame['worker_evidence']['rank'], 0)
        self.assertEqual(frame['worker_evidence']['pid'], 1533910)
        self.assertEqual(frame['source_binding']['source_record']['sha256'], SOURCE_REC['sha256'])
        bounds = frame['byte_boundaries']
        self.assertEqual(RECORD[bounds['json_start']:bounds['json_end']], frame['raw_json'].encode())
        self.assertEqual(RECORD[bounds['suffix_start']:bounds['suffix_end']], frame['raw_suffix'].encode())
        self.assertEqual(frame['original_gate_sha256'], frame['clean_gate_sha256'])

    def test_actual_complete_driver76_aggregate75(self):
        draw = (RAW / 'driver.out').read_bytes()
        araw = (RAW / 'train.log').read_bytes()
        draw = draw[:draw.rfind(b'\n') + 1]
        araw = araw[:araw.rfind(b'\n') + 1]
        driver = m.parse_records(draw, CONTEXT)
        aggregate = m.parse_records(araw, CONTEXT)
        join = m.join_log_prefixes(driver, aggregate, CONTEXT, 75)
        self.assertEqual((join['authoritative_last_outer'], join['aggregate_last_outer']), (76, 75))
        self.assertEqual(aggregate['outers'][73], driver['outers'][73])
        frame = join['interleaved_frames']['aggregate'][0]
        self.assertEqual(frame['source_offset'], FAILURE['offset'])
        self.assertEqual(frame['line'], FAILURE['line'])
        self.assertEqual(frame['record_sha256'], FAILURE['record_sha256'])
        self.assertEqual(len(join['interleaved_frames']['aggregate']), 1)
        self.assertEqual(m.check_evaluation_sequence(driver, 76)['per_question_raw_evidence_available'], False)

    def test_original_gate_still_rejects_raw(self):
        with self.assertRaisesRegex(ValueError, 'Unrecognized/unowned'):
            GATE.parse_log_marker(RECORD.decode(), 'OPSD_E1_OUTER_JSON=', driver_pid=m.DRIVER_PID,
                                  job_id=m.JOB, worker_pids={w['pid'] for w in WORKERS},
                                  checkpoint_warning_path=CONTEXT['checkpoint_warning_path'])
        self.assertEqual(m.parse_records(clean_record().encode(), CONTEXT)['interleaved_frames'], [])

    def test_unknown_worker_pid(self):
        self.reject(RECORD.replace(b'WorkerDict pid=1533910', b'WorkerDict pid=9999999'))

    def test_driver_is_not_warning_worker(self):
        self.reject(RECORD.replace(b'WorkerDict pid=1533910', b'WorkerDict pid=1533312'))

    def test_wrong_source_path(self):
        self.reject(RECORD.replace(b'fully_sharded_data_parallel.py:689', b'other_file.py:689'))

    def test_wrong_source_line(self):
        self.reject(RECORD.replace(b'.py:689:', b'.py:688:'))

    def test_changed_warning_text(self):
        self.reject(RECORD.replace(b'FSDP1, FSDP2, DDP.', b'FSDP2, FSDP1, DDP.'))

    def test_extra_suffix(self):
        self.reject(RECORD[:-1] + b' extra text\n')

    def test_second_warning(self):
        suffix = FAILURE['suffix'].encode()
        self.reject(RECORD[:-1] + suffix)

    def test_second_json(self):
        self.reject(RECORD[:-1] + b' {"other":1}\n')

    def test_duplicate_marker(self):
        self.reject(RECORD[:-1] + b' OPSD_E1_OUTER_JSON={"other":1}\n')

    def test_other_marker(self):
        self.reject(RECORD[:-1] + b' SEARCH_R1_FINAL_METRICS_JSON={"other":1}\n')

    def test_json_duplicate_key(self):
        self.reject(RECORD.replace(b'"actual_outer_updates": 74', b'"actual_outer_updates": 74, "actual_outer_updates": 74', 1))

    def test_json_nonfinite(self):
        self.reject(RECORD.replace(b'"actual_outer_updates": 74', b'"actual_outer_updates": NaN', 1))

    def test_json_overflow(self):
        self.reject(RECORD.replace(b'"actual_outer_updates": 74', b'"actual_outer_updates": 1e999', 1))

    def test_warning_inside_broken_json(self):
        self.reject(RECORD.replace(b'"actual_outer_updates": 74', b'"actual_outer_updates": FSDP.state_dict_type()', 1))

    def test_wrong_prefix_owner(self):
        self.reject(RECORD.replace(b'main_task pid=1533312', b'main_task pid=9999999', 1))

    def test_wrong_body_driver(self):
        self.reject(RECORD.replace(b'"driver_pid": 1533312', b'"driver_pid": 9999999', 1))

    def test_wrong_body_job(self):
        self.reject(RECORD.replace(b'"job_id": "3261"', b'"job_id": "9999"', 1))

    def test_missing_source_pin(self):
        ctx = clone_context()
        del ctx['fsdp_warning_source']
        self.reject(RECORD, ctx)

    def test_bad_binding_sha(self):
        ctx = clone_context()
        ctx['fsdp_warning_source']['source_record']['sha256'] = 'not-a-source-pin'
        self.reject(RECORD, ctx)

    def test_wrong_torch_root(self):
        ctx = clone_context()
        ctx['fsdp_warning_source']['torch_anchor']['path'] = '/tmp/other/torch/utils/_contextlib.py'
        self.reject(RECORD, ctx)

    def test_missing_rank_identity(self):
        ctx = clone_context()
        ctx['workers'] = ctx['workers'][:7]
        self.reject(RECORD, ctx)

    def test_no_colour_variant_is_not_observed_new_frame(self):
        self.reject(RECORD.replace(b'\x1b[36m(WorkerDict pid=1533910)\x1b[0m', b'(WorkerDict pid=1533910)'))

    def test_no_implicit_fsdprepeat_dedup(self):
        self.reject(RECORD[:-1] + b' \x1b[32m [repeated 7x across cluster]\x1b[0m\n')

    def test_existing_exact_amp_after_json_still_original_gate(self):
        clean = clean_record().rstrip('\n')
        warning = ('\x1b[36m(WorkerDict pid=1533910)\x1b[0m '
                   + str(CONTEXT['checkpoint_warning_path'])
                   + ':1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. '
                   + "Please use `torch.amp.autocast('cpu', args...)` instead.")
        parsed = m.parse_records((clean + warning + '\n').encode(), CONTEXT)
        self.assertEqual(len(parsed['outers']), 1)
        self.assertEqual(parsed['interleaved_frames'], [])

    def test_existing_native_repeat_tag_before_amp(self):
        row = dict(pid=1533910, rank=0)
        warning = ('\x1b[36m(WorkerDict pid=1533910)\x1b[0m '
                   + str(CONTEXT['checkpoint_warning_path'])
                   + ':1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. '
                   + "Please use `torch.amp/autocast('cpu', args...)` instead.")
        warning = warning.replace('torch.amp/autocast', 'torch.amp.autocast')
        raw = ('\x1b[36m(WorkerDict pid=1533910)\x1b[0m FULL_STATE_RANK_JSON='
               + json.dumps(row) + '\x1b[32m [repeated 7x across cluster]\x1b[0m'
               + warning + '\n').encode()
        parsed = m.parse_records(raw, CONTEXT)
        self.assertEqual(parsed['native_console'], [row])
        self.assertEqual(parsed['native_transport'][0]['observed_repeat_tag'], 7)
        self.assertEqual(parsed['interleaved_frames'], [])


class SourceBindingFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.tools = self.root / 'tools'
        self.tools.mkdir()
        self.selfpath = self.tools / 'author200_observation.py'
        self.selfpath.write_bytes((HERE / 'author200_observation.py').read_bytes())
        self.contextlib = self.root / 'torch/utils/_contextlib.py'
        self.contextlib.parent.mkdir(parents=True)
        self.contextlib.write_bytes(b'# explicit synthetic CPU fixture\n')
        self.fsdp = self.root / 'torch/distributed/fsdp/fully_sharded_data_parallel.py'
        self.fsdp.parent.mkdir(parents=True)
        self.fsdp.write_bytes(SOURCE_RAW)
        self.closure = {'modules': {'torch.utils._contextlib': {
            'path': str(self.contextlib), 'sha256': hashlib.sha256(self.contextlib.read_bytes()).hexdigest()}}}
        self.manifest = {'source_commit': m.SOURCE, 'tools': {'author200_observation.py':
            hashlib.sha256(self.selfpath.read_bytes()).hexdigest()}, 'dependencies': {
            str(self.fsdp): hashlib.sha256(SOURCE_RAW).hexdigest()}}
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        (self.tools / 'manifest.json').write_bytes(m.canonical(self.manifest))

    def bind(self):
        with patch.object(m, '__file__', str(self.selfpath)):
            return m.bind_fsdp_warning_source(self.closure)

    def test_actual_source_bytes_under_synthetic_owned_tools_layout(self):
        value = self.bind()
        self.assertEqual(value['source_record']['sha256'], SOURCE_REC['sha256'])
        self.assertEqual(value['warning_line'], 689)

    def test_missing_dependency_rejected(self):
        self.manifest['dependencies'] = {}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'absent'):
            self.bind()

    def test_actual_source_changed_rejected(self):
        self.fsdp.write_bytes(b'changed source')
        with self.assertRaises(ValueError):
            self.bind()

    def test_common_source_pin_changed_rejected(self):
        self.selfpath.write_bytes(b'not reviewed helper')
        with self.assertRaises(ValueError):
            self.bind()

    def test_warning_literal_different_even_repin_rejected(self):
        changed = SOURCE_RAW.replace(b'FSDP1, FSDP2, DDP.', b'FSDP2, FSDP1, DDP.')
        self.fsdp.write_bytes(changed)
        self.manifest['dependencies'][str(self.fsdp)] = hashlib.sha256(changed).hexdigest()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'literal/category'):
            self.bind()

    def test_wrong_source_warning_line_even_repin_rejected(self):
        changed = b'\n' + SOURCE_RAW
        self.fsdp.write_bytes(changed)
        self.manifest['dependencies'][str(self.fsdp)] = hashlib.sha256(changed).hexdigest()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'line689'):
            self.bind()

    def test_missing_native_torch_anchor_rejected(self):
        self.closure['modules'] = {}
        with self.assertRaisesRegex(ValueError, 'Torch helper'):
            self.bind()

    def test_warning_keyword_even_repin_rejected(self):
        changed = SOURCE_RAW.replace(b'            FutureWarning,\n', b'            FutureWarning, stacklevel=1,\n', 1)
        self.assertNotEqual(changed, SOURCE_RAW)
        self.fsdp.write_bytes(changed)
        self.manifest['dependencies'][str(self.fsdp)] = hashlib.sha256(changed).hexdigest()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'literal/category'):
            self.bind()


if __name__ == '__main__':
    unittest.main(verbosity=2)
