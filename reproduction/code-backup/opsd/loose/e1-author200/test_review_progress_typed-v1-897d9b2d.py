"""Local CPU actual-original replays and explicitly synthetic fail-closed fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_progress_typed as p

SNAP = p.HERE / 'actual-snapshots/3261-20261005T232501896059Z'
RAW = p.HERE / 'raw-log-framing75-20261005T235349402397Z'
gate, common, manifest, manifest_record = p.runtime_helpers()
summary = common.loads((SNAP/'observation-summary.json').read_bytes())
context, _ = p.source_context(SNAP,summary,gate,common,manifest,manifest_record)
driver = (RAW/'driver.out').read_bytes()
aggregate = (RAW/'train.log').read_bytes()
failure = common.loads((RAW/'source-parser-framing-failures.json').read_bytes())[0]
frame = aggregate.splitlines(keepends=True)[failure['line']-1]


class RealReplayTests(unittest.TestCase):
    def test_full_actual64_matches_oldcaller_output(self):
        old = p.module_original(p.HERE/'review_progress.py',
            '402760999d507bf874da06813f601132644ce52a60bc291a0d8f1938fd49f6e4','original_progress_checker')
        before = old.review(SNAP)
        after = p.review(SNAP)
        for key,value in before.items():
            self.assertEqual(after[key],value,key)
        self.assertEqual(after['actual_complete_outer'],64)
        self.assertEqual(after['aggregate_complete_outer'],63)
        self.assertEqual(after['log_join']['interleaved_frames'],{'driver':[],'aggregate':[]})

    def test_real76_75_logs_only_preserve_exact_frame(self):
        value = p.review_logs(driver,aggregate,context,common)
        self.assertEqual((value['actual_complete_outer'],value['aggregate_complete_outer']),(76,75))
        proof = value['log_join']['interleaved_frames']['aggregate']
        self.assertEqual(len(proof),1)
        self.assertEqual(proof[0]['raw'].encode(),frame)
        self.assertEqual(proof[0]['record_sha256'],failure['record_sha256'])
        self.assertEqual(proof[0]['source_binding']['source_record']['sha256'],p.FSDP_SHA)
        self.assertEqual(value['original_log_prefixes']['aggregate']['excluded_unclosed_tail_bytes'],18035)
        self.assertNotIn('observed_gpu_used_gib',value)
        self.assertNotIn('snapshot_finished_utc',value)

    def test_original_gate_still_rejects_and_raw_unchanged(self):
        with self.assertRaisesRegex(ValueError,'Unrecognized/unowned'):
            gate.parse_log_marker(frame.decode(),'OPSD_E1_OUTER_JSON=',driver_pid=1533312,
                job_id='3261',worker_pids=p.WORKERS,checkpoint_warning_path=context['checkpoint_warning_path'])
        self.assertEqual(hashlib.sha256((RAW/'train.log').read_bytes()).hexdigest(),
                         hashlib.sha256(aggregate).hexdigest())

    def test_packet_is_not_full_resource_snapshot(self):
        with self.assertRaises(FileNotFoundError):
            p.review(RAW)


class SnapshotIdentityFixtures(unittest.TestCase):
    def rejected_summary(self,modify):
        data=copy.deepcopy(summary);modify(data)
        original=p.read_original
        def read(path,*args,**kwargs):
            raw,record=original(path,*args,**kwargs)
            if Path(path)==SNAP/'observation-summary.json':
                raw=common.canonical(data)
                record=dict(record,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            return raw,record
        with patch.object(p,'read_original',side_effect=read):
            with self.assertRaises((ValueError,KeyError)):
                p.review(SNAP)

    def test_wrong_source(self):
        self.rejected_summary(lambda d:d['commands']['source_head'].update(stdout='bad\n'))
    def test_source_dirty(self):
        self.rejected_summary(lambda d:d['commands']['source_porcelain'].update(stdout=' M bad.py\n'))
    def test_slurm_stopped(self):
        self.rejected_summary(lambda d:d['commands']['scontrol'].update(stdout=d['commands']['scontrol']['stdout'].replace('JobState=RUNNING','JobState=FAILED')))
    def test_wrong_gpu_owner(self):
        self.rejected_summary(lambda d:d['unfiltered_application_csv_rows'][0].__setitem__(1,'99999'))
    def test_missing_gpu_application(self):
        self.rejected_summary(lambda d:d['unfiltered_application_csv_rows'].pop())
    def test_restarted_rank(self):
        self.rejected_summary(lambda d:d['processes']['1533910'].update(start_ticks='9999'))
    def test_restarted_driver(self):
        self.rejected_summary(lambda d:d['processes']['1533312'].update(start_ticks='9999'))
    def test_restarted_retriever(self):
        self.rejected_summary(lambda d:d['processes']['1531763'].update(start_ticks='9999'))
    def test_wrong_uid(self):
        self.rejected_summary(lambda d:d['processes']['1533910'].update(uids=[1]*4))
    def test_wrong_cgroup(self):
        self.rejected_summary(lambda d:d['processes']['1533910'].update(cgroup='/job_99/step_0'))
    def test_wrong_captured_log_origin(self):
        self.rejected_summary(lambda d:d['files']['train.log'].update(physical_path='/tmp/wrong.log'))
    def test_wrong_captured_log_sha(self):
        self.rejected_summary(lambda d:d['files']['train.log'].update(sha256='0'*64))


class LogFixtures(unittest.TestCase):
    def reject(self,newaggregate):
        with self.assertRaises((ValueError,KeyError)):
            p.review_logs(driver,newaggregate,context,common)
    def test_unknown_warning_owner(self):
        changed=frame.replace(b'WorkerDict pid=1533910)\x1b[0m /mnt/data',b'WorkerDict pid=9999999)\x1b[0m /mnt/data',1)
        self.assertNotEqual(changed,frame)
        self.reject(aggregate.replace(frame,changed,1))
    def test_wrong_warning_path(self):
        changed=frame.replace(b'fully_sharded_data_parallel.py:689',b'bad.py:689',1)
        self.assertNotEqual(changed,frame)
        self.reject(aggregate.replace(frame,changed,1))
    def test_extra_frame_suffix(self):
        self.reject(aggregate.replace(frame,frame[:-1]+b' extra\n',1))
    def test_second_json(self):
        self.reject(aggregate.replace(frame,frame[:-1]+b' {"extra":1}\n',1))
    def test_duplicate_json_key(self):
        self.reject(aggregate.replace(frame,frame.replace(b'"actual_outer_updates": 74',b'"actual_outer_updates": 74,"actual_outer_updates":74',1),1))
    def test_nonfinite_json(self):
        self.reject(aggregate.replace(frame,frame.replace(b'"actual_outer_updates": 74',b'"actual_outer_updates":NaN',1),1))
    def test_duplicate_outer(self):
        self.reject(aggregate+frame)
    def test_release_not_signed_receipt(self):
        self.reject(aggregate.replace(p.RECEIPT.encode(),b'0'*64))
    def test_missing_release(self):
        self.reject(b''.join(line for line in aggregate.splitlines(keepends=True) if b'OPSD_E1_STEP5_RELEASED_JSON=' not in line))
    def test_aggregate_conflicts_with_driver(self):
        self.reject(aggregate.replace(b'"actual_outer_updates": 74',b'"actual_outer_updates": 73',1))


class ByteBindingFixtures(unittest.TestCase):
    def test_modified_source_packet_rejected_before_context_use(self):
        with tempfile.TemporaryDirectory() as d:
            packet=Path(d).resolve()
            for file in p.FSDP_PACKET.iterdir():
                if file.is_file(): (packet/file.name).write_bytes(file.read_bytes())
            (packet/'fully_sharded_data_parallel.py').write_bytes(b'# synthetic changed source\n')
            with patch.object(p,'FSDP_PACKET',packet):
                with self.assertRaisesRegex(ValueError,'Pinned original SHA differs'):
                    p.source_context(SNAP,summary,gate,common,manifest,manifest_record)
    def test_manifest_pin_refuses_byte_drift(self):
        with tempfile.TemporaryDirectory() as d:
            tools=Path(d).resolve()
            for file in p.TOOLS.iterdir():
                if file.is_file(): (tools/file.name).write_bytes(file.read_bytes())
            with (tools/'manifest.json').open('ab') as f:f.write(b' ')
            with patch.object(p,'TOOLS',tools):
                with self.assertRaisesRegex(ValueError,'Pinned original SHA differs'):
                    p.runtime_helpers()
    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'file').write_bytes(b'x');(root/'alias').symlink_to(root/'file')
            with self.assertRaisesRegex(ValueError,'Physical original required'):
                p.read_original(root/'alias')
    def test_duplicate_summary_json_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):
            common.loads(b'{"job_id":"3261","job_id":"99"}')


if __name__=='__main__':unittest.main(verbosity=2)
