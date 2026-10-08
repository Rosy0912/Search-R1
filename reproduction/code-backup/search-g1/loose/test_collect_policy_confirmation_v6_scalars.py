"""CPU readonly/transport fixtures only, not native completion or healthy evidence."""
import base64
import copy
import importlib.util
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v6_scalar_collector',HERE/'collect_policy_confirmation_v6_scalars.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def actual_input_paths():
    # CPU/prereg originals plus explicit synthetic method/Slurm metadata.
    import test_submit_policy_confirmation_v6_native as fixtures
    f=fixtures.fixture()
    temporary=tempfile.TemporaryDirectory(); path=Path(temporary.name).resolve()
    artifacts=fixtures.materialize(path,f)
    result={label:artifacts[{'birth':'observation'}.get(label,label)] for label in ('method','preregistration','cpu','launch','birth','terminal')}
    result['terminal']=result['birth']  # RUNNING must reject before RPC.
    _fixture_directories.append(temporary)
    return result


_fixture_directories=[]


def projection_fixture():
    small,_=m.load_modules()
    fields=small.SCALAR_SEGMENT_FIELDS
    document={key:None for key in fields|m.SKIPPED_SEGMENT_FIELDS}
    document.update(role='CPU_protocol_only',origin='not_native_fixture',run_id='CPU',
        actual_optimizer_states={'CPU_discarded_payload':list(range(5000))},
        actual_training_inputs={'CPU_discarded_string':'DO_NOT_TRANSFER'*5000},
        production_admission=False,healthy_checkpoint_accepted=False)
    return fields,document


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.small,cls.reader=m.load_modules()

    def test_actual_approved_contract_rejects_current_RUNNING_no_rpc_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory).resolve()/'packet.json';rpc=lambda *args:self.fail('RPC must not run')
            with self.assertRaisesRegex(ValueError,'terminal state'):
                m.execute_once(actual_input_paths(),264,output=output,execute=True,rpc=rpc)
            self.assertFalse(output.exists())

    def test_default_execute_false_rejects_before_inputs_or_rpc(self):
        with patch.object(m,'input_contract',side_effect=AssertionError('Must not inspect input')):
            with self.assertRaisesRegex(ValueError,'explicit future execute'):
                m.execute_once({},264,output=HERE/'must-not-create.json',rpc=lambda *args:None)

    def test_missing_originals_or_wrong_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'six explicit'):
            m.input_contract({},264)
        for count in (None,True,226,238):
            with self.assertRaisesRegex(ValueError,'actual264'):
                m.input_contract(actual_input_paths(),count)
        bad=actual_input_paths();bad['method']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'fixed input SHA'):
            m.input_contract(bad,264)

    def test_cli_requires_explicit_future_terminal_and_does_not_submit(self):
        result=subprocess.run([sys.executable,'-B',str(HERE/'collect_policy_confirmation_v6_scalars.py')],
            capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('--terminal',result.stderr)
        source=Path(m.__file__).read_text()
        self.assertNotIn("['sbatch'",source);self.assertNotIn('signal.pidfd_send_signal',source)
        self.assertNotIn('prepare_selection(',source);self.assertNotIn('verify_child(',source)
        self.assertNotIn('safe_open(',source);self.assertNotIn('get_tensor(',source)

    def test_frozen_helper_SHA_and_embedded_original_helpers_compile(self):
        code=m.remote_code(self.small,self.reader)
        namespace={};exec(code.split('p=json.load(sys.stdin)',1)[0],namespace)
        self.assertEqual(namespace['small'].SCALAR_SEGMENT_FIELDS,self.small.SCALAR_SEGMENT_FIELDS)
        self.assertEqual(namespace['reader'].verify_terminal_current.__code__.co_names,
                         self.reader.verify_terminal_current.__code__.co_names)
        self.assertNotIn('source_precheck',namespace['reader'].__dict__)
        self.assertNotIn('target_evidence',namespace['reader'].__dict__)
        self.assertNotIn('torch',sys.modules)

    def test_owned_regular_read_exact_original_bytes_and_metadata_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'scalar.json';raw=b'{"number":2}';path.write_bytes(raw)
            found,receipt=m.read_owned(path)
            self.assertEqual(found,raw);self.assertEqual(receipt['sha256'],m.sha(raw))
            self.assertEqual(receipt['size_bytes'],len(raw));self.assertEqual(receipt['inode'],path.stat().st_ino)
            self.assertLessEqual(receipt['read_started_unix'],receipt['read_finished_unix'])
            self.assertEqual(path.read_bytes(),raw)

    def test_symlink_hardlink_empty_and_oversize_original_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();path=base/'scalar';path.write_bytes(b'abc');link=base/'link';link.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'symlink'):m.read_owned(link)
            with self.assertRaisesRegex(ValueError,'bounded'):m.read_owned(path,maximum=2)
            hard=base/'hard';os.link(path,hard)
            with self.assertRaisesRegex(ValueError,'bounded'):m.read_owned(hard)
            empty=base/'empty';empty.touch()
            with self.assertRaisesRegex(ValueError,'bounded'):m.read_owned(empty)

    def test_same_size_mutation_during_read_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'original';path.write_bytes(b'abc')
            fd,before=m.open_owned(path,3)
            try:
                path.write_bytes(b'xyz');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000000))
                with self.assertRaisesRegex(ValueError,'changed during'):
                    m.finish_read(path,fd,before,3,m.sha(b'abc'),started=0.)
            finally:os.close(fd)

    def test_segment_streaming_projection_preserves_full_original_hash_only(self):
        fields,document=projection_fixture();raw=m.canonical(document)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'segment.json';path.write_bytes(raw)
            projected,receipt=m.project_segment(path,fields)
            self.assertEqual(set(projected),fields);self.assertEqual(receipt['sha256'],m.sha(raw))
            self.assertNotEqual(receipt['sha256'],m.sha(m.canonical(projected)))
            self.assertNotIn(b'DO_NOT_TRANSFER',m.canonical(projected));self.assertEqual(path.read_bytes(),raw)

    def test_stream_skips_excluded_values_across_UTF8_chunks(self):
        fields,document=projection_fixture();document['semantics']='原字符'*30000
        parser=m.SegmentProjection(io.BytesIO(m.canonical(document)),fields)
        projected=parser.project();self.assertEqual(set(projected),fields)
        self.assertEqual(parser.digest.hexdigest(),m.sha(m.canonical(document)))
        self.assertLessEqual(len(parser.buffer),65536)

    def test_stream_unknown_missing_duplicate_malformed_and_raw_field_reject(self):
        fields,document=projection_fixture()
        for change in ('unknown','missing','raw'):
            value=copy.deepcopy(document)
            if change=='missing':value.pop('actual_updates')
            else:value['raw_question' if change=='raw' else 'future_unknown_field']='never export'
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.SegmentProjection(io.BytesIO(m.canonical(value)),fields).project()
        raw=m.canonical(document)
        for bad in (raw+b'x',raw[:-1],b'{"role":1,"role":2}',b'{"role":"bad\\q"}',b'{"role":NaN}'):
            with self.subTest(raw=bad[:80]),self.assertRaises(ValueError):
                m.SegmentProjection(io.BytesIO(bad),fields).project()

    def test_stream_nested_duplicates_in_discarded_values_also_reject(self):
        fields,document=projection_fixture();raw=m.canonical(document)
        raw=raw.replace(b'"actual_worker_births_at_segment_close":null',
                        b'"actual_worker_births_at_segment_close":{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            m.SegmentProjection(io.BytesIO(raw),fields).project()

    def test_metadata_closed_filename_and_raw_payload_reject(self):
        for name in ('discovery/logs/raw-journal.json','discovery/logs/policy-calibration-training-inputs.json',
                     'discovery/model/model.safetensors','discovery/checkpoints/optimizer-state.pt'):
            with self.assertRaisesRegex(ValueError,'whitelisted'):m.whitelist_metadata(name,b'{}',self.small)
        with self.assertRaisesRegex(ValueError,'Forbidden'):
            m.whitelist_metadata('discovery/logs/run-manifest.json',m.canonical({'run_id':'CPU','raw_question':'x'}),self.small)
        with self.assertRaisesRegex(ValueError,'Unknown manifest'):
            m.whitelist_metadata('discovery/logs/run-manifest.json',m.canonical({'run_id':'CPU','unknown':0}),self.small)

    def test_exact_current_runtime_claim_and_manifest_new_fields_are_allowed(self):
        claim=dict(run_id='CPU',slurm_job_id='1',shell_pid=1,shell_start_ticks=2,claimed_unix=0.,
            semantics='exclusive launch claim, not evidence of a completed update')
        m.whitelist_metadata('discovery/logs/training-launch-claim.json',m.canonical(claim),self.small)
        m.whitelist_metadata('discovery/logs/run-manifest.json',m.canonical({'probe_logical_ordinal':7}),self.small)
        m.whitelist_metadata('discovery/preparation/parent-run-manifest.json',m.canonical({'run_id':'CPUparent'}),self.small)
        with self.assertRaises(ValueError):
            m.whitelist_metadata('discovery/logs/training-launch-claim.json',m.canonical(claim|{'raw_answer':'x'}),self.small)

    def test_worker_count_and_global_ledger_exact_schema(self):
        sys.path.insert(0,str(self.small.REPO))
        from reproduction.common8.test_safety_calibration_evidence import fabricated_segment
        logs,binding=fabricated_segment(role='policy_safety_calibration_discovery')
        for rows in logs.values():
            for row in rows:
                if row['kind']=='POLICY_MONITOR':
                    # A bounded synthetic delta=.001 scalar distribution. This
                    # adjusts actual arithmetic coherently, never a native fact.
                    k3=math.expm1(.001)-.001
                    row.update(sampled_old_policy_kl=k3,
                        weighted_k3_numerator=row['effective_weight_mass']*k3,
                        negative_tail_count=0,negative_tail_count_fraction=0.,
                        tail_effective_weight_mass=0.,negative_tail_weight_fraction=0.,
                        dual_clip_saturated_objective_weight_fraction=0.,
                        negative_log_ratio_p95=.001,negative_log_ratio_p99=.001,negative_log_ratio_max=.001)
        readers=self.small.pure_readers(m.strict_json((HERE/'policy-discovery-v6-native-launch.json').read_bytes())['source_sha256'])
        structural=readers.ledger.verify_segment(logs,binding)
        raw={'discovery/logs/optimizer-rank-'+str(rank)+'.jsonl':
            b'\n'.join(m.canonical(row) for row in rows)+b'\n' for rank,rows in logs.items()}
        segment=dict(role=binding['role']+'_evidence',run_id=binding['run_id'],plan_sha256=binding['calibration_sha256'],
            worker_log_sha256={str(rank):m.sha(raw['discovery/logs/optimizer-rank-'+str(rank)+'.jsonl']) for rank in range(8)},structural=structural)
        for name,content in raw.items():m.whitelist_metadata(name,content,self.small)
        self.assertEqual(self.small.check_ledgers(raw,segment,readers),structural)
        first=next(iter(raw));bad=raw[first]+raw[first].splitlines()[0]+b'\n'
        with self.assertRaisesRegex(ValueError,'61'):m.whitelist_metadata(first,bad,self.small)

    def test_paired512_arithmetic_can_capture_actual_drop_without_selecting_health(self):
        launch=m.strict_json((HERE/'policy-discovery-v6-native-launch.json').read_bytes())
        aggregate=self.small.pure_readers(launch['source_sha256']).aggregate
        ids=['CPU-hash-'+str(i) for i in range(512)];sources=['nq']*256+['hotpotqa']*256
        rows=[dict(question_id=identity,data_source=source,em=1,f1=1.,actual_calls=1,valid_action=2,
            turns=2,answer_present=True,answer_sha256='1'*64,response_token_count=4,model_token_count=2)
            for identity,source in zip(ids,sources)]
        run=dict(run_id='CPU',parent_production_run_id='CPUparent',policy_calibration_slurm_job_id='1')
        segment=dict(role='policy_safety_calibration_discovery_evidence',plan_sha256='2'*64,evaluation_sha256={})
        raw={}
        for label,step in (('before',0),('after',1)):
            actual=copy.deepcopy(rows)
            if step:
                for row in actual:row.update(em=0,f1=0.)
            value=aggregate(actual,ids,sources)|dict(run_id='CPU',parent_production_run_id='CPUparent',
                slurm_job_id='1',plan_sha256='2'*64,step=step,phase_role='policy_safety_calibration_discovery',
                validation_sha256='3'*64,checked_unix=0.)
            content=m.canonical(value);name='discovery/logs/policy-calibration-evaluation-'+label+'.json'
            m.whitelist_metadata(name,content,self.small);raw[name]=content
            segment['evaluation_'+label]=value;segment['evaluation_sha256'][label]=m.sha(content)
        validation=dict(validation_sha256='3'*64,question_ids=ids,data_sources=sources)
        m.evaluation_arithmetic(raw,segment,run,validation,aggregate)
        with self.assertRaisesRegex(ValueError,'zero-allowed-drop'):
            self.small.check_evaluations(raw,segment,run|{'input_sha256':{'validation.parquet':'3'*64}},validation,aggregate)

    def test_source354_exact_parent_and_exact_three_Hydra_not_blanket_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();parent=root/'parent';child=root/'child';mapping={}
            for base in (parent,child):
                for index in range(354):
                    name='module'+str(index)+'.py';path=base/'repo'/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'# CPU source fixture\n');mapping[name]=m.sha(path.read_bytes())
            folder=child/'repo/outputs/2026-10-06/00-00-00/.hydra';folder.mkdir(parents=True)
            for name in ('config','hydra','overrides'):(folder/(name+'.yaml')).write_bytes(b'{}')
            extras=m.source_map_before_import(parent,child,mapping);self.assertEqual(len(extras),3)
            unknown=child/'repo/outputs/foreign.py';unknown.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'Only three'):m.source_map_before_import(parent,child,mapping)
            unknown.unlink();(parent/'repo/module0.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):m.source_map_before_import(parent,child,mapping)

    def test_exclusive_local_writer_does_not_overwrite_or_create_admission(self):
        # Only byte-archival fixture. No COMPLETED/native capture is fabricated.
        captured=dict(packet={'role':'CPU_protocol_fixture'},receipt={'physical_execution_authenticated':False})
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory).resolve()/'capture.json';result=m.write_capture(output,captured)
            self.assertEqual(output.read_bytes(),m.canonical(captured));self.assertEqual(stat.S_IMODE(output.stat().st_mode),0o600)
            self.assertFalse(result['production_admission']);self.assertFalse(result['healthy_checkpoint_accepted'])
            with self.assertRaisesRegex(ValueError,'exclusive'):m.write_capture(output,captured)
            self.assertEqual(output.read_bytes(),m.canonical(captured))


if __name__=='__main__':unittest.main()
