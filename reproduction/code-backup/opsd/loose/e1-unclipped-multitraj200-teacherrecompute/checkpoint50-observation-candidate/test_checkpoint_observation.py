"""Six small software cases; main/SSH/native PT load/GPU are never executed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('checkpoint_observation_fixture',HERE/'collect_checkpoints50_100.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
H=HERE.parent


class TestObservation(unittest.TestCase):
    def test_actual111_small_log_original50_100_events_console_and_warmup(self):
        code=(H/'job3304-first5-complete-originals-v3/raw/source/search_r1/baselines/checkpoint_gate.py').read_bytes();gate=c.load_gate(code)
        warnings=(H/'job3304-first5-complete-originals-v3/raw/native-source/fully_sharded_data_parallel.py').read_bytes()
        self.assertEqual(c.sha(warnings),c.FSDP_SHA)
        raw=(H/'job3304-release-live-originals-v15/driver-current-prefix.out').read_bytes()
        events,partial=c.parse_events(raw,gate,warnings)
        for step in (50,100):
            item,rows=c.progress(events,gate,step);self.assertEqual(len(rows),111);self.assertEqual(item['optimizer_calls_per_rank'],[step*10]*8)
            v=c.evaluations_and_console(raw,events,rows,gate,step)
            self.assertEqual(v['dev_at_step']['value']['val/qa_count/all'],512)
            self.assertEqual(v['stage_at_step']['value']['val/qa_count/all'],64)
            self.assertEqual(v['dev_at_step']['value']['val/em/all'],.2265625 if step==50 else .2578125)
        self.assertEqual(partial,[])

    def test_regular_json_no_lf_and_emitted_complete_namespace(self):
        raw=(H/'job3304-first5-complete-originals-v3/raw/trajectories/outer_000001/questions.json').read_bytes()
        self.assertFalse(raw.endswith(b'\n'));self.assertIsInstance(c.parse(raw),dict)
        namespace={'__name__':'isolated_compiled_reader'}
        exec(compile((HERE/'collect_checkpoints50_100.py').read_bytes(),'<actual-assembled-reader>','exec',dont_inherit=True,optimize=0),namespace)
        self.assertEqual(len(namespace['SOURCE_FILES']),19);self.assertEqual(namespace['WORKERS'][6][0],2549230)
        self.assertIsInstance(namespace['parse'](b'{"document":true}'),dict)
        with self.assertRaises(ValueError):namespace['parse'](b'{"a":1,"a":2}')

    def test_prefix_append_allowed_no_replacement_or_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();p=root/'log';p.write_bytes(b'{"complete":true}')
            def append():
                with p.open('ab') as f:f.write(b'\nnext\n')
            raw,record=c.log_prefix(p,append);self.assertEqual(raw,b'{"complete":true}');self.assertGreater(record['observed_size_after'],record['bytes'])
            def overwrite():p.write_bytes(b'x'*p.stat().st_size)
            with self.assertRaises(ValueError):c.log_prefix(p,overwrite)
            p.write_bytes(b'original\n')
            def replace():q=root/'other';q.write_bytes(b'original\n');os.replace(q,p)
            with self.assertRaises(ValueError):c.log_prefix(p,replace)

    def test_requested_step_strict_types_and_absent_complete_outer(self):
        gate=type('FixtureGate',(),{})()
        for step in (True,50.0,0,49,200):
            with self.assertRaisesRegex(ValueError,'Exact stored'):c.progress([],gate,step)
        with self.assertRaisesRegex(ValueError,'completed OUTER'):c.progress([],gate,50)
        for steps in ([True],[100,50],[50,50],[]):
            with self.assertRaisesRegex(ValueError,'Exact ordered'):c.collect({},steps)

    def test31_saved_schema_and_full_file_hash_not_native_restore(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name).resolve();actor=run/'checkpoints/actor/global_step_50';state=actor/'training_state';state.mkdir(parents=True)
            def dump(path,obj):path.write_bytes(c.canonical(obj))
            for n in c.HF_SMALL:(actor/n).write_bytes(b'fixture-metadata')
            shards=[f'model-{i:05d}-of-00003.safetensors' for i in range(1,4)]
            for n in shards:(actor/n).write_bytes(b'fake-weight-byte-fixture-not-loadable')
            dump(actor/'model.safetensors.index.json',dict(weight_map={f'k{i}':shards[i%3] for i in range(435)}))
            for r in range(8):
                pt=state/f'rank_{r}.pt';pt.write_bytes(b'not-native-state');pin=c.sha(pt.read_bytes())
                dump(state/f'rank_{r}.json',dict(version=3,algorithm='pure_opsd',status='FULL_STATE_SERIALIZED_OK',rank=r,world_size=8,step=50,
                    pid=c.WORKERS[r][0],scheduler_last_epoch=50,optimizer_state_entries=37,model_dtensor_entries=435,path=str(pt),sha256=pin,state_sha256=pin,
                    teacher=dict(counters=dict(cuda_update_entries=50))))
            driver=run/'full_state/global_step_50';driver.mkdir(parents=True);(driver/'driver.pt').write_bytes(b'not-native-driver')
            dump(driver/'driver.json',dict(status='DRIVER_STATE_SERIALIZED_OK',step=50,pid=c.DRIVER[0],uid=0,job_id='3304',state_sha256=c.sha((driver/'driver.pt').read_bytes())))
            algorithm=run/'checkpoints/algorithm/global_step_50';algorithm.mkdir(parents=True)
            dump(algorithm/'state.json',dict(version=1,method='pure_opsd',global_step=50,outcome_rewards_train_policy=False,trajectory_archive=dict(completed_outer=50)))
            files={}
            with patch.object(c,'RUN',run):report=c.saved_members(files,50)
            self.assertEqual(len(files),31);self.assertIsNone(report['native_reload']);self.assertIsNone(report['rng_content_readback'])
            value=json.loads((state/'rank_0.json').read_bytes());value['rank']=False;dump(state/'rank_0.json',value)
            with patch.object(c,'RUN',run):
                with self.assertRaisesRegex(ValueError,'serialized report'):c.saved_members({},50)
            p=run/'hash';p.write_bytes(b'fixture')
            with self.assertRaises(ValueError):c.read_member(p,'0'*64)
            link=run/'hard';os.link(p,link)
            with self.assertRaises(ValueError):c.read_member(p)

    def test_unknown_command_is_not_true0_and_not_retried(self):
        child=type('FixtureChild',(),{'pid':17,'communicate':lambda self,timeout:(_ for _ in ()).throw(subprocess.TimeoutExpired('command',timeout,output=b'partial',stderr=b'error'))})()
        with patch.object(c.subprocess,'Popen',return_value=child) as pop:
            with self.assertRaises(TimeoutError) as result:c.command(['fixture'])
        self.assertEqual(pop.call_count,1);v=result.exception.command_original
        self.assertIs(v['waited'],False);self.assertIsNone(v['returncode']);self.assertEqual(v['stdout'],'partial')


if __name__=='__main__':unittest.main()
