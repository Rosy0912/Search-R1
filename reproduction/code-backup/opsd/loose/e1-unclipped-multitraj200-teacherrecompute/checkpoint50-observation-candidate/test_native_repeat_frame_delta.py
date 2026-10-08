"""Narrow local framing regression, never main/SSH/PT load/GPU."""
import base64
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
H=HERE.parent
spec=importlib.util.spec_from_file_location('checkpoint_native_repeat_delta',HERE/'collect_checkpoints50_100_v2.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class TestNativeRepeatDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals=H/'checkpoint50_100-collection-once-v1-original'
        cls.gate=c.load_gate((H/'job3304-first5-complete-originals-v3/raw/source/search_r1/baselines/checkpoint_gate.py').read_bytes())
        cls.warnings=(H/'job3304-first5-complete-originals-v3/raw/native-source/fully_sharded_data_parallel.py').read_bytes()
        assert c.sha(cls.warnings)==c.FSDP_SHA
        cls.line=(cls.originals/'failed-frame-4923202.original').read_bytes()
        cls.prefix,raw=cls.line.decode().split('FULL_STATE_RANK_JSON=',1)
        cls.body,end=json.JSONDecoder().raw_decode(raw)
        cls.suffix=raw[end:]
        cls.warning=cls.suffix.split('\x1b[0m',1)[1]

    def gate_parse(self,line):
        return self.gate.parse_log_marker(line.decode(),'FULL_STATE_RANK_JSON=',driver_pid=c.DRIVER[0],job_id='3304',
            worker_pids=tuple(x[0] for x in c.WORKERS.values()),checkpoint_warning_path=c.TORCH_ROOT/'utils/checkpoint.py',
            fsdp_warning_source_bytes=self.warnings)

    def frame(self,body=None,prefix=None,tag='\x1b[32m [repeated 5x across cluster]\x1b[0m',tail='\n'):
        return ((self.prefix if prefix is None else prefix)+'FULL_STATE_RANK_JSON='+c.canonical(self.body if body is None else body).decode()+tag+tail).encode()

    def test_two_actual_rejected_frames_and_full_aggregate_preserve_bytes(self):
        raw=(self.originals/'train-original-prefix.log').read_bytes()
        self.assertEqual(c.sha(raw),'ce946501ea3f271f4e8449779c79a2bac891c0b0b4c63b63465ffb387bbe74a0')
        events,partial=c.parse_events(raw,self.gate,self.warnings)
        frames=[e for e in events if 'native_repeat_transport' in e]
        self.assertEqual([(e['offset'],e['native_repeat_transport']['repeated_count'],e['value']['rank']) for e in frames],
            [(4866679,5,3),(4923202,2,7)])
        self.assertEqual(len([e for e in events if e['marker']=='OPSD_E1_OUTER_JSON=']),114)
        self.assertEqual(partial,[])
        for e in frames:
            line=(self.originals/f"failed-frame-{e['offset']}.original").read_bytes()
            self.assertEqual(base64.b64decode(e['raw_base64']),line)
            self.assertEqual(raw[e['offset']:e['offset']+e['bytes']],line)
            self.assertEqual(c.sha(line),e['sha256'])
            with self.assertRaisesRegex(ValueError,'Unrecognized/unowned'):self.gate_parse(line)
            start=e['native_repeat_transport']['tag_offset_in_line'];length=e['native_repeat_transport']['tag_bytes']
            tag=line[start:start+length]
            self.assertEqual(base64.b64decode(e['native_repeat_transport']['tag_base64']),tag)
            self.assertEqual(self.gate_parse(line[:start]+line[start+length:]),e['value'])
        # All other already accepted records keep their exact original payload.
        for e in events:
            if 'native_repeat_transport' not in e:
                line=base64.b64decode(e['raw_base64'])
                v=self.gate.parse_log_marker(line.decode(),e['marker'],driver_pid=c.DRIVER[0],job_id='3304',
                    worker_pids=tuple(x[0] for x in c.WORKERS.values()),checkpoint_warning_path=c.TORCH_ROOT/'utils/checkpoint.py',fsdp_warning_source_bytes=self.warnings)
                self.assertEqual(v,e['value'])

    def test_exact_coloured_counts1_to8_and_original7_remain_accepted(self):
        for count in range(1,9):
            for tail in ('\n',self.warning):
                line=self.frame(tag=f'\x1b[32m [repeated {count}x across cluster]\x1b[0m',tail=tail)
                events,partial=c.parse_events(line,self.gate,self.warnings)
                self.assertEqual(len(events),1);self.assertEqual(events[0]['value'],self.body);self.assertEqual(partial,[])
                if count==7:
                    self.assertNotIn('native_repeat_transport',events[0]);self.assertEqual(self.gate_parse(line),self.body)
                else:self.assertEqual(events[0]['native_repeat_transport']['repeated_count'],count)
        for tag in (' [repeated 5x across cluster]','\x1b[32m [repeated 0x across cluster]\x1b[0m',
                    '\x1b[32m [repeated 9x across cluster]\x1b[0m','\x1b[32m [repeated 05x across cluster]\x1b[0m',
                    '\x1b[31m [repeated 5x across cluster]\x1b[0m','\x1b[32m [repeated 5x across cluster]',
                    '\x1b[32m [repeated 5x across cluster]\x1b[0m\x1b[32m [repeated 2x across cluster]\x1b[0m'):
            with self.assertRaises(ValueError):c.parse_events(self.frame(tag=tag),self.gate,self.warnings)

    def test_adapter_requires_periodic_typed_rank_to_original_pid(self):
        for key,bad in [('version',True),('version',3.0),('algorithm','pg'),('status','EXACT_FULL_STATE_RELOAD_OK'),
                        ('rank',True),('rank',7.0),('rank',3),('rank',8),('pid',True),('pid',c.WORKERS[3][0]),
                        ('world_size',8.0),('world_size',True),('world_size',9),('step',True),('step',100.0),('step',5),('step',200)]:
            body=dict(self.body);body[key]=bad
            with self.subTest(key=key,bad=bad):
                with self.assertRaises(ValueError):c.parse_events(self.frame(body=body),self.gate,self.warnings)
        for prefix in (self.prefix.replace('2549233','2549229'),self.prefix.replace('2549233','99999'),
                       self.prefix.replace('\x1b[36m','').replace('\x1b[0m',''),self.prefix.replace('WorkerDict','main_task')):
            with self.assertRaises(ValueError):c.parse_events(self.frame(prefix=prefix),self.gate,self.warnings)
        body=dict(self.body);body['step']=50
        self.assertEqual(c.parse_events(self.frame(body=body),self.gate,self.warnings)[0][0]['value']['step'],50)

    def test_remaining_warning_always_uses_unchanged_gate(self):
        for tail in (self.warning.replace('2548974','99999'),self.warning.replace(':1399:',':1400:'),
                     self.warning.replace('utils/checkpoint.py','utils/unknown.py'),self.warning.replace('is deprecated','is unknown'),
                     ' arbitrary warning\n',' {}\n',' OPSD_E1_OUTER_JSON={}\n'):
            with self.assertRaises(ValueError):c.parse_events(self.frame(tail=tail),self.gate,self.warnings)
        # Other marker kinds never acquire this native-only compatibility path.
        line=self.frame().replace(b'FULL_STATE_RANK_JSON=',b'BASELINE_EVALUATION_METRICS_JSON=')
        with self.assertRaises(ValueError):c.parse_events(line,self.gate,self.warnings)

    def test_original_driver111_events_and_metrics_are_unchanged(self):
        raw=(H/'job3304-release-live-originals-v15/driver-current-prefix.out').read_bytes()
        events,partial=c.parse_events(raw,self.gate,self.warnings)
        self.assertEqual(partial,[]);self.assertFalse(any('native_repeat_transport' in e for e in events))
        for step in (50,100):
            metric,rows=c.progress(events,self.gate,step)
            self.assertEqual(len(rows),111);self.assertEqual(metric['optimizer_calls_per_rank'],[10*step]*8)
            paired=c.evaluations_and_console(raw,events,rows,self.gate,step)
            self.assertEqual(paired['dev_at_step']['value']['val/em/all'],.2265625 if step==50 else .2578125)


if __name__=='__main__':unittest.main()
