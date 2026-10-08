"""Three targeted local cases, synthetic150 is never actual training evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
H=HERE.parent
spec=importlib.util.spec_from_file_location('checkpoint150_delta',HERE/'collect_checkpoints50_100_150.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
REPO=H.parent/'opsd-unclipped-multitraj200-teacherrecompute'


class TestCheckpoint150Delta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate=c.load_gate((REPO/'search_r1/baselines/checkpoint_gate.py').read_bytes())
        cls.warnings=(H/'job3304-first5-complete-originals-v3/raw/native-source/fully_sharded_data_parallel.py').read_bytes()
        raw=(H/'job3304-release-live-originals-v15/driver-current-prefix.out').read_bytes()
        events,_=c.parse_events(raw,cls.gate,cls.warnings)
        cls.template=next(e['value'] for e in events if e['marker']=='OPSD_E1_OUTER_JSON=')
        cls.dev=copy.deepcopy(next(e['value'] for e in events if e['marker']=='BASELINE_EVALUATION_METRICS_JSON='))
        cls.stage=copy.deepcopy([e['value'] for e in events if e['marker']=='BASELINE_EVALUATION_METRICS_JSON='][1])

    def test_exact_literal_only_delta_and_frozen_source150_save_order(self):
        old=(H/'checkpoint50-observation-candidate/collect_checkpoints50_100_v2.py').read_bytes()
        self.assertEqual(c.sha(old),'61368d54b0e6732c9fef424b18df1903cce3bd50ed2ba3582c8c122b8426deba')
        text=old.decode()
        for a,b,n in [('(50,100)','(50,100,150)',5),('checkpoints50/100,','checkpoints50/100/150,',1),
                      ('completed OUTER50/100','completed OUTER50/100/150',1),('step50 or100 required','step50 or100 or150 required',2),
                      ('steps50/100 required','steps50/100/150 required',1),
                      ('TRAINING3304_CHECKPOINT50_100_READONLY_ORIGINALS','TRAINING3304_CHECKPOINT50_100_150_READONLY_ORIGINALS',1)]:
            self.assertEqual(text.count(a),n);text=text.replace(a,b)
        self.assertEqual(text.encode(),(HERE/'collect_checkpoints50_100_150.py').read_bytes())
        for relative,pin in c.SOURCE_FILES.items():self.assertEqual(c.sha((REPO/relative).read_bytes()),pin)
        self.assertTrue(self.gate.should_save(150,horizon=200,profile='multitraj200'))
        self.assertFalse(self.gate.should_save(149,horizon=200,profile='multitraj200'))
        source=(REPO/'verl/trainer/ppo/ray_trainer.py').read_text()
        body=source[source.index('                    # validate\n'):source.index('                    if self.global_steps == 5:')]
        needles=['val_metrics: dict = self._validate()','self.baseline_controller.stage_validation(self.global_steps)',
                 'self._save_checkpoint(metrics)','logger.log(data=metrics, step=self.global_steps)',"print('OPSD_E1_OUTER_JSON='"]
        locations=[body.index(x) for x in needles];self.assertEqual(locations,sorted(locations))
        save=source[source.index('    def _save_checkpoint('):source.index('    def _balance_batch(')]
        order=[save.index(x) for x in ['self.actor_rollout_wg.save_checkpoint(', 'self.baseline_controller.save(',
                                    'self.actor_rollout_wg.save_training_state(', 'self.e1_gate.save_driver(']]
        self.assertEqual(order,sorted(order));self.assertIn('roundtrip = self.global_steps == 5',save)

    def synthetic150(self):
        # Fixed real metadata forms the schema template only; all150 timings,
        # evaluations, calls and future checkpoint evidence here are synthetic.
        lines=[]
        def event(marker,value):lines.append(marker+c.canonical(value).decode()+'\n')
        event('BASELINE_EVALUATION_METRICS_JSON=',self.dev)
        for step in range(1,151):
            row=copy.deepcopy(self.template);row['global_step']=row['actual_outer_updates']=step
            m=row['metrics'];m={k:v for k,v in m.items() if not k.startswith(('val/','opsd_stage/'))};row['metrics']=m
            m['actor/lr']=min(step/57,1)*1e-6;m['actor/optimizer_lr_used']=min((step-1)/57,1)*1e-6
            if step==5:event('BASELINE_EVALUATION_METRICS_JSON=',self.stage)
            if step in (50,100,150):
                event('BASELINE_EVALUATION_METRICS_JSON=',self.dev);event('BASELINE_EVALUATION_METRICS_JSON=',self.stage)
                m.update(self.dev);m.update({'opsd_stage/'+k:v for k,v in self.stage.items()})
                console={**self.dev,**{'opsd_stage/'+k:v for k,v in self.stage.items()}}
                lines.append('step:'+str(step)+' - '+' - '.join(k+':'+format(float(v),'.3f') for k,v in console.items())+'\n')
            event('OPSD_E1_OUTER_JSON=',row)
            if step==5:event('OPSD_E1_STEP5_RELEASED_JSON=',dict(driver_pid=c.DRIVER[0],actual_outer_updates=5,next_global_step=6,receipt_sha256=c.RECEIPT))
        return ''.join(lines).encode()

    def test_synthetic150_clock1500_dev512_stage64_and_strict_boundaries(self):
        raw=self.synthetic150();events,partial=c.parse_events(raw,self.gate,self.warnings)
        metric,rows=c.progress(events,self.gate,150)
        self.assertEqual(len(rows),150);self.assertEqual(partial,[]);self.assertEqual(metric['optimizer_calls_per_rank'],[1500]*8)
        self.assertEqual(metric['expected_questions_consumed'],76800);self.assertIsNone(metric['actual_rng_contents'])
        view=c.evaluations_and_console(raw,events,rows,self.gate,150)
        self.assertEqual(view['dev_at_step']['value']['val/qa_count/all'],512)
        self.assertEqual(view['stage_at_step']['value']['val/qa_count/all'],64)
        self.assertEqual(len(view['earlier_events']),6)
        for step in (True,150.0,149,151,200):
            with self.assertRaises(ValueError):c.progress(events,self.gate,step)
        with self.assertRaisesRegex(ValueError,'completed OUTER'):c.progress([e for e in events if e is not rows[-1]],self.gate,150)
        for steps in ([150.0],[True],[150,100],[150,150],[]):
            with self.assertRaisesRegex(ValueError,'Exact ordered'):c.collect({},steps)

    def test_synthetic150_native_tag_joins_exact_fixed_rank_pid_only(self):
        original=(H/'checkpoint50_100-collection-once-v1-original/failed-frame-4923202.original').read_bytes().decode()
        prefix,raw=original.split('FULL_STATE_RANK_JSON=',1);value,end=json.JSONDecoder().raw_decode(raw)
        value['step']=150
        def line(v,p=prefix):return (p+'FULL_STATE_RANK_JSON='+c.canonical(v).decode()+raw[end:]).encode()
        event=c.parse_events(line(value),self.gate,self.warnings)[0][0]
        self.assertEqual(event['native_repeat_transport']['step'],150)
        self.assertEqual(event['native_repeat_transport']['repeated_count'],2)
        self.assertEqual(event['value']['pid'],c.WORKERS[7][0])
        for key,bad in [('step',150.0),('step',True),('step',151),('rank',True),('pid',c.WORKERS[3][0])]:
            v=dict(value);v[key]=bad
            with self.assertRaises(ValueError):c.parse_events(line(v),self.gate,self.warnings)


if __name__=='__main__':unittest.main()
