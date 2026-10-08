"""Two focused CPU regressions: actual producer schema and completed-C0 reuse."""
import copy
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import utils as u
import completion
import completed_c0 as recovery
import study_runner


class CompletedStats(unittest.TestCase):
    def test_real_producer_full_diagnostics_schema_not_base_miner_only(self):
        runtime=Path(os.environ.get('OPSD_ABLATION_TEST_RUNTIME') or HERE.parent/'opsd_debug_frozen_runtime_20261008/runtime')
        helper=Path(os.environ.get('OPSD_ABLATION_TEST_TRAIN_HELPER') or HERE.parent/'opsd_ablation_v1')
        with tempfile.TemporaryDirectory() as td:
            train=dict(schema='opsd-ablation-config-v1',variant='C0',study_id='fixture',run_id='c'*32,
                scientific=dict(outer_iterations=10,component_clip=10),paths=dict(helper_root=str(helper)),
                helper_source_manifest=u.fileproof(helper/'source-manifest.json'))
            ref=u.atomic_json(Path(td)/'config.json',train)
            config=dict(study_id='fixture',variant='C0',training=dict(config=ref,run_id=train['run_id']))
            modules,proof=u.frozen_helpers(config)
            self.assertEqual(proof['files']['sampling.py']['sha256'],
                'fbab8c4318b15e3e89bafe89edd6c896daf22faa36f98a60ed2f1ac8c1d724f5')
            paths=['search_e1/__init__.py','search_e1/core.py','upstream/Search-R1/search_r1/__init__.py','upstream/Search-R1/search_r1/qa_protocol.py']
            science=dict(root=str(runtime),files={name:u.fileproof(runtime/name) for name in paths})
            with patch.object(u,'runtime_files',return_value=science):core,_,_=u.science(config)
            questions=[dict(question_id='synthetic-'+str(i),question='Synthetic item '+str(i),answers=['yes']) for i in range(512)]
            rows=[]
            for question in questions:
                for sibling in range(5):
                    response='<think>synthetic</think><answer>'+('yes' if sibling==0 else 'no')+'</answer>'
                    row=dict(question,sibling=sibling,response=response,policy_spans=[[0,len(response)]],
                        reward=int(sibling==0),search_calls=0,token_count=9,termination='answer')
                    row['diagnostics']=modules['sampling'].rollout_diagnostics(row);rows.append(row)
            pairs,stats=modules['sampling'].verify_rollouts(rows,questions,core,'C0',0)
            _,base=modules['miner_v1'].verify_rollouts(rows,questions,'C0',0,core)
            self.assertNotEqual(base,stats);self.assertEqual(stats['diagnostics']['denominator'],2560)
            sampling=dict(stats=stats);manifest=dict(sampling_stats=stats,P=512)
            self.assertEqual(completion.verify_mined_inputs(modules,rows,questions,train,0,core,pairs,sampling,manifest),stats)
            broken=copy.deepcopy(manifest);broken['sampling_stats']['diagnostics']['denominator']=2559
            with self.assertRaises(u.Hold):completion.verify_mined_inputs(modules,rows,questions,train,0,core,pairs,sampling,broken)
            with self.assertRaises(u.Hold):completion.verify_mined_inputs(modules,rows,questions,train,0,core,pairs,dict(stats=base),manifest)
            self.assertNotIn('torch',sys.modules)

    def test_completed_terminal_preflight_then_c0_has_no_training_spawn(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);output=root/'training-runs';output.mkdir()
            save=lambda name,value:u.atomic_json(root/name,value)
            train=dict(schema='opsd-ablation-config-v1',variant='C0',study_id='s',run_id='c'*32,
                scientific=dict(outer_iterations=10),paths=dict(output_root=str(output),python='/fixed/python'))
            tf=save('config.json',train);groups=[dict(variant=v,training_config=tf) for v in u.VARIANTS]
            old=dict(study_id='s',run_id='a'*32,groups=groups,helper_root='/old/opsd_ablation_eval_v1')
            oldref=save('old-study.json',old)
            parent=dict(pid=20,uid=1005,starttick=200,run_id=old['run_id'],argv=[train['paths']['python'],'-B',
                old['helper_root']+'/study_runner.py','--study-manifest',oldref['path'],'--study-manifest-sha256',oldref['sha256']])
            start=save('driver-start.json',dict(owner=parent,input_manifest=oldref))
            controller=dict(pid=30,uid=1005,starttick=300,run_id=train['run_id'])
            exited=save('training-exit.json',dict(identity=controller,run_id=train['run_id'],actual_wait_returncode=0))
            released=save('training-release.json',dict(accepted=True))
            cp=save('checkpoint.json',dict(accepted=True,cursor=dict(outer_index=9)))
            repair=dict(path='/fixed/repair.json',sha256='d'*64)
            status=dict(phase='TRAIN_COMPLETE',outer_iterations=10,owner=controller,run_id=train['run_id'],config_sha256=tf['sha256'],latest_checkpoint=cp)
            status_ref=u.atomic_json(output/'status.json',status)
            terminal=save('group-terminal.json',dict(schema='opsd-ablation-group-foreground-terminal-v1',accepted=True,
                actual_exit_code=0,signal_termination_not_exit0=False,run_id=train['run_id'],config_sha256=tf['sha256'],
                dtype_repair=repair,controller=controller,status=status_ref,study_driver=dict(owner=parent,input_manifest=oldref),
                actual_child_exit=exited,actual_release=released))
            command=['env','CUDA_VISIBLE_DEVICES=','E1_RUN_ID='+old['run_id'],'PYTHONDONTWRITEBYTECODE=1']+parent['argv']
            parent_terminal=save('parent-terminal.json',dict(schema='opsd-ablation-study-actual-terminal-v1',accepted=True,
                actual_exit_code=2,signal_termination_not_exit0=False,owner=parent,driver_start=start,input_manifest=oldref,
                session='synthetic-session',pane_dead_record='$73 %73 20 1 2',foreground_log=str(root/'driver.log'),
                pane_start_command=shlex.join(command)+' > '+shlex.quote(str(root/'driver.log'))+' 2>&1'))
            value=dict(groups=groups,study_id='s',run_id='b'*32,output_root=str(root/'new-driver'),dtype_repair=repair,completed_c0=dict(
                schema='opsd-ablation-completed-c0-eval-recovery-v1',training_config=tf,completed_training_terminal=terminal,
                training_status=status_ref,raw10_checkpoint=cp,previous_driver_start=start,previous_study_manifest=oldref,
                previous_driver_terminal=parent_terminal,previous_session='synthetic-session',expected_dead_pane='$73 %73 20 1 2'))
            self.assertEqual(recovery.preflight(value)['status'],status)
            bad=copy.deepcopy(value);bad['completed_c0']['raw10_checkpoint']=dict(cp,sha256='0'*64)
            with self.assertRaises(u.Hold):recovery.preflight(bad)
            status['phase']='TRAINING';u.atomic_json(output/'status.json',status)
            with self.assertRaises(u.Hold):recovery.preflight(value)
            status['phase']='TRAIN_COMPLETE';u.atomic_json(output/'status.json',status)
            history=output/'child-history.jsonl';history.write_text(u.canonical(dict(identity=controller)).decode()+'\n')
            events=[]
            admit=types.SimpleNamespace(process=lambda pid:None,require_external_study_release=lambda *a,**k:dict(accepted=True),
                require_launch=lambda *a,**k:dict(accepted=True))
            modules=dict(prepare=types.SimpleNamespace(validate_config=lambda *a,**k:True),gates=admit)
            driver=study_runner.Driver(value,dict(path='new-study',sha256='f'*64));driver.output=root/'new-driver';driver.output.mkdir()
            driver.owner=dict(pid=40,uid=1005,starttick=400,run_id=value['run_id'])
            driver.observer=lambda *a:events.append('retained-observer')
            driver.spawn=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No C0 training spawn'))
            evaluation=dict(training=dict(terminal=None,raw10_checkpoint=None,run_id=train['run_id'],config=tf))
            with patch.object(recovery,'check_dead_parent',return_value=recovery.preflight(value)), \
                 patch.object(recovery,'frozen_helpers',return_value=(modules,{})), \
                 patch.object(recovery.completion,'validate',return_value=dict(accepted=True,totals={'outer_iterations':10})):
                actual,status_actual=driver.training(groups[0],0,evaluation)
            self.assertEqual(actual,terminal);self.assertEqual(status_actual,status)
            self.assertEqual(events,['retained-observer'])
            proof=json.loads((driver.output/'00-C0/completed-training-reused.json').read_text())
            self.assertTrue(proof['no_training_spawn']);self.assertTrue(proof['no_adam_repeated'])


if __name__=='__main__':unittest.main()
