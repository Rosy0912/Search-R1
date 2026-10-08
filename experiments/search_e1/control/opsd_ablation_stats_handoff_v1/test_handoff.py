"""Two focused CPU handoff contracts; no SSH, tmux, Torch, or model execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stats_handoff_tested', HERE / 'handoff.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


class HandoffTests(unittest.TestCase):
    def test_not_actual_terminal_cannot_build_import_or_spawn(self):
        touched = []
        for state in ('WAITING_NATURAL_PARENT_EXIT', 'MISSING_C0_TERMINAL0', 'UNKNOWN_PID'):
            with self.subTest(state=state), self.assertRaises(h.NotReady):
                h.finalize({}, None, inspector=lambda *_: dict(ready=False, state=state),
                    loader=lambda *_: touched.append('source-import'), builder=lambda *_: touched.append('write-inputs'),
                    launcher=lambda *_: touched.append('spawn'))
        self.assertEqual(touched, [])

    def test_true_terminal_builds_independent_exact_inputs_before_single_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); oldbase = root / 'old'; newbase = root / 'new'
            source = root / 'source'; source.mkdir()
            def put(path, value): return h.save(path, value)
            source_payload = b'# no model import\n'
            (source / 'study_runner.py').write_bytes(source_payload)
            manifest = put(source / 'source-manifest.json', dict(files={'study_runner.py':dict(sha256=h.digest(source_payload),bytes=len(source_payload))}))
            commit = '6' * 40
            pub = put(root / 'published.json', dict(accepted=True, branch='search-e1', source_commit=commit,
                remote_readback_sha=commit, source_manifest_sha256=manifest['sha256'], files=[
                    dict(path=h.STATS_NAMESPACE+'study_runner.py',sha256=h.digest(source_payload),bytes=len(source_payload),equal=True),
                    dict(path=h.STATS_NAMESPACE+'source-manifest.json',sha256=manifest['sha256'],bytes=manifest['bytes'],equal=True)]))
            groups = []; originals = {}
            for index, name in enumerate(list(h.VARIANTS) + ['zero']):
                variant = 'C0' if name == 'zero' else name
                train = put(oldbase / ('train-'+variant+'.json'), dict(schema='opsd-ablation-config-v1',variant=variant,
                    paths=dict(output_root=str(oldbase / ('train-'+variant) / 'runs'))))
                run_id = '%032x' % (100+index); base = oldbase / ('eval-'+name)
                storage = put(oldbase / ('storage-'+name+'.json'), dict(accepted=True,cluster='HKUST',mode='human_hkust_local_exception',
                    run_id=run_id,output_root=str(base/'runs'),human_authorization=dict(explicit_exception=True,text='authorized local seven',
                    observed_unix=1,thread_id='test'),backup_policy=dict(durable_target='Mac',synchronize_completed_checkpoints=True),
                    scope=dict(run_id=run_id,output_root=str(base/'runs'),study_id='same-study',variant=variant)))
                value = dict(run_id=run_id,study_id='same-study',variant=variant,paths=dict(base=str(base),output_root=str(base/'runs'),helper_root='old-source',python='/python'),
                    scientific=dict(validation_questions=512,rank_seed=42,total_primary_slots=1024 if name in ('zero','P','J') else 512),
                    training=dict(config=train,terminal=None,raw10_checkpoint=None),gates=dict(storage=storage,publication=None),
                    source_commit='1'*40,helper_source_manifest={'path':'old','sha256':'1'*64})
                ref = put(oldbase / ('template-'+name+'.json'), value); originals[name] = copy.deepcopy(value)
                if name != 'zero': groups.append(dict(variant=name,training_config=train,evaluation_template=ref))
                else: zero = ref
            old = dict(schema='opsd-ablation-seven-sequential-study-v1',run_id='a'*32,study_id='same-study',state='READY',
                groups=groups,common_zero_template=zero,output_root=str(oldbase/'driver'/'runs'),recovery={'old-failure':'preserved'},dtype_repair={'same':'exact'})
            oldref = put(oldbase/'study.json',old)
            plan = dict(study_root=str(root),destination=str(newbase/'inputs'),output_root=str(newbase/'driver'/'runs'),
                old_study=oldref,old_session='old-session',old_pane='%73',new_session='new-session',
                eval_bases={name:str(newbase/('eval-'+name)) for name in list(h.VARIANTS)+['zero']},
                run_ids={name:'%032x'%(200+index) for index,name in enumerate(['driver','zero']+list(h.VARIANTS))},
                new_source=dict(helper_root=str(source),manifest=manifest,publication=pub,commit=commit))
            terminal = dict(schema='opsd-ablation-study-actual-terminal-v1',accepted=True,actual_exit_code=2,
                signal_termination_not_exit0=False,pane_dead_record='$73 %73 1701673 1 2')
            observation = dict(ready=True,old=old,parent=terminal,driver_start={'path':'old-start','sha256':'2'*64},
                training_config=groups[0]['training_config'],training_status={'path':'status10','sha256':'3'*64},
                training_terminal={'path':'rc0-terminal','sha256':'4'*64},raw10_checkpoint={'path':'raw9','sha256':'5'*64})
            # Real tmux producer retains literal 'bash' after that same pane
            # PID execs Python. Only the exact live proc/source binding permits it.
            argv = ['/python','-B',str(source/'study_runner.py'),'--study-manifest',oldref['path'],'--study-manifest-sha256',oldref['sha256']]
            owner = dict(pid=1701673,uid=1005,starttick=927513372,argv=argv,run_id=old['run_id'],ppid=2373025,pgid=1701673)
            actual = dict(owner,cuda_visible_devices='',dont_write_bytecode='1')
            common = dict(driver_start=observation['driver_start'],input_manifest=oldref,session='old-session',session_id='$73',
                pane_id='%73',pane_pid=owner['pid'],owner=owner,server=dict(pid=2373025,uid=1005,starttick=884304691),
                socket=dict(path='/tmp/socket',device=1,inode=2,uid=1005,mode=49584),pane_start_command='bash',foreground_log=str(oldbase/'driver.log'))
            exec_old = dict(old,helper_root=str(source),helper_source_manifest=manifest)
            binding = h.exec_same_pid(common,actual,dict(owner=owner),exec_old,h.read(source/'study_runner.py')[1],'/python')
            with self.assertRaises(h.NotReady):
                h.exec_same_pid(common,dict(actual,pid=1701674),dict(owner=owner),exec_old,binding['source'],'/python')
            observation['binding']=binding
            order = []
            def validate(new):
                order.append('validate')
                self.assertNotIn('recovery',new)
                self.assertEqual(new['completed_c0']['raw10_checkpoint'],observation['raw10_checkpoint'])
                self.assertEqual([x['training_config'] for x in new['groups']],[x['training_config'] for x in groups])
                self.assertEqual(new['dtype_repair'],old['dtype_repair'])
                for name, ref in [(x['variant'],x['evaluation_template']) for x in new['groups']]+[('zero',new['common_zero_template'])]:
                    value,_ = h.checked(ref)
                    self.assertEqual(value['scientific'],originals[name]['scientific'])
                    self.assertEqual(value['training'],originals[name]['training'])
                    self.assertNotEqual(value['run_id'],originals[name]['run_id'])
                    self.assertEqual(value['source_commit'],commit)
                return True
            gates = types.SimpleNamespace(publication=lambda _:order.append('publication'),storage=lambda _:order.append('storage'))
            runner = types.SimpleNamespace(validate_manifest=validate,gates=gates)
            completed = types.SimpleNamespace(preflight=lambda _:order.append('c0-preflight'),check_dead_parent=lambda *_:order.append('fresh-dead-parent'))
            result = h.finalize(plan,{'bound':'live'},inspector=lambda *_:observation,
                loader=lambda _: (runner,completed,{}),launcher=lambda new,ref:order.append('single-spawn') or ref)
            self.assertEqual(order[:3],['validate','c0-preflight','fresh-dead-parent'])
            self.assertEqual(order[-1],'single-spawn'); self.assertEqual(order.count('single-spawn'),1)
            self.assertEqual(order.count('publication'),8); self.assertEqual(order.count('storage'),8)
            self.assertTrue(Path(result['path']).is_file())
            self.assertFalse(Path(plan['output_root']).exists())
            parent,_ = h.checked(h.read(Path(plan['destination'])/'previous-driver-terminal.json')[1])
            self.assertEqual({k:parent[k] for k in terminal},terminal)
            live,_=h.checked(parent['live_foreground_binding'])
            self.assertEqual(live,binding)
            new,_=h.checked(result)
            self.assertEqual(new['completed_c0']['live_foreground_binding'],parent['live_foreground_binding'])
            # Only the new own window is retained before any exec text/Enter.
            tmux_calls=[]
            def native(args,**kwargs):
                tmux_calls.append(args)
                if args[1]=='has-session': return types.SimpleNamespace(returncode=1,stdout='',stderr='')
                self.assertEqual(args[1],'new-session')
                return types.SimpleNamespace(returncode=0,stdout='$88 @88 %88 111\n',stderr='')
            def tmux(args):
                tmux_calls.append(args)
                if args[1]=='show-options': return 'on'
                if args[1]=='display-message': return '0'
                return ''
            def proc(pid):
                return dict(pid=pid,uid=1005,starttick=1,argv=['bash'] if pid==111 else ['python','handoff.py'],run_id=None)
            zero_value,_=h.checked(new['common_zero_template'])
            with patch.object(h.subprocess,'run',native),patch.object(h,'command',tmux),patch.object(h,'process',proc),patch.dict(os.environ,{},clear=True):
                launched=h.launch(plan,new,result,zero_value)
            self.assertTrue(launched['actual_exec_command_submitted'])
            option=next(i for i,args in enumerate(tmux_calls) if args[1]=='set-option')
            literal=next(i for i,args in enumerate(tmux_calls) if args[1]=='send-keys')
            self.assertLess(option,literal)
            self.assertEqual(tmux_calls[option],['tmux','set-option','-w','-t','@88','remain-on-exit','on'])
            self.assertIn('-l',tmux_calls[literal]);self.assertEqual(tmux_calls[-1],['tmux','send-keys','-t','%88','Enter'])
            self.assertFalse(any('-g' in args or '-gqv' in args or '%73' in args for args in tmux_calls))


if __name__ == '__main__': unittest.main()
