"""Focused regression boundaries; no models, Node access or GPU operations."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import recovery


class RecoveryTests(unittest.TestCase):
    def test_exact_previous_pid_cannot_be_recovered_while_alive(self):
        old=dict(pid=1,uid=1005,starttick=10)
        with self.assertRaises(recovery.Hold):
            recovery.absent(old,lambda _:dict(old))
        self.assertTrue(recovery.absent(old,lambda _:dict(old,starttick=11)))

    def test_zero_return_cannot_be_relabeled_failure(self):
        tf=dict(path='/original/config',sha256='a'*64)
        train=dict(variant='C0',run_id='r',paths=dict(output_root='/original/runs'))
        value=dict(recovery=dict(schema='opsd-ablation-c0-before-adam-recovery-v1',training_config=tf,
            run_id='r',output_root='/original/runs',driver_start={'path':'start'},training_failure={'path':'status'},
            actual_training_exit={'path':'exit'},actual_training_release={'path':'release'}))
        owner=dict(pid=1,uid=1005,starttick=10)
        records=dict(start=dict(owner=owner),status=dict(phase='FAILED_OWNED_RELEASE_CONFIRMED',
            signals_sent=False,run_id='r',config_sha256=tf['sha256'],owner=owner),
            exit=dict(actual_wait_returncode=1,identity=owner),release=dict(accepted=True))
        with patch.object(recovery,'checked_json',side_effect=lambda ref:(records[ref['path']],ref)):
            self.assertEqual(recovery.previous_values(value,train,tf)[2],records['status'])
            records['exit']['actual_wait_returncode']=0
            with self.assertRaises(recovery.Hold):recovery.previous_values(value,train,tf)

    def test_repair_cannot_change_config_digest(self):
        value=dict(dtype_repair=dict(path='/repair',sha256='b'*64))
        train=dict(study_id='s',variant='C0')
        tf=dict(path='/original/config',sha256='a'*64)
        repair=dict(schema='opsd-ablation-dtype-repair-manifest-v1',study_id='s',
            original_configs=[dict(tf,sha256='c'*64)])
        with patch.object(recovery,'checked_json',return_value=(repair,value['dtype_repair'])):
            with self.assertRaises(recovery.Hold):recovery.repair_command(value,train,tf,0)

    def test_reparented_observer_keeps_binding_but_uses_actual_role_identity(self):
        tf=dict(path='/original/config',sha256='a'*64)
        train=dict(study_id='s',run_id='train')
        driver=dict(pid=20,uid=1005,starttick=200,run_id='driver2')
        observer=dict(pid=30,uid=1005,starttick=300,ppid=10,pgid=30,argv=['python','observer'],run_id='observe')
        actual=dict(observer,ppid=1)
        start_ref=dict(path='/driver2-start',sha256='b'*64)
        input_ref=dict(path='/driver2-input',sha256='c'*64)
        r=dict(role_previous_driver_start=start_ref,role_previous_session='session2',
            role_previous_dead_pane='$72 %72 20 1 2')
        value=dict(recovery=r)
        old=dict(run_id='train',config_sha256=tf['sha256'],roles=[
            dict(role='study_driver',identity=driver,input_config=input_ref),
            dict(role='memory_observer',identity=observer,input_config=tf,source={'sha256':'original'})])
        records={'/driver2-start':dict(owner=driver,input_manifest=input_ref),
            '/driver2-input':dict(study_id='s',run_id='driver2',groups=[dict(training_config=tf)])}
        admit=SimpleNamespace(process=lambda pid:actual if pid==30 else None,
            same_process=lambda a,b:all(a[k]==b[k] for k in ('pid','uid','starttick')))
        with patch.object(recovery,'previous_values',return_value=(r,dict(owner={'pid':10}),{})), \
             patch.object(recovery,'checked_json',side_effect=lambda ref:(records[ref['path']],ref)), \
             patch.object(recovery.subprocess,'run',return_value=SimpleNamespace(stdout='$72 %72 20 1 2\n')) as pane:
            role=recovery.validate_old_roles(value,train,tf,old,admit)
            self.assertEqual(role['identity'],actual)
            self.assertEqual(old['roles'][1]['identity']['ppid'],10)
            self.assertEqual(role['input_config'],tf)
            self.assertIn('session2',pane.call_args.args[0])
            actual['starttick']=301
            with self.assertRaises(recovery.Hold):recovery.validate_old_roles(value,train,tf,old,admit)
            actual['starttick']=300;records['/driver2-input']['study_id']='foreign'
            with self.assertRaises(recovery.Hold):recovery.validate_old_roles(value,train,tf,old,admit)

    def test_additional_role_driver_requires_complete_bound_terminal(self):
        r=dict(role_previous_driver_start=dict(path='/start',sha256='b'*64))
        with patch.object(recovery,'previous_values',return_value=(r,{},{})):
            with self.assertRaises(recovery.Hold):
                recovery.validate_old_roles({}, {}, {}, {}, SimpleNamespace())


if __name__=='__main__':unittest.main()
