"""Focused regression boundaries; no models, Node access or GPU operations."""
import unittest
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


if __name__=='__main__':unittest.main()
