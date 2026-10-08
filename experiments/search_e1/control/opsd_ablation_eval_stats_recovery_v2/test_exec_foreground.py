"""Two CPU regressions for metadata Bash after exact same-PID exec."""
import copy
from pathlib import Path
import stat
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import completed_c0 as c
import utils as u


class ExecForeground(unittest.TestCase):
    def fixture(self,root):
        source=root/'old-helper'/'study_runner.py';source.parent.mkdir()
        source.write_text('# exact synthetic immutable CPU driver\n')
        sf=u.fileproof(source);mf=u.atomic_json(source.parent/'source-manifest.json',dict(files={
            'study_runner.py':dict(sha256=sf['sha256'],bytes=sf['bytes'])}))
        old=dict(helper_root=str(source.parent),helper_source_manifest=mf,run_id='a'*32)
        op=u.atomic_json(root/'old-study.json',old)
        owner=dict(pid=170,uid=1005,starttick=100,ppid=20,pgid=170,run_id=old['run_id'],argv=[
            '/original/python','-B',str(source),'--study-manifest',op['path'],'--study-manifest-sha256',op['sha256']])
        start=dict(owner=owner,input_manifest=op);sp=u.atomic_json(root/'driver-start.json',start)
        log=str(root/'old-inputs'/'study-driver.log')
        binding=dict(schema='opsd-ablation-study-live-foreground-binding-v1',accepted=True,route='exec_same_pid',
            driver_start=sp,input_manifest=op,session='original-session',session_id='$73',pane_id='%73',pane_pid=170,
            pane_live_record='$73 %73 170 0',pane_start_command='bash',owner=owner,actual_owner=copy.deepcopy(owner),
            source=sf,helper_source_manifest=mf,foreground_log=log,stdout_fd1=log,
            cuda_visible_devices='',pythondontwritebytecode='1',
            server=dict(pid=20,uid=1005,starttick=1),
            socket=dict(path='/tmp//tmux-1005/default',device=10,inode=20,uid=1005,mode=stat.S_IFSOCK|0o660))
        bp=u.atomic_json(root/'live-binding.json',binding)
        parent=dict(pane_start_command='bash',pane_dead_record='$73 %73 170 1 2',foreground_log=log,live_foreground_binding=bp)
        value=dict(completed_c0=dict(live_foreground_binding=bp,previous_session='original-session',expected_dead_pane='$73 %73 170 1 2'))
        return value,start,sp,old,op,parent,binding

    def test_exact_actual_python_exec_same_pid_accepts_literal_bash_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            value,start,sp,old,op,parent,binding=self.fixture(Path(td))
            actual=c.exec_foreground(value,start,sp,old,op,parent)
            self.assertEqual(actual,binding)
            self.assertEqual(actual['actual_owner']['pid'],actual['pane_pid'])
            self.assertNotIn('torch',sys.modules)

    def test_different_wrapper_pid_cannot_use_exec_route(self):
        with tempfile.TemporaryDirectory() as td:
            value,start,sp,old,op,parent,binding=self.fixture(Path(td))
            binding['actual_owner']['pid']=171
            ref=u.atomic_json(Path(td)/'wrong-wrapper-binding.json',binding)
            value['completed_c0']['live_foreground_binding']=ref;parent['live_foreground_binding']=ref
            with self.assertRaisesRegex(u.Hold,'exact actual Python driver'):
                c.exec_foreground(value,start,sp,old,op,parent)


if __name__=='__main__':unittest.main()
