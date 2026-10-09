"""Only focused regression checks for the actual unreaped-mirror failure."""
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import gates
import supervisor
from utils import fileproof

class FakeChild:
    def __init__(self,rc):self.rc=rc;self.polls=0;self.waits=0
    def poll(self):self.polls+=1;return self.rc
    def wait(self):self.waits+=1;return self.rc

class MirrorReaping(unittest.TestCase):
    def make(self,d,rc):
        root=Path(d);scope=root/'scope';scope.mkdir();(root/'mirror.py').write_text('# CPU mirror\n')
        request=root/'request.json';request.write_text('{}\n');request_ref=fileproof(request)
        target=root/'copies';target.mkdir();(target/'outer-010-complete.json').write_text(json.dumps({'accepted':True,'request':request_ref})+'\n')
        child=FakeChild(rc);obj=supervisor.Supervisor.__new__(supervisor.Supervisor)
        obj.value={'helper_root':str(root),'mirror_root':str(target)}
        obj.mirror_jobs=[{'process':child,'handle':io.StringIO(),'identity':{'pid':1,'uid':1,'starttick':1},'command':['CPU'],
            'scope':scope,'request':request_ref,'completed_outers':10}]
        return obj,child,scope
    def test_completed_copy_reaped_and_reused(self):
        with tempfile.TemporaryDirectory() as d:
            obj,child,scope=self.make(d,0);first=obj.harvest_mirrors()
            self.assertTrue(first[0]['accepted']);self.assertEqual(json.loads((scope/'actual-exit.json').read_text())['actual_wait_returncode'],0)
            self.assertEqual(obj.finish_mirrors(),first);self.assertEqual(child.polls,1);self.assertEqual(child.waits,0)
    def test_live_copy_never_blocks_gpu_poll(self):
        with tempfile.TemporaryDirectory() as d:
            obj,child,scope=self.make(d,None)
            self.assertEqual(obj.harvest_mirrors(),[]);self.assertEqual(child.waits,0);self.assertFalse((scope/'actual-exit.json').exists())
    def test_actual_failed_copy_recorded_without_gpu_barrier(self):
        with tempfile.TemporaryDirectory() as d:
            obj,child,scope=self.make(d,2);result=obj.harvest_mirrors()
            self.assertFalse(result[0]['accepted']);self.assertTrue(result[0]['does_not_block_gpu'])
            self.assertEqual(json.loads((scope/'actual-exit.json').read_text())['actual_wait_returncode'],2)
    def test_ownership_gate_unchanged_and_foreign_child_blocked(self):
        here=Path(__file__).resolve().parent;old=here.parent/'opsd_pr100_periodic_v1'/'gates.py'
        self.assertEqual((here/'gates.py').read_text(),old.read_text().replace('experiments/search_e1/control/opsd_pr100_periodic_v1/',
            'experiments/search_e1/control/opsd_pr100_periodic_recovery_v1/'))
        scanner=types.ModuleType('unchanged_scanner');p=here.parent/'opsd_pr100_training_v1'/'owned_process_audit.py'
        exec(compile(p.read_bytes(),str(p),'exec'),scanner.__dict__)
        def record(pid,uid,ppid):return dict(pid=pid,uid=uid,starttick=pid,ppid=ppid,pgid=pid,state='S',comm='python',
            argv=['python','foreign.py'],argv_sha256='fixed',kernel_uids=[uid]*4,proc_directory_owner_uid=uid)
        values={1:record(1,0,0),10:record(10,1005,1),11:record(11,1005,10)}
        result=scanner.audit_same_uid('evaluation',1005,10,[],['/unrelated'],ignored_exact=[values[10]],
            snapshot_reader=lambda pid:values[pid],environment_reader=lambda pid:{'E1_RUN_ID':'foreign'},pid_reader=lambda:[11],ancestry_anchors=[values[10]])
        self.assertFalse(result['accepted_no_owned_or_opaque']);self.assertEqual(result['blockers'][0]['pid'],11)

if __name__=='__main__':unittest.main()
