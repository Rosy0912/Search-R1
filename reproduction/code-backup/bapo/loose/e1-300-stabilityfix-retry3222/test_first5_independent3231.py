"""Independent end-to-end wrapper tests: real temp files, mocked Slurm/proc/gate only."""
import contextlib, copy, importlib.util, io, json, sys, tarfile, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import first5_guard3231 as g

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
publisher=load('independent_publisher','validate-publish-first5-release3231.py')
archive=load('independent_archive','archive-first5-extraction3231.py')

class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.run=self.root/'runs/bapo-instruct300-formal300-3231'
        self.logs=self.run/'logs';self.logs.mkdir(parents=True)
        self.directory=self.logs/'extract';self.directory.mkdir()
        self.manifest=self.run/'checkpoints/full_state/global_step_5/manifest.json';self.manifest.parent.mkdir(parents=True);self.manifest.write_text('{}')
        self.tool=self.root/'tools/step5-extract.py';self.tool.parent.mkdir();self.tool.write_text('reviewed extractor fixture')
        self.toolsha=g.digest(self.tool)
        self.snapshot=self.directory/'train-prefix.log'
        text=''.join('BAPO_TRAIN_METRICS_JSON='+json.dumps(dict(step=i))+'\n' for i in range(1,6))
        text+='E1_STEP5_PAUSED_JSON='+json.dumps(dict(pid=901,manifest_path=str(self.manifest),manifest_sha256=g.digest(self.manifest)))+'\n'
        self.snapshot.write_text(text);(self.run/'train.log').write_text(text)
        self.processes=[dict(pid=901+i,start_ticks=500+i,state='S',selected_env=dict(SLURM_JOB_ID='3231',BAPO_RUN_ID=str(self.run)))for i in range(9)]
        self.e=dict(extraction_complete=True,issues=[],actual_training_accepted=False,job_id=3231,run_root=str(self.run),source_expected=g.SOURCE,
          tool=dict(path=str(self.tool),sha256=self.toolsha),release_identity=dict(run_root=str(self.run),source_commit=g.SOURCE,global_step=5,actual_outer_updates=5,horizon=300,warmup_steps=114),
          expected_driver_pid=901,driver_state=dict(driver_pid=901),actual_processes_after_cpu=self.processes,
          train_log_prefix=dict(snapshot_path=str(self.snapshot),live_path=str(self.run/'train.log'),bytes=self.snapshot.stat().st_size,sha256=g.digest(self.snapshot)),checkpoint_manifest=dict(sha256=g.digest(self.manifest)))
        self.evidence=self.directory/'evidence.json';self.evidence.write_text(json.dumps(self.e));self.esha=g.digest(self.evidence)
        self.receipt=self.logs/'independent-staging.json';self.receipt.write_text(json.dumps(dict(status='PASS_INDEPENDENT_E1_STEP5_REVIEW',evidence=[dict(path=str(self.evidence),sha256=self.esha)])));self.rsha=g.digest(self.receipt)
        self.release=self.logs/'step5-independent-release.json';self.archive=self.logs/'extraction.tar.gz'
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        for module in (g,publisher,archive):
            self.stack.enter_context(patch.object(module,'ROOT',self.root));self.stack.enter_context(patch.object(module,'RUN',self.run));self.stack.enter_context(patch.object(module,'EXTRACTOR_SHA',self.toolsha))
        original_owned=g.owned_file
        for module in (g,publisher,archive):
            self.stack.enter_context(patch.object(module,'owned_file',side_effect=lambda path,base=None:original_owned(path,self.run if base is None else base)))
        self.stack.enter_context(patch.object(g,'proc_proof',side_effect=lambda pid:copy.deepcopy(next(p for p in self.processes if p['pid']==pid))))
        self.slurm=self.stack.enter_context(patch.object(publisher.subprocess,'check_output',return_value='JobId=3231 JobState=RUNNING Restarts=0 NodeList=h100-5'))
        self.gate=self.stack.enter_context(patch.object(publisher.subprocess,'run'))
    def pub(self):
        args=['publisher','--evidence',str(self.evidence),'--evidence-sha256',self.esha,'--signed-receipt',str(self.receipt),'--receipt-sha256',self.rsha]
        with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):publisher.main()
    def arc(self):
        args=['archive','--evidence',str(self.evidence),'--evidence-sha256',self.esha,'--archive',str(self.archive)]
        with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):archive.main()
    def test_actual_gate_call_and_same_inode_publish(self):
        self.pub();self.assertEqual(self.gate.call_count,1)
        args,kw=self.gate.call_args
        self.assertEqual(args[0][0],str(self.root/'env/bin/python'));self.assertEqual(args[0][-4:],[str(self.receipt),str(self.manifest),str(self.root/'repo'),'901'])
        self.assertIn('checkpoint_gate import validate_release',args[0][2]);self.assertIn('expected_driver_pid=int(sys.argv[4])',args[0][2])
        self.assertEqual(kw['cwd'],str(self.root/'repo'));self.assertTrue(kw['check']);self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kw['env']['PYTHONPATH'],str(self.root/'repo'))
        self.assertEqual(self.slurm.call_count,2);self.assertEqual(self.receipt.stat().st_ino,self.release.stat().st_ino)
    def test_fullstate_gate_failure_never_releases(self):
        self.gate.side_effect=RuntimeError('actual gate failed')
        with self.assertRaises(RuntimeError):self.pub()
        self.assertFalse(self.release.exists())
    def test_changed_receipt_during_gate_rejected(self):
        self.gate.side_effect=lambda *a,**k:self.receipt.write_text('changed')
        with self.assertRaises(ValueError):self.pub()
        self.assertFalse(self.release.exists())
    def test_changed_manifest_during_gate_rejected(self):
        self.gate.side_effect=lambda *a,**k:self.manifest.write_text('changed')
        with self.assertRaises(ValueError):self.pub()
        self.assertFalse(self.release.exists())
    def test_actual_advance_during_gate_rejected(self):
        def advance(*a,**k):
            with (self.run/'train.log').open('a')as f:f.write('BAPO_TRAIN_METRICS_JSON={"step":6}\n')
        self.gate.side_effect=advance
        with self.assertRaises(ValueError):self.pub()
        self.assertFalse(self.release.exists())
    def test_released_marker_before_gate_rejected(self):
        with (self.run/'train.log').open('a')as f:f.write('E1_STEP5_RELEASED_JSON={}\n')
        with self.assertRaises(ValueError):self.pub()
        self.assertEqual(self.gate.call_count,0);self.assertFalse(self.release.exists())
    def test_pause_pid_mismatch_rejected(self):
        (self.run/'train.log').write_text(self.snapshot.read_text().replace('"pid": 901','"pid": 999'))
        with self.assertRaises(ValueError):self.pub()
        self.assertEqual(self.gate.call_count,0)
    def test_slurm_change_after_gate_rejected(self):
        self.slurm.side_effect=['JobId=3231 JobState=RUNNING Restarts=0 NodeList=h100-5','JobId=3231 JobState=COMPLETED Restarts=0 NodeList=h100-5']
        with self.assertRaises(ValueError):self.pub()
        self.assertFalse(self.release.exists())
    def test_unsigned_archive_real_members_and_no_release(self):
        self.arc()
        with tarfile.open(self.archive)as f:self.assertEqual(set(f.getnames()),{'extract','extract/evidence.json','extract/train-prefix.log'})
        self.assertFalse(self.release.exists());self.assertEqual(self.gate.call_count,0)
        with self.assertRaises(ValueError):self.arc()
    def test_archive_symlink_rejected_before_creation(self):
        (self.directory/'link').symlink_to(self.evidence)
        with self.assertRaises(ValueError):self.arc()
        self.assertFalse(self.archive.exists())
    def test_archive_fifo_rejected_before_creation(self):
        import os
        os.mkfifo(self.directory/'special')
        with self.assertRaises(ValueError):self.arc()
        self.assertFalse(self.archive.exists())

if __name__=='__main__':unittest.main()
