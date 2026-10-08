import copy,importlib.util,json,sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import first5_guard3231 as g

def sample():
    processes=[dict(pid=900+i,start_ticks=100+i,state='S',selected_env=dict(SLURM_JOB_ID='3231',BAPO_RUN_ID=str(g.RUN)))for i in range(9)]
    return dict(extraction_complete=True,issues=[],actual_training_accepted=False,job_id=3231,run_root=str(g.RUN),source_expected=g.SOURCE,tool=dict(sha256=g.EXTRACTOR_SHA),release_identity=dict(run_root=str(g.RUN),source_commit=g.SOURCE,global_step=5,actual_outer_updates=5,horizon=300,warmup_steps=114),expected_driver_pid=900,driver_state=dict(driver_pid=900),actual_processes_after_cpu=processes)
class Guards(unittest.TestCase):
    def test_actual_schema(self):self.assertEqual(g.evidence_identity(sample())[0],900)
    def test_identity_mutations(self):
        mutations=[('job_id',3177),('job_id',True),('run_root','/old'),('source_expected','733'),('extraction_complete',False),('issues',['bad']),('actual_training_accepted',True),('expected_driver_pid',True),('expected_driver_pid',0)]
        for key,value in mutations:
            with self.subTest(key=key,value=value):
                e=sample();e[key]=value
                with self.assertRaises(ValueError):g.evidence_identity(e)
    def test_gate_counter_mutations(self):
        for key,value in [('warmup_steps',57),('horizon',500),('global_step',4),('actual_outer_updates',True),('source_commit','b20'),('run_root','/old')]:
            with self.subTest(key=key):
                e=sample();e['release_identity'][key]=value
                with self.assertRaises(ValueError):g.evidence_identity(e)
    def test_old_extractor(self):
        e=sample();e['tool']['sha256']='old'
        with self.assertRaises(ValueError):g.evidence_identity(e)
    def test_process_mutations(self):
        for key,value in [('pid',True),('start_ticks',True),('start_ticks',0),('state','Z')]:
            with self.subTest(key=key):
                e=sample();e['actual_processes_after_cpu'][0][key]=value
                with self.assertRaises(ValueError):g.evidence_identity(e)
        for key,value in [('SLURM_JOB_ID','3177'),('BAPO_RUN_ID','/old')]:
            e=sample();e['actual_processes_after_cpu'][0]['selected_env'][key]=value
            with self.assertRaises(ValueError):g.evidence_identity(e)
    def test_duplicate_rank_pid(self):
        e=sample();e['actual_processes_after_cpu'][1]['pid']=900
        with self.assertRaises(ValueError):g.evidence_identity(e)
    def test_receipt_requires_review(self):
        p=g.RUN/'logs/new/evidence.json'
        for status in ['REQUIRES_INDEPENDENT_REVIEW','PASS_HARDWARE',None]:
            with self.assertRaises(ValueError):g.signed_binding(dict(status=status),p,'a'*64)
        r=dict(status='PASS_INDEPENDENT_E1_STEP5_REVIEW',evidence=[dict(path=str(p),sha256='a'*64)])
        g.signed_binding(r,p,'a'*64)
        with self.assertRaises(ValueError):g.signed_binding(r,p,'b'*64)
    def test_slurm(self):
        text='JobId=3231 JobState=RUNNING Restarts=0 NodeList=h100-5'
        g.validate_slurm(text)
        g.validate_slurm(text.replace(' ', '\r\n')+'\r\n')
        for suffix in ['NodeList=h100-5,h100-6','NodeList=h100-5-other','JobState=RUNNING-extra','Restarts=0-extra','JobId=3231-other']:
            key=suffix.split('=',1)[0];bad=' '.join(x for x in text.split()if not x.startswith(key+'='))+' '+suffix
            with self.subTest(suffix=suffix):
                with self.assertRaises(ValueError):g.validate_slurm(bad)
        for old,new in [('3231','3177'),('RUNNING','COMPLETED'),('Restarts=0','Restarts=1'),('h100-5','a800-1')]:
            with self.assertRaises(ValueError):g.validate_slurm(text.replace(old,new))
    def test_regular_owned_path(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t).resolve();p=root/'evidence';p.write_text('x')
            self.assertEqual(g.owned_file(p,root),p)
            s=root/'symlink';s.symlink_to(p)
            with self.assertRaises(ValueError):g.owned_file(s,root)
            with self.assertRaises(ValueError):g.owned_file(root,root)
            with self.assertRaises(ValueError):g.owned_file(Path('/etc/passwd'),root)
    def test_pid_reuse_and_foreign_env(self):
        from unittest.mock import patch
        processes=sample()['actual_processes_after_cpu']
        with patch.object(g,'proc_proof',side_effect=lambda pid:copy.deepcopy(next(p for p in processes if p['pid']==pid))):g.check_processes(processes)
        for mutation in [('start_ticks',999),('state','X'),('selected_env',dict(SLURM_JOB_ID='1',BAPO_RUN_ID='foreign'))]:
            now=copy.deepcopy(processes[0]);now[mutation[0]]=mutation[1]
            with patch.object(g,'proc_proof',return_value=now):
                with self.assertRaises(ValueError):g.check_processes(processes)
class PublisherImport(unittest.TestCase):
    def test_gate_subprocess_exact_repo_cwd(self):
        import ast
        tree=ast.parse((Path(__file__).parent/'validate-publish-first5-release3231.py').read_text())
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='subprocess'and n.func.attr=='run']
        self.assertEqual(len(calls),1)
        kw={x.arg:x.value for x in calls[0].keywords}
        self.assertEqual(ast.unparse(kw['cwd']), "str(ROOT / 'repo')")
        self.assertEqual(ast.unparse(kw['check']),'True')

class AtomicPublish(unittest.TestCase):
    def test_new_link_never_overwrites(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as t:
            run=Path(t).resolve();logs=run/'logs';logs.mkdir();receipt=logs/'reviewed.json';receipt.write_text('actual signed bytes')
            release=logs/'step5-independent-release.json';sha=g.digest(receipt)
            original_owned=g.owned_file
            with patch.object(g,'RUN',run),patch.object(g,'owned_file',side_effect=lambda p:original_owned(p,run)):
                g.publish_link(receipt,release,sha)
                self.assertEqual(receipt.stat().st_ino,release.stat().st_ino)
                with self.assertRaises(ValueError):g.publish_link(receipt,release,sha)
                self.assertEqual(release.read_text(),'actual signed bytes')
    def test_changed_receipt_and_existing_dangling_symlink(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as t:
            run=Path(t).resolve();logs=run/'logs';logs.mkdir();receipt=logs/'reviewed.json';receipt.write_text('actual')
            release=logs/'step5-independent-release.json';original_owned=g.owned_file
            with patch.object(g,'RUN',run),patch.object(g,'owned_file',side_effect=lambda p:original_owned(p,run)):
                with self.assertRaises(ValueError):g.publish_link(receipt,release,'0'*64)
                self.assertFalse(release.exists())
                release.symlink_to(logs/'missing')
                with self.assertRaises(ValueError):g.publish_link(receipt,release,g.digest(receipt))
                self.assertTrue(release.is_symlink())
    def test_symlink_staging_refused(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as t:
            run=Path(t).resolve();logs=run/'logs';logs.mkdir();original=logs/'actual';original.write_text('x');receipt=logs/'symlink';receipt.symlink_to(original)
            original_owned=g.owned_file
            with patch.object(g,'RUN',run),patch.object(g,'owned_file',side_effect=lambda p:original_owned(p,run)):
                with self.assertRaises(ValueError):g.publish_link(receipt,logs/'step5-independent-release.json',g.digest(original))

if __name__=='__main__':unittest.main()
