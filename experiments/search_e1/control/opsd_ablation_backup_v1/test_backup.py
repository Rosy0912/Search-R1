"""Small synthetic CPU files only; no SSH/transfer/model imports."""
import io
import json
from pathlib import Path
import shlex
import stat
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
b = types.ModuleType('ablation_backup'); b.__file__ = str(HERE/'backup.py')
exec(compile((HERE/'backup.py').read_bytes(), b.__file__, 'exec'), b.__dict__)


class Fixture:
    def __init__(self, root):
        self.base = root/'node'; self.base.mkdir()
        self.out = self.base/'runs'; self.out.mkdir()
        self.cpdir = self.out/'checkpoints/update-00000091-outer-00009'; self.cpdir.mkdir(parents=True)
        self.outer = self.out/'outers/outer-00009'; self.outer.mkdir(parents=True)
        self.run = 'a'*32
        refs = {k:self.save(self.base/'receipts'/(k+'.json'), {'identity':k}) for k in
                ('runtime_manifest','source_model_manifest','helper_source_manifest','common_initial_adapter_manifest')}
        self.config = dict(run_id=self.run, paths=dict(base=str(self.base),output_root=str(self.out)),
            variant='C0',precision={'mode':'bf16'},scientific={'outer_iterations':10},**refs)
        self.config_path = self.base/'receipts/config.json'; self.cref=self.save(self.config_path,self.config)
        self.save(self.out/'run-identity.json',dict(schema='opsd-online-run-identity-v1',run_id=self.run,
            config_sha256=self.cref['sha256'],config=self.cref))
        pairs=self.save(self.outer/'sampling-attempt-00000/pairs.jsonl',b'fixture\n')
        inputs=dict(pairs=pairs,order=self.save(self.outer/'training/order.json',{'pair_indices':[0]}),
            encoding=self.save(self.outer/'training/encoding.json',{'rows':['fixture']}))
        inputs['outer_manifest']=self.save(self.outer/'outer-manifest.json',dict(run_id=self.run,
            config_sha256=self.cref['sha256'],outer_index=9,pairs=pairs,
            runtime_identity=refs['runtime_manifest']['sha256'],source_model_identity=refs['source_model_manifest']['sha256']))
        self.cp=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=self.run,
            config_sha256=self.cref['sha256'],optimizer_updates=91,
            runtime_identity=refs['runtime_manifest']['sha256'],source_model_identity=refs['source_model_manifest']['sha256'],
            contract=dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',variant='C0',precision='bf16',mixture_alpha=0.),
            common_initial_adapter=refs['common_initial_adapter_manifest'],
            cursor=dict(outer_index=9,next_pair_cursor=1,usable_pair_count=1),protection_reason=['outer_end'],
            complete_rank_count=8,equal_optimizer_steps=True,
            ranks=[dict(rank=i,file=self.save(self.cpdir/('rank-%d-state.pt'%i),('rank%d'%i).encode())) for i in range(8)],
            adapter=dict(path=str(self.cpdir/'adapter'),files=[self.save(self.cpdir/'adapter'/n,n.encode()) for n in
                ('README.md','adapter_config.json','adapter_model.safetensors')]),training_input=inputs)
        self.cpref=self.save(self.cpdir/'manifest.json',self.cp)
        self.final_ref=self.save(self.outer/'training/outer-complete.json',dict(schema='opsd-online-outer-complete-v1',
            accepted=True,run_id=self.run,outer_index=9,optimizer_updates=91,checkpoint=self.cpref))
        self.goal=dict(schema='opsd-ablation-backup-goal-v1',accepted=True,goal_created_unix=b.GOAL_CREATED_UNIX,
            goal_scope='ablation-final10',variant='C0',run_id=self.run,optimizer_updates=91,outer_index=9,
            config=self.minref(self.cref),checkpoint=self.minref(self.cpref),final_completion=self.final_ref,
            source_job_exit={'classification':'NOT_AUDITED_BY_BACKUP'})
        self.goalpath=self.base/'receipts/backup-goal.json';self.gref=self.save(self.goalpath,self.goal)

    @staticmethod
    def minref(ref):return {k:ref[k] for k in ('path','sha256')}
    @staticmethod
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else b.canonical(value)+b'\n')
        return b.fileproof(path)
    def args(self):return (str(self.config_path),self.cref['sha256'],str(self.cpdir/'manifest.json'),self.cpref['sha256'],self.run,91,9)
    def inv(self):return b.inventory(*self.args(),goal_ref=self.gref)
    def goal_update(self):self.gref=self.save(self.goalpath,self.goal)


class Backup(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.f=Fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def test_exact_18_full_sha_private_disk_roundtrip(self):
        value=self.f.inv();expected=b.validate_inventory(value,*self.f.args(),self.f.gref)
        buffer=io.BytesIO();b.stream_archive(value,buffer)
        archive=self.root/'fixture.tar';archive.write_bytes(buffer.getvalue())
        local,_=b.extract_verified(archive,self.root/'out',expected)
        self.assertEqual(len(local),18);self.assertEqual(stat.S_IMODE((self.root/'out').stat().st_mode),0o700)
        self.assertEqual(value['goal_identity']['value']['source_job_exit']['classification'],'NOT_AUDITED_BY_BACKUP')
    def test_scoped_target_wrong_group_counter_or_sha_rejected(self):
        for key,value in [('variant','eighth'),('outer_index',8),('optimizer_updates',92),('goal_created_unix',0)]:
            goal=dict(self.f.goal);goal[key]=value
            with self.subTest(key=key),self.assertRaises(b.Rejected):b.target_scope(self.f.run,91,9,self.f.cpref['sha256'],goal)
    def test_old_current143_not_historical_unknown_or_zero(self):
        goal=dict(self.f.goal,goal_scope='old-online374',run_id=b.RUN,optimizer_updates=374,outer_index=37,
            checkpoint={'path':'/explicit/manifest.json','sha256':b.OLD_CP374_SHA},historical_wait_exit_status='UNKNOWN',
            source_job_exit=dict(classification='authorized_signal_stop',actual_tmux_status=143,receipt={'sha256':b.OLD_TERMINAL_SHA}))
        self.assertEqual(b.target_scope(b.RUN,374,37,b.OLD_CP374_SHA,goal),'old-online374')
        for status in (0,None):
            changed=dict(goal,source_job_exit=dict(goal['source_job_exit'],actual_tmux_status=status))
            with self.assertRaises(b.Rejected):b.target_scope(b.RUN,374,37,b.OLD_CP374_SHA,changed)
    def test_missing_final_or_k0_prior_checkpoint_hold(self):
        final,_=b.checked_json(self.f.final_ref['path'],self.f.final_ref['sha256']);final['checkpoint']=dict(final['checkpoint'],sha256='0'*64)
        self.f.goal['final_completion']=self.f.save(Path(self.f.final_ref['path']),final);self.f.goal_update()
        with self.assertRaises(b.Rejected):self.f.inv()
    def test_model_contract_and_actual_source_mutation_reject(self):
        self.f.cp['contract']['direction']='teacher_to_student'
        self.f.cpref=self.f.save(self.f.cpdir/'manifest.json',self.f.cp);self.f.goal['checkpoint']=self.f.minref(self.f.cpref);self.f.goal_update()
        with self.assertRaises(b.Rejected):self.f.inv()
    def test_member_wrong_sha_rejected(self):
        row=self.f.cp['ranks'][0]['file'];Path(row['path']).write_bytes(b'changed')
        with self.assertRaises(b.Rejected):self.f.inv()
    def test_stream_changed_source_fails_and_local_partial_retained(self):
        value=self.f.inv();Path(value['files'][0]['path']).write_bytes(b'changed')
        with self.assertRaises(b.Rejected):b.stream_archive(value,io.BytesIO())
    def test_tar_hidden_gnu_duplicate_and_traversal_reject(self):
        expected={'x':{'bytes':1,'sha256':b.hashlib.sha256(b'x').hexdigest()}}
        for name in ('../x','x'):
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w',format=tarfile.GNU_FORMAT) as tar:
                item=tarfile.TarInfo(name);item.size=1;tar.addfile(item,io.BytesIO(b'x'))
                if name=='x':tar.addfile(item,io.BytesIO(b'x'))
            path=self.root/('bad'+str(len(list(self.root.iterdir())))+'.tar');path.write_bytes(buffer.getvalue())
            with self.assertRaises(b.Rejected):b.extract_verified(path,self.root/('extract'+path.stem),expected)
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w',format=tarfile.GNU_FORMAT) as tar:
            item=tarfile.TarInfo('x'*150);item.size=1;tar.addfile(item,io.BytesIO(b'x'))
        path=self.root/'gnu.tar';path.write_bytes(buffer.getvalue())
        with self.assertRaises(b.Rejected):b.extract_verified(path,self.root/'gnuout',{'x'*150:expected['x']})
    def test_remote_argv_explicit_scope_and_no_latest(self):
        a=types.SimpleNamespace(host='hkust1',remote_helper_root='/explicit/source',config='/explicit/config.json',config_sha256='a'*64,
            checkpoint='/explicit/cp/manifest.json',checkpoint_sha256='b'*64,expected_run=self.f.run,expected_updates=91,expected_outer=9,
            remote_goal_identity='/explicit/goal.json',goal_identity_sha256='c'*64,source_manifest_sha256='d'*64,
            remote_publication='/explicit/pub.json',publication_sha256='e'*64,source_commit='f'*40)
        command=b.remote_command(a,'stream');remote=shlex.split(command[-1]);self.assertIn('/explicit/goal.json',remote)
        self.assertIn('CUDA_VISIBLE_DEVICES=',remote);self.assertNotIn('latest',' '.join(command))
    def test_real_nonzero_transfer_keeps_partial_and_never_accepts(self):
        value=self.f.inv();pub={'manifest':{'sha256':'a'*64},'source_commit':'b'*40,'publication':{'sha256':'c'*64}}
        value['source_publication']=pub
        a=types.SimpleNamespace(destination=str(self.root/'backup'),config=str(self.f.config_path),config_sha256=self.f.cref['sha256'],
            checkpoint=str(self.f.cpdir/'manifest.json'),checkpoint_sha256=self.f.cpref['sha256'],expected_run=self.f.run,
            expected_updates=91,expected_outer=9,remote_goal_identity=str(self.f.goalpath),goal_identity=str(self.f.goalpath),
            goal_identity_sha256=self.f.gref['sha256'],margin_gib=5)
        class Child:
            pid=123;stdin=None;returncode=23
            def communicate(self,data):pass
            def wait(self):return self.returncode
        result=types.SimpleNamespace(returncode=0,stdout=b.canonical(value),stderr=b'')
        with patch.object(b,'remote_command',return_value=['not-executed']),patch.object(b.subprocess,'run',return_value=result),patch.object(b.subprocess,'Popen',return_value=Child()),patch.object(b.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**12)):
            with self.assertRaises(b.Rejected):b.backup(a,pub)
        dest=Path(a.destination);self.assertTrue((dest/'checkpoint.tar.partial').exists());self.assertFalse((dest/'complete-sha-verified.json').exists())
        self.assertEqual(json.loads((dest/'transfer-exit.json').read_text())['actual_returncode'],23)
        self.assertEqual(json.loads((dest/'transfer-wait.json').read_text())['actual_returncode'],23)
        self.assertEqual(stat.S_IMODE(dest.stat().st_mode),0o700)

    def test_success_requires_actual_zero_and_independent_local_readback(self):
        value=self.f.inv();pub={'manifest':{'sha256':'a'*64},'source_commit':'b'*40,'publication':{'sha256':'c'*64}}
        value['source_publication']=pub
        a=types.SimpleNamespace(destination=str(self.root/'success'),config=str(self.f.config_path),config_sha256=self.f.cref['sha256'],
            checkpoint=str(self.f.cpdir/'manifest.json'),checkpoint_sha256=self.f.cpref['sha256'],expected_run=self.f.run,
            expected_updates=91,expected_outer=9,remote_goal_identity=str(self.f.goalpath),goal_identity=str(self.f.goalpath),
            goal_identity_sha256=self.f.gref['sha256'],margin_gib=5)
        class Child:
            pid=123;stdin=None;returncode=0
            def communicate(self,data):pass
            def wait(self):return self.returncode
        def popen(command,**kwargs):
            b.stream_archive(value,kwargs['stdout']);return Child()
        result=types.SimpleNamespace(returncode=0,stdout=b.canonical(value),stderr=b'')
        with patch.object(b,'remote_command',return_value=['not-executed']),patch.object(b.subprocess,'run',return_value=result),patch.object(b.subprocess,'Popen',side_effect=popen),patch.object(b.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**12)):
            self.assertTrue(b.backup(a,pub)['accepted'])
        complete=json.loads((Path(a.destination)/'complete-sha-verified.json').read_text())
        self.assertTrue(complete['independent_local_disk_readback']);self.assertEqual(complete['actual_transfer_returncode'],0)
        self.assertEqual(complete['goal_identity']['value']['source_job_exit']['classification'],'NOT_AUDITED_BY_BACKUP')
        self.assertTrue((Path(a.destination)/'checkpoint.tar').exists())
        with self.assertRaises(b.Rejected):b.backup(a,pub)

    def test_actual_symlink_rejected(self):
        row=self.f.cp['ranks'][0]['file'];path=Path(row['path']);other=path.with_name('alias')
        path.rename(other);path.symlink_to(other)
        with self.assertRaises(b.Rejected):self.f.inv()


    def test_shared_optimizer_optional_exact19_and_digest_mutation_rejected(self):
        shared=self.f.save(self.f.cpdir/'optimizer-shared.pt',b'complete synthetic Adam')
        self.f.cp['optimizer_storage']='shared_exact_eight_rank_optimizer_v1';self.f.cp['shared_optimizer']=shared;self.f.cp['all_rank_optimizer_tensor_sha256']=['a'*64]*8
        self.f.cpref=self.f.save(self.f.cpdir/'manifest.json',self.f.cp)
        completion=dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id=self.f.run,outer_index=9,
            optimizer_updates=91,checkpoint=self.f.cpref)
        self.f.goal['checkpoint']=self.f.minref(self.f.cpref)
        self.f.goal['final_completion']=self.f.save(Path(self.f.final_ref['path']),completion);self.f.goal_update()
        value=self.f.inv();self.assertEqual(len(value['files']),19)
        self.assertEqual(len(b.validate_inventory(value,*self.f.args(),self.f.gref)),19)
        Path(shared['path']).write_bytes(b'altered optimizer')
        with self.assertRaises(b.Rejected):self.f.inv()


if __name__=='__main__':unittest.main()
