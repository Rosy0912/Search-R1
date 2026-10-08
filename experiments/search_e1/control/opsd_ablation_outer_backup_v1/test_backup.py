"""Focused changed-scope/closure/queue CPU fixtures; no real SSH or tensors."""
import io
import json
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
b = types.ModuleType('backup_fixture'); b.__file__ = str(HERE / 'backup.py')
exec(compile((HERE / 'backup.py').read_bytes(), b.__file__, 'exec'), b.__dict__)
q = types.ModuleType('queue_fixture'); q.__file__ = str(HERE / 'queue.py')
exec(compile((HERE / 'queue.py').read_bytes(), q.__file__, 'exec'), q.__dict__)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else b.canonical(value) + b'\n')
    return b.fileproof(path)


def ref(proof): return {k: proof[k] for k in ('path', 'sha256')}


class Fixture:
    def __init__(self, root, outer=0, updates=0, K=0):
        self.base=root/'group-C0'; self.out=self.base/'runs'; self.run='a'*32
        self.outer=outer; self.updates=updates
        common=root/'common'
        initial=save(common/'manifest.json',dict(adapter_tensor_sha256='1'*64,payload_files=[
            save(common/n,n.encode()) for n in ('README.md','adapter_config.json','adapter_model.safetensors')]))
        ids={k:save(root/'identities'/(k+'.json'),{'identity':k}) for k in ('runtime_manifest','source_model_manifest','helper_source_manifest')}
        self.config=dict(run_id=self.run,study_id='synthetic-fixture',variant='C0',precision={'mode':'bf16'},scientific={'outer_iterations':10},
            paths=dict(base=str(self.base),output_root=str(self.out),model='/model/not-transferred',runtime='/runtime/not-transferred'),
            common_initial_adapter_manifest=ref(initial),common_initial_adapter_tensor_sha256='1'*64,**ids)
        self.cref=save(self.base/'receipts/config.json',self.config)
        save(self.out/'run-identity.json',dict(schema='opsd-online-run-identity-v1',run_id=self.run,config_sha256=self.cref['sha256'],config=self.cref))
        groups=[]
        for n,v in enumerate(b.VARIANTS):
            config=self.config if n==0 else dict(self.config,run_id=('%032x'%(n+100)),variant=v,
                paths=dict(self.config['paths'],base=str(root/('group-'+v)),output_root=str(root/('group-'+v)/'runs')))
            cr=self.cref if n==0 else save(root/('group-'+v)/'receipts/config.json',config)
            groups.append(dict(variant=v,run_id=config['run_id'],config=ref(cr),output_root=config['paths']['output_root']))
        self.goal=dict(schema='opsd-ablation-outer-backup-goal-v1',accepted=True,goal_scope='seven-group-outer-end10',
            cluster='HKUST',study_id='synthetic-fixture',explicit_node_local_with_mac_backup=True,groups=groups)
        self.gref=save(root/'goal.json',self.goal)
        od=self.out/'outers'/('outer-%05d'%outer)
        pairs=save(od/'sampling-attempt-00001/pairs.jsonl',b'fixture\n')
        inputs=dict(pairs=pairs,order=save(od/'training/order.json',{'pair_indices':list(range(K))}),
            encoding=save(od/'training/encoding.json',{'K':K,'rows':[]}))
        inputs['outer_manifest']=save(od/'outer-manifest.json',dict(run_id=self.run,config_sha256=self.cref['sha256'],outer_index=outer,pairs=pairs))
        self.cpdir=self.out/'checkpoints'/('update-%08d-outer-%05d'%(updates,outer))
        old=initial
        if outer:
            oldroot=self.out/'checkpoints'/('update-%08d-outer-%05d'%(updates-1,outer-1))
            old=save(oldroot/'manifest.json',dict(run_id=self.run,config_sha256=self.cref['sha256'],cursor={'outer_index':outer-1},
                adapter=dict(files=[save(oldroot/'adapter'/n,('old'+n).encode()) for n in ('README.md','adapter_config.json','adapter_model.safetensors')])))
        shared=save(self.cpdir/'optimizer-shared.pt',b'actual opaque full optimizer fixture')
        self.cp=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id=self.run,config_sha256=self.cref['sha256'],
            optimizer_updates=updates,cursor=dict(outer_index=outer,next_pair_cursor=K,usable_pair_count=K),protection_reason=['outer_end'],
            contract=dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',variant='C0',precision='bf16',mixture_alpha=0.),
            common_initial_adapter=ref(initial),outer_old_adapter=ref(old),all_rank_scaler_state=[{'enabled':False}]*8,
            all_rng=['python','numpy','torch_cpu','cuda_all'],optimizer_storage='shared_exact_eight_rank_optimizer_v1',
            shared_optimizer=shared,all_rank_optimizer_tensor_sha256=['2'*64]*8,complete_rank_count=8,equal_optimizer_steps=True,
            ranks=[dict(rank=r,file=save(self.cpdir/('rank-%d-state.pt'%r),('RNG/scaler-rank%d'%r).encode())) for r in range(8)],
            adapter=dict(path=str(self.cpdir/'adapter'),files=[save(self.cpdir/'adapter'/n,n.encode()) for n in ('README.md','adapter_config.json','adapter_model.safetensors')]),
            training_input=inputs,runtime_identity=ids['runtime_manifest']['sha256'],source_model_identity=ids['source_model_manifest']['sha256'])
        self.cpref=save(self.cpdir/'manifest.json',self.cp)
        self.complete=save(od/'training/outer-complete.json',dict(schema='opsd-online-outer-complete-v1',accepted=True,run_id=self.run,
            outer_index=outer,optimizer_updates=updates,optimizer_updates_delta=(K+31)//32,K=K,checkpoint=self.cpref,training_input=inputs))
    def args(self): return (self.cref['path'],self.cref['sha256'],self.cpref['path'],self.cpref['sha256'],self.run,self.updates,self.outer)
    def inv(self): return b.inventory(*self.args(),goal_ref=self.gref,completion_ref=ref(self.complete))


class Changes(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name).resolve()
    def tearDown(self): self.tmp.cleanup()
    def test_outer0_k0_zero_updates_full_shared_state_closure(self):
        f=Fixture(self.root); v=f.inv(); self.assertTrue(v['empty_outer_no_new_adam']); self.assertEqual(v['optimizer_updates'],0)
        self.assertEqual(v['core_file_count'],19); self.assertEqual(len({x['path'] for x in v['files']}),len(v['files']))
        expected=b.validate_inventory(v,*f.args(),f.gref,ref(f.complete)); buff=io.BytesIO(); b.stream_archive(v,buff)
        archive=self.root/'out.tar'; archive.write_bytes(buff.getvalue())
        local,_=b.extract_verified(archive,self.root/'extracted',expected)
        self.assertIn('group/runs/checkpoints/update-00000000-outer-00000/optimizer-shared.pt',local)
        self.assertEqual(stat.S_IMODE((self.root/'extracted').stat().st_mode),0o700)
        self.assertTrue(any(x.startswith('closure/common/') for x in local))
    def test_final9_nonzero_updates_old_adapter_original_bytes_included(self):
        f=Fixture(self.root,outer=9,updates=91,K=1); v=f.inv()
        self.assertFalse(v['empty_outer_no_new_adam']); self.assertTrue(any(x['relative'].startswith('closure/outer-old/payload-') for x in v['files']))
        b.validate_inventory(v,*f.args(),f.gref,ref(f.complete))
    def test_outer_outside0_9_wrong_run_and_legacy_scope_rejected(self):
        f=Fixture(self.root)
        for run,outer,goal in [(f.run,10,f.goal),('f'*32,0,f.goal),(f.run,0,dict(f.goal,goal_scope='ablation-final10'))]:
            with self.subTest(outer=outer,run=run),self.assertRaises(b.Rejected): b.target_scope(run,0,outer,f.cpref['sha256'],goal)
    def test_changed_shared_or_old_closure_rejects(self):
        f=Fixture(self.root,outer=1,updates=10,K=32); old,_=b.checked_json(f.cp['outer_old_adapter']['path'],f.cp['outer_old_adapter']['sha256'])
        Path(old['adapter']['files'][0]['path']).write_bytes(b'mutated')
        with self.assertRaises(b.Rejected): f.inv()
    def test_completion_mutation_with_same_checkpoint_rejected(self):
        f=Fixture(self.root); pinned=q.discover(f.goal)[0]; cp_before=b.fileproof(f.cpref['path'])['sha256']
        value,_=b.checked_json(f.complete['path'],f.complete['sha256']); value['added_after_pin']='changed completion bytes'
        save(Path(f.complete['path']),value)
        self.assertEqual(b.fileproof(f.cpref['path'])['sha256'],cp_before)
        self.assertEqual(pinned['checkpoint']['sha256'],cp_before)
        with self.assertRaises(b.Rejected): f.inv()
    def test_success_and_restart_check_saved_exact_completion_closure(self):
        f=Fixture(self.root); value=f.inv(); pub={'manifest':{'sha256':'a'*64},'source_commit':'b'*40,'publication':{'sha256':'c'*64}}
        value['source_publication']=pub
        args=types.SimpleNamespace(destination=str(self.root/'successful'),config=f.cref['path'],config_sha256=f.cref['sha256'],
            checkpoint=f.cpref['path'],checkpoint_sha256=f.cpref['sha256'],expected_run=f.run,expected_updates=0,expected_outer=0,
            remote_goal_identity=f.gref['path'],goal_identity=f.gref['path'],goal_identity_sha256=f.gref['sha256'],margin_gib=5,
            expected_completion=f.complete['path'],expected_completion_sha256=f.complete['sha256'])
        class Child:
            pid=123; stdin=None; returncode=0
            def communicate(self,data): pass
            def wait(self): return self.returncode
        def popen(command,**kwargs): b.stream_archive(value,kwargs['stdout']); return Child()
        result=types.SimpleNamespace(returncode=0,stdout=b.canonical(value),stderr=b'')
        with patch.object(b,'remote_command',return_value=['not-executed']),patch.object(b.subprocess,'run',return_value=result),\
                patch.object(b.subprocess,'Popen',side_effect=popen),patch.object(b.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**12)):
            self.assertTrue(b.backup(args,pub)['accepted'])
        complete_path=Path(args.destination)/'complete-sha-verified.json'
        item=dict(job=q.discover(f.goal)[0],complete_receipt=b.fileproof(complete_path))
        q.verified_complete(item)
        complete=json.loads(complete_path.read_bytes()); invpath=Path(complete['node_inventory']['path']); inv=json.loads(invpath.read_bytes())
        inv['outer_complete']['sha256']='f'*64
        complete['node_inventory']=save(invpath,inv); item['complete_receipt']=save(complete_path,complete)
        with self.assertRaises(q.b.Rejected): q.verified_complete(item)
    def test_shared_optimizer_required_and_no_added_checkpoint_files(self):
        f=Fixture(self.root); save(f.cpdir/'unexpected.pt',b'unexpected')
        with self.assertRaises(b.Rejected): f.inv()
    def test_discovery_only_complete_and_keeps_actual_k0_count(self):
        f=Fixture(self.root); jobs=q.discover(f.goal); self.assertEqual(len(jobs),1); self.assertEqual(jobs[0]['optimizer_updates'],0)
        self.assertTrue(jobs[0]['empty_outer_no_new_adam']); self.assertNotIn('latest',jobs[0]['checkpoint']['path'])
        q.validate_discovery(dict(schema='opsd-ablation-outer-backup-discovery-v1',accepted=True,jobs=jobs),f.goal)
    def test_queue_pinning_dedup_and_changed_cp_reject(self):
        f=Fixture(self.root); jobs=q.discover(f.goal); state=q.merge_jobs({},jobs)
        self.assertEqual(len(q.merge_jobs(state,jobs)),1)
        changed=dict(jobs[0],checkpoint=dict(jobs[0]['checkpoint'],sha256='f'*64))
        with self.assertRaises(q.b.Rejected): q.merge_jobs(state,[changed])
    def test_backlog_nonzero_retains_partial_not_complete(self):
        f=Fixture(self.root); value=f.inv(); pub={'manifest':{'sha256':'a'*64},'source_commit':'b'*40,'publication':{'sha256':'c'*64}}
        value['source_publication']=pub
        args=types.SimpleNamespace(destination=str(self.root/'backup'),config=f.cref['path'],config_sha256=f.cref['sha256'],
            checkpoint=f.cpref['path'],checkpoint_sha256=f.cpref['sha256'],expected_run=f.run,expected_updates=0,expected_outer=0,
            remote_goal_identity=f.gref['path'],goal_identity=f.gref['path'],goal_identity_sha256=f.gref['sha256'],margin_gib=5,expected_completion=f.complete['path'],expected_completion_sha256=f.complete['sha256'])
        class Child:
            pid=123; stdin=None; returncode=255
            def communicate(self,data): pass
            def wait(self): return self.returncode
        result=types.SimpleNamespace(returncode=0,stdout=b.canonical(value),stderr=b'')
        with patch.object(b,'remote_command',return_value=['not-executed']),patch.object(b.subprocess,'run',return_value=result),patch.object(b.subprocess,'Popen',return_value=Child()),patch.object(b.shutil,'disk_usage',return_value=types.SimpleNamespace(free=10**12)):
            with self.assertRaises(b.Rejected): b.backup(args,pub)
        dest=Path(args.destination); self.assertTrue((dest/'checkpoint.tar.partial').exists()); self.assertFalse((dest/'complete-sha-verified.json').exists())
        self.assertEqual(json.loads((dest/'transfer-wait.json').read_text())['actual_returncode'],255)
    def test_disk_budget_is_two_copies_plus_at_least5gib(self):
        f=Fixture(self.root); v=f.inv(); need=b.required_space(v,5<<30)
        self.assertGreaterEqual(need,2*v['total_bytes']+(5<<30))
        with self.assertRaises(b.Rejected): b.required_space(v,1<<30)
    def test_queue_space_failure_is_backlog_and_never_blocks_training(self):
        f=Fixture(self.root); jobs=q.discover(f.goal)
        args=types.SimpleNamespace(destination=str(self.root/'mac-queue'),goal_identity_sha256=f.gref['sha256'],
            source_manifest_sha256='a'*64,publication_sha256='b'*64,source_commit='c'*40,once=True,
            interval_seconds=60)
        discovery=dict(schema='opsd-ablation-outer-backup-discovery-v1',accepted=True,jobs=jobs,
            goal_sha256=f.gref['sha256'],source_manifest_sha256='a'*64,source_commit='c'*40)
        result=types.SimpleNamespace(returncode=0,stdout=b.canonical(discovery),stderr=b'')
        with patch.object(q,'discovery_command',return_value=['not-executed']),patch.object(q.subprocess,'run',return_value=result),\
                patch.object(q.b,'backup',side_effect=q.b.Rejected('Insufficient local disk')):
            state=q.queue(args,f.goal,{})
        item=next(iter(state['jobs'].values())); self.assertEqual(item['state'],'BACKLOG')
        self.assertTrue(state['training_not_blocked']); self.assertEqual(len(item['attempts']),1)
        self.assertGreater(item['retry_after_unix'],state['updated_unix'])
    def test_queue_remote_command_independent_cpu_marker_no_gpu_lock(self):
        args=types.SimpleNamespace(host='hkust1',remote_helper_root='/published/new',goal_identity_sha256='a'*64,
            remote_goal_identity='/goal.json',source_manifest_sha256='b'*64,remote_publication='/pub.json',publication_sha256='c'*64,source_commit='d'*40)
        cmd=q.discovery_command(args); self.assertIn('CUDA_VISIBLE_DEVICES=',cmd[-1]); self.assertIn('E1_RUN_ID=backup-',cmd[-1])
        self.assertNotIn('experiment.lock',cmd[-1])


if __name__=='__main__': unittest.main()
