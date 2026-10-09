"""Four focused CPU identity/initial/mutation/space regressions; no Torch."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).absolute().parent
DEPENDENCY=HERE.parent/'opsd_ablation_outer_backup_v2/backup.py'


def load(name):
    path=HERE/name;ns=dict(__name__='_fixture_'+name.replace('.','_'),__file__=str(path))
    exec(compile(path.read_bytes(),str(path),'exec'),ns)
    return ns


op=load('source-operation.py');mac=load('backup.py');b=op['primitives'](DEPENDENCY)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b['canonical'](value)+b'\n')
    return b['fileproof'](path)


class Protected(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        base=self.root/'group-R';output=base/'runs';output.mkdir(parents=True)
        cfg=dict(schema='opsd-ablation-config-v1',run_id='1'*32,variant='R',study_id='fixture-seven',
            scientific=dict(outer_iterations=10),precision=dict(mode='bf16'),
            paths=dict(base=str(base),output_root=str(output)),runtime_manifest=dict(sha256='a'*64),
            source_model_manifest=dict(sha256='b'*64),common_initial_adapter_manifest=dict(path=str(self.root/'common.json'),sha256='c'*64))
        cr=save(base/'receipts/config.json',cfg)
        groups=[]
        for i,v in enumerate(b['VARIANTS']):
            groups.append(dict(variant=v,run_id='1'*32 if v=='R' else ('%032x'%(i+20)),
                config=dict(path=cr['path'],sha256=cr['sha256']) if v=='R' else dict(path=str(self.root/('group-'+v)/'config.json'),sha256='d'*64),
                output_root=str(output) if v=='R' else str(self.root/('group-'+v)/'runs')))
        goal=dict(schema='opsd-ablation-outer-backup-goal-v1',accepted=True,goal_scope='seven-group-outer-end10',
            cluster='HKUST',explicit_node_local_with_mac_backup=True,study_id='fixture-seven',groups=groups)
        gr=save(self.root/'goal.json',goal)
        self.cfg=cfg;self.output=output
        self.identity=dict(schema='opsd-ablation-protected-identity-v1',variant='R',run_id='1'*32,
            config=dict(path=cr['path'],sha256=cr['sha256']),output_root=str(output),goal=dict(path=gr['path'],sha256=gr['sha256']))
        save(output/'run-identity.json',dict(schema='opsd-online-run-identity-v1',run_id='1'*32,config_sha256=cr['sha256']))
        self.checkpoints={}
        # u1/u5 are legitimately in outer1 after a hypothetical K0 outer0.
        for update,outer in [(0,0),(1,1),(5,1)]:
            directory=output/'checkpoints'/('update-%08d-outer-%05d'%(update,outer));directory.mkdir(parents=True)
            shared=directory/'optimizer-shared.pt'
            shared.write_bytes(b'{"state":{},"param_groups":[]}' if update==0 else b'opaque-committed-shared-Adam')
            ranks=[]
            for rank in range(8):
                p=directory/('rank-%d-state.pt'%rank);p.write_bytes(('original-RNG-scaler-rank%d'%rank).encode())
                ranks.append(dict(rank=rank,file=b['fileproof'](p)))
            adapter=directory/'adapter';adapter.mkdir()
            af=[]
            for name in ('README.md','adapter_config.json','adapter_model.safetensors'):
                p=adapter/name;p.write_bytes(('opaque-'+name).encode());af.append(b['fileproof'](p))
            cp=dict(schema='opsd-online-checkpoint-v1',accepted=True,run_id='1'*32,config_sha256=cr['sha256'],
                optimizer_updates=update,cursor=dict(outer_index=outer,next_pair_cursor=0 if update==0 else update,usable_pair_count=6),
                protection_reason=[{0:'initial',1:'update1',5:'update5'}[update]],
                contract=dict(direction='student_to_target_reverse',target_version='opsd-ablation-reverse-v1',variant='R',precision='bf16',mixture_alpha=0.),
                runtime_identity='a'*64,source_model_identity='b'*64,common_initial_adapter=cfg['common_initial_adapter_manifest'],
                complete_rank_count=8,equal_optimizer_steps=True,ranks=ranks,
                optimizer_storage='shared_exact_eight_rank_optimizer_v1',shared_optimizer=b['fileproof'](shared),
                all_rng=['python','numpy','torch_cpu','cuda_all'],all_rank_adapter_sha256=['e'*64]*8,
                all_rank_optimizer_tensor_sha256=['f'*64]*8,all_rank_scaler_state=[dict(scale=1.,enabled=False)]*8,
                adapter=dict(path=str(adapter),files=af))
            self.checkpoints[update]=(cp,save(directory/'manifest.json',cp))

    def tearDown(self):
        self.temp.cleanup()

    def scope(self):
        scope=op['pin'](b,self.identity)
        raw=b['canonical'](scope);return scope,hashlib.sha256(raw).hexdigest()

    def test_identity_rejection_and_missing_cp5_pending(self):
        wrong=dict(self.identity,run_id='2'*32)
        with self.assertRaises(ValueError):op['pin'](b,wrong)
        cp,ref=self.checkpoints[5]
        # Simulate not-yet-committed state without ever inferring it exists.
        directory=Path(ref['path']).parent
        directory.rename(directory.with_name(directory.name+'.pending-fixture'))
        pending=op['pin'](b,self.identity)
        self.assertFalse(pending['accepted']);self.assertEqual(pending['status'],'PENDING')
        self.assertEqual(pending['missing_updates'],[5])

    def test_cp0_empty_adam_exact13_and_actual_outer_schema(self):
        scope,sha=self.scope()
        self.assertEqual([r['outer_index'] for r in scope['protected']],[0,1,1])
        value=op['inventory'](b,scope,sha)
        expected=op['validate_inventory'](b,scope,sha,value)
        self.assertEqual(len(expected),42);self.assertEqual(value['core_file_count'],39)
        self.assertTrue(value['checkpoints'][0]['empty_adam_initial'])
        original=self.checkpoints[0][0]
        self.assertNotIn('outer_end',original['protection_reason'])
        self.assertEqual(sum('/update-00000000-' in name for name in expected),13)
        with self.assertRaises(ValueError):
            op['checkpoint_shape'](original,self.checkpoints[0][1],self.cfg,self.identity,0,-1)

    def test_mutation_rejects_scope_manifest_and_payload_source_after(self):
        scope,sha=self.scope();value=op['inventory'](b,scope,sha)
        invsha=hashlib.sha256(b['canonical'](value)).hexdigest()
        after=op['after'](b,scope,sha,value,invsha)
        mac['verify_after'](b,sha,dict(sha256=invsha),value,after)
        target=next(x for x in value['files'] if x['relative'].endswith('rank-7-state.pt'))
        Path(target['path']).write_bytes(b'changed-and-larger-payload')
        with self.assertRaises(ValueError):op['after'](b,scope,sha,value,invsha)
        cp,ref=self.checkpoints[5];cp['accepted']=False;save(Path(ref['path']),cp)
        with self.assertRaises(ValueError):op['inventory'](b,scope,sha)

    def test_space_backlog_never_claims_training_dependency(self):
        scope,sha=self.scope();value=op['inventory'](b,scope,sha)
        need=b['required_space'](value,5<<30)
        backlog=mac['require_space'](b,value,need-1)
        self.assertFalse(backlog['accepted']);self.assertEqual(backlog['status'],'BACKLOG')
        self.assertFalse(backlog['backup_is_training_dependency'])
        self.assertTrue(mac['require_space'](b,value,need)['accepted'])


if __name__=='__main__':unittest.main()
