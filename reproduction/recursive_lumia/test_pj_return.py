import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from .pj_return_contracts import PROFILES,PARENT_SOURCE,validate_parent,project_config,validate_projection,mapped_manifest
from .pj_return import validate_probe

class ReturnContracts(unittest.TestCase):
    def fixture(self,group='budget-5'):
        p=PROFILES[group];n=p['step']*20-7
        parent=dict(group=group,run_id=p['run_id'],source_commit=PARENT_SOURCE,config_sha256=p['config'],data_identity='data')
        entry=dict(parent,allocation=p['job'])
        identity=dict(job_id=p['job'],code_commit=PARENT_SOURCE,config_sha256=p['config'],data_identity='data',
            completed_step=p['step'],actual_update_limit=150,guard_continue_accounting=dict(total_optimizer_calls_per_rank={str(i):n for i in range(8)}))
        workers=[dict(identity=dict(identity,rank=i,world_size=8),actual_total_optimizer_calls=n,
            observed_adam_steps=[n],scheduler_last_epoch=p['step'],full_worker_state_finite=True,
            path='/old/rank-%02d.pt'%i,sha256='sha'+str(i),logical_state_sha256='logical'+str(i)) for i in range(8)]
        manifest=dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',identity=identity,worker_states=workers,
            driver=dict(next_step=p['step']+1,path='/old/driver.pt',sha256='driver',logical_state_sha256='driverlogical'))
        return parent,entry,manifest,group
    def test_observed_adam_counters_preserved_not_inferred(self):
        for group in PROFILES:validate_parent(*self.fixture(group))
    def test_wrong_parent_or_incomplete_native_rejected(self):
        for group in PROFILES:
            for mode in ['source','config','job','step','rank','finite','adam','cursor']:
                p,e,m,g=self.fixture(group)
                if mode=='source':p['source_commit']='wrong'
                if mode=='config':m['identity']['config_sha256']='wrong'
                if mode=='job':e['allocation']='wrong'
                if mode=='step':m['identity']['completed_step']-=1
                if mode=='rank':m['worker_states'].pop()
                if mode=='finite':m['worker_states'][0]['full_worker_state_finite']=False
                if mode=='adam':m['worker_states'][0]['observed_adam_steps']=[0]
                if mode=='cursor':m['driver']['next_step']-=1
                with self.assertRaises(ValueError):validate_parent(p,e,m,g)
    def test_mapping_changes_only_paths(self):
        _,_,m,_=self.fixture();before=copy.deepcopy(m)
        members=[dict(name=Path(x['path']).name,original=x['path'],sha256=x['sha256']) for x in m['worker_states']+[m['driver']]]
        mapped=mapped_manifest(m,'/stage',members)
        self.assertEqual(m,before)
        for a,b in zip(mapped['worker_states']+[mapped['driver']],m['worker_states']+[m['driver']]):
            self.assertEqual(a,dict(b,path='/stage/'+Path(b['path']).name))
        for bad in ['sha','name','duplicate','original']:
            x=copy.deepcopy(members)
            if bad=='sha':x[0]['sha256']='bad'
            if bad=='name':x[0]['name']='bad'
            if bad=='duplicate':x[0]=x[1]
            if bad=='original':x[0]['original']='/wrong/rank-00.pt'
            with self.assertRaises(ValueError):mapped_manifest(m,'/stage',x)
    def config(self):
        return dict(baselines=dict(local_runtime_host='old',holdout=dict(manifest_path='old',data_root='old'),bapo=dict(stage_validation_files='old')),
            data=dict(train_files='old',val_files='old',final_val_files='old'),actor_rollout_ref=dict(model=dict(path='old'),ref=dict(model_path='old'),actor=dict(lr=1e-6)),
            critic=dict(model=dict(tokenizer_path='old')),reward_model=dict(model=dict(input_tokenizer='old')),trainer=dict(default_local_dir='old',actual_update_limit=150))
    def test_only_explicit_paths_and_host_change(self):
        for group in PROFILES:
            c=self.config();p=project_config(c,group)
            with patch('release.recursive_bran_a.ablation150_v1.contracts.validate_config_dict'):
                validate_projection(c,p,group)
                for mutate in ['lr','limit','extra','model']:
                    bad=copy.deepcopy(p)
                    if mutate=='lr':bad['actor_rollout_ref']['actor']['lr']=1e-5
                    if mutate=='limit':bad['trainer']['actual_update_limit']=149
                    if mutate=='extra':bad['new_science']=True
                    if mutate=='model':bad['actor_rollout_ref']['model']['path']='/other'
                    with self.assertRaises(ValueError):validate_projection(c,bad,group)
    def test_probe_gate_requires_real_full_state_equality(self):
        _,_,m,g=self.fixture();b=dict(group=g,source_commit='new',metadata_sha256={'parent-manifest.json':'hash'})
        probe=dict(status='INDEPENDENT_EIGHT_GPU_COLD_STATE_EQUALITY_VERIFIED',job='123',probe_source='new',parent_manifest_sha256='hash',
            completed_outer=PROFILES[g]['step'],training_updates_executed=0,driver_logical_sha256='driverlogical',
            ranks=[dict(rank=i,allocated_visible_device=str(i),actual_state_equal=True,observed_adam_steps=w['observed_adam_steps'],
                scheduler_step=w['scheduler_last_epoch'],live_logical_sha256=w['logical_state_sha256'],parent_sha256=w['sha256']) for i,w in enumerate(m['worker_states'])])
        validate_probe(probe,b,m,'123')
        for mode in ['missing','source','job','adam','driver','updates','device']:
            p=copy.deepcopy(probe)
            if mode=='missing':p['ranks'].pop()
            if mode=='source':p['probe_source']='wrong'
            if mode=='job':p['job']='124'
            if mode=='adam':p['ranks'][0]['observed_adam_steps']=[0]
            if mode=='driver':p['driver_logical_sha256']='wrong'
            if mode=='updates':p['training_updates_executed']=1
            if mode=='device':p['ranks'][0]['allocated_visible_device']='1'
            with self.assertRaises(ValueError):validate_probe(p,b,m,'123')

if __name__=='__main__':unittest.main()

class DispatchAdmission(unittest.TestCase):
    def test_partial_files_never_make_boundary_ready(self):
        import tempfile,json
        from . import pj_return_dispatch as d
        _,_,m,g=ReturnContracts().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'migrations'/PROFILES[g]['stage'];stage.mkdir(parents=True)
            (stage/'parent-manifest.json').write_text(json.dumps(m))
            for row in m['worker_states']+[m['driver']]:(stage/(Path(row['path']).name+'.part')).write_bytes(b'x')
            with patch.object(d,'BASE',root):
                self.assertFalse(d.ready(g))
                for f in stage.glob('*.part'):f.rename(f.with_suffix(''))
                self.assertTrue(d.ready(g))
    def test_uncertain_submission_never_replayed(self):
        import tempfile
        from . import pj_return_dispatch as d
        g='budget-5'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'migrations'/PROFILES[g]['stage'];stage.mkdir(parents=True)
            (stage/'return-submit-intent.json').write_text('{}')
            with patch.object(d,'BASE',root),patch.object(d.subprocess,'run') as call,patch.object(d,'prepare') as prepare:
                with self.assertRaises(ValueError):d.submit(g,'a'*40)
                call.assert_not_called();prepare.assert_not_called()
