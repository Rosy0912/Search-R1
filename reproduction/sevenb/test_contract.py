import copy
import json
from pathlib import Path
import tempfile
import unittest
from .contract import PROFILE, REVISION, ROOT, model_contract, make_config, models, model_path, validate_plan
from reproduction.seven_dataset.contracts import digest


class Contracts(unittest.TestCase):
    def config(self):
        return make_config(ROOT/'runs/original',model_path('base'),'gpu-ada6000-1')

    def test_7b_explicit_optin_and_fixed_original_science(self):
        from search_r1.baselines.tree_fresh200 import validate_recursive_config
        c=self.config();validate_recursive_config(c)
        self.assertTrue(model_contract(c.baselines))
        for field,value in [('model_revision','wrong'),('scale_profile','wrong'),('model_repository','wrong')]:
            q=copy.deepcopy(c);q.baselines[field]=value
            with self.assertRaises(ValueError):validate_recursive_config(q)
        q=copy.deepcopy(c);del q.baselines.scale_profile
        with self.assertRaises(ValueError):validate_recursive_config(q)
        q=copy.deepcopy(c);q.actor_rollout_ref.actor.optim.lr=2e-6
        with self.assertRaises(ValueError):validate_recursive_config(q)
        q=copy.deepcopy(c);q.baselines.tree.continuation_budget_per_question=5
        with self.assertRaises(ValueError):validate_recursive_config(q)

    def test_3b_contract_still_valid(self):
        from omegaconf import OmegaConf
        from search_r1.baselines.tree_fresh200 import validate_recursive_config
        c=OmegaConf.load(Path(__file__).with_name('original-template.yaml'))
        validate_recursive_config(c)

    def test_only_input_identity_changes(self):
        from omegaconf import OmegaConf
        a=OmegaConf.to_container(OmegaConf.load(Path(__file__).with_name('original-template.yaml')),resolve=True)
        b=OmegaConf.to_container(self.config(),resolve=True)
        def diffs(a,b,p=''):
            if isinstance(a,dict) and isinstance(b,dict):
                return sum((diffs(a.get(k),b.get(k),p+'.'+k) for k in a.keys()|b.keys()),[])
            return [] if a==b else [p[1:]]
        self.assertEqual(set(diffs(a,b)),{
            'actor_rollout_ref.model.path','actor_rollout_ref.ref.model_path','critic.model.tokenizer_path',
            'reward_model.model.input_tokenizer','baselines.model_repository','baselines.model_revision',
            'baselines.scale_profile','baselines.local_runtime_host','trainer.experiment_name','trainer.default_local_dir'})

    def plan(self,kind='base',repeat=1):
        label=kind if kind in ('base','searchr1') else 'base';m=models()[label]
        files={f['file']:f['sha256'] for f in m['files']}
        root=ROOT/'runs'/kind if repeat==0 else ROOT/'evaluations'/kind/('r'+str(repeat))
        return dict(profile=PROFILE,base=str(ROOT),kind=kind,repeat_index=repeat,model_label=label,
            root=str(root),output=str(root/'output'),job='zhangjunming-7b-'+kind+'-'+str(repeat),
            model=str(model_path(label)),model_revision=m['revision'],model_files=files,
            checkpoint_identity=digest(dict(profile=PROFILE,model=m['repo'],revision=m['revision'],files=files)))

    def test_exact_four_groups_and_repeats(self):
        for kind in ('original','treegrpo'):validate_plan(self.plan(kind,0))
        for kind in ('base','searchr1'):
            for r in (1,2,3):validate_plan(self.plan(kind,r))
        for k,v in [('kind','budget5'),('repeat_index',4),('repeat_index',True),('model_revision','main'),('root','/tmp/escape')]:
            p=self.plan();p[k]=v
            with self.assertRaises(ValueError):validate_plan(p)

    def test_durable_intent_cannot_overwrite(self):
        from .submit import write_once
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'intent.json';write_once(f,{'job':'first'})
            with self.assertRaises(FileExistsError):write_once(f,{'job':'second'})
            self.assertEqual(json.loads(f.read_text()),{'job':'first'})

    def test_eval_changes_only_memory_batch_and_loopback(self):
        c=self.config();e=make_config(ROOT/'runs/original',model_path('base'),'gpu-ada6000-1',evaluation=True)
        e.data.val_batch_size=c.data.val_batch_size
        e.actor_rollout_ref.rollout.gpu_memory_utilization=c.actor_rollout_ref.rollout.gpu_memory_utilization
        e.retriever.url=c.retriever.url
        self.assertEqual(c,e)


if __name__=='__main__':unittest.main()
