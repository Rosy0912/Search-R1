import ast
import copy
from pathlib import Path
import subprocess
import unittest
import os
from . import test_contract
from .contract import ROOT,make_config,model_path
from .original_memory_recovery import JOB,RUN,ALLOCATION,OLD_SOURCE,validate


class MemoryRecovery(unittest.TestCase):
    def plans(self):
        from omegaconf import OmegaConf
        old=test_contract.Contracts().plan('original',0)
        old['config_text']=OmegaConf.to_yaml(make_config(ROOT/'runs/original',model_path('base'),'gpu-ada6000-1'),resolve=True)
        for k,v in dict(data_root='frozen',data_manifest_sha256='frozen',ordered_ids_sha256='frozen',final_content_identity='frozen',retriever_receipt='frozen',retriever_receipt_sha256='frozen',retriever='frozen',repeat_count=3,outer=200).items():old[k]=v
        new=copy.deepcopy(old);c=OmegaConf.create(old['config_text']);c.trainer.default_local_dir=str(RUN/'checkpoints')
        new.update(protocol='sevenb_original_host_memory_replacement_v1',job=JOB,root=str(RUN),output=str(RUN/'output'),allocation=copy.deepcopy(ALLOCATION),old_job='139101',old_source=OLD_SOURCE,fresh_from_common_initialization=True,resume_from_partial_state=False,config_text=OmegaConf.to_yaml(c,resolve=True))
        return new,old

    def test_memory_only_replacement(self):
        n,o=self.plans();self.assertEqual(validate(n,o),RUN)

    def test_reject_scientific_change(self):
        from omegaconf import OmegaConf
        for field,value in [('data.train_batch_size',256),('actor_rollout_ref.actor.optim.lr',2e-6),('trainer.actual_update_limit',5)]:
            n,o=self.plans();c=OmegaConf.create(n['config_text']);OmegaConf.update(c,field,value);n['config_text']=OmegaConf.to_yaml(c,resolve=True)
            with self.assertRaises(AssertionError):validate(n,o)

    def test_partial_state_cannot_be_resume(self):
        n,o=self.plans();n['resume_from_partial_state']=True
        with self.assertRaises(AssertionError):validate(n,o)

    def test_fixed_namespace_resources_identity(self):
        for k,v in [('job','wrong'),('root','/tmp/escape'),('model_revision','main'),('old_job','139102'),('allocation',dict(gpus=8,cpus=64,memory_gib=600,time='7-00:00:00'))]:
            n,o=self.plans();n[k]=v
            with self.assertRaises(AssertionError):validate(n,o)

    def test_extracted_original_runner_ast_identical(self):
        frozen=os.environ.get('SEVENB_ORIGINAL_FROZEN_ENTRY')
        old=ast.parse(Path(frozen).read_text() if frozen else subprocess.check_output(['git','show',OLD_SOURCE+':reproduction/sevenb/entry.py'],text=True))
        run=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        start=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets))
        new=ast.parse(Path(__file__).with_name('entry.py').read_text())
        extracted=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='run_original')
        self.assertIsInstance(extracted.body[0],ast.ImportFrom)
        self.assertEqual(extracted.body[0].module,'omegaconf')
        self.assertEqual(ast.dump(ast.Module(body=run.body[start:],type_ignores=[])),ast.dump(ast.Module(body=extracted.body[1:],type_ignores=[])))


if __name__=='__main__':unittest.main()
