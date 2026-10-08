"""Execute actual Ray bootstrap and trainer control-flow AST on CPU doubles."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf
from verl.utils.weight_continuation import model_initialization_path
from test_final_checkpoint import real_iteration_tail, FakeTrainer

ROOT=Path(__file__).resolve().parents[2]

class ContinuationWiring(unittest.TestCase):
    def test_actual_worker_path_expressions_select_actor_and_base(self):
        tree=ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model_initialization_path']
        self.assertEqual(len(calls),2)
        roles=set()
        config=OmegaConf.create({'model':{'path':'/base','actor_checkpoint_path':'/checkpoint'}})
        for node in calls:
            roles.add(ast.literal_eval(next(k.value for k in node.keywords if k.arg=='actor')))
            code=compile(ast.Expression(node),'worker-callsite','eval')
            result=eval(code,{'model_initialization_path':model_initialization_path,'self':SimpleNamespace(config=config)})
            role=ast.literal_eval(next(k.value for k in node.keywords if k.arg=='actor'))
            self.assertEqual(result,'/checkpoint' if role else '/base')
        self.assertEqual(roles,{False,True})

    def test_actual_private_ray_bootstrap_and_cleanup(self):
        tree=ast.parse((ROOT/'verl/trainer/main_ppo.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        main.decorator_list=[]
        events=[]
        ray=SimpleNamespace(is_initialized=lambda:False,init=lambda **kw:events.append(kw),
                            get=lambda x:events.append(x),shutdown=lambda:events.append('shutdown'))
        ns={'ray':ray,'main_task':SimpleNamespace(remote=lambda c:'own_task')}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),'real-main','exec'),ns)
        cfg=OmegaConf.create({'trainer':{'private_local_ray':True,'n_gpus_per_node':4}})
        with patch.dict(os.environ,{'RAY_TMPDIR':'/data/search-r1/ray-a800','RAY_ADDRESS':'foreign:6379'}):ns['main'](cfg)
        self.assertEqual(events[0]['address'],'local')
        self.assertEqual(events[0]['num_gpus'],4)
        self.assertEqual(events[0]['_temp_dir'],'/data/search-r1/ray-a800')
        self.assertEqual(events[-1],'shutdown')
        ray.is_initialized=lambda:True
        with self.assertRaises(RuntimeError):ns['main'](cfg)

    def test_actual_fit_counter_initialization_is_101(self):
        tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
        fit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        end=next(i for i,n in enumerate(fit.body) if isinstance(n,ast.AugAssign) and ast.unparse(n.target)=='self.global_steps')
        body=fit.body[:end+1]
        config=OmegaConf.create({'trainer':{'weight_continuation_step':100,'acceptance_step':105},'actor_rollout_ref':{'model':{'path':'/base','actor_checkpoint_path':'/checkpoint'}}})
        trainer=SimpleNamespace(config=config,total_training_steps=501,logger=None,val_reward_fn=None)
        function=ast.FunctionDef(name='initialize',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
        ns={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'real-fit-initial','exec'),ns)
        ns['initialize'](trainer)
        self.assertEqual(trainer.global_steps,101)

    def test_actual_five_updates_save105_and_continue_same_counter(self):
        trainer=FakeTrainer(101,501);trainer.config.trainer.acceptance_step=105
        run=real_iteration_tail()
        for _ in range(6):run(trainer)
        self.assertEqual([e for e in trainer.events if e[0]=='save'],[('save',105,5)])
        self.assertEqual(trainer.global_steps,107)
        self.assertFalse(any(e[0]=='final_data' for e in trainer.events))

    def test_actual_continuation_terminal_saves500(self):
        trainer=FakeTrainer(499,501);trainer.config.trainer.acceptance_step=105
        with redirect_stdout(io.StringIO()):
            for _ in range(2):real_iteration_tail()(trainer)
        self.assertEqual([e for e in trainer.events if e[0]=='save'],[('save',500,2)])
        self.assertEqual(trainer.events[-1],('validate',501,2))
