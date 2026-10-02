"""CPU regression for the actual trainer's update/save/final-validation control flow.

Compile the terminal section directly from fit(), avoiding Ray/CUDA imports.
Only RPC workers and metric/timer dependencies are replaced by CPU doubles.
This checks checkpoint timing and model identity, not PPO or GPU numerics.
"""
import ast
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/ray_trainer.py"


def real_iteration_tail():
    module = ast.parse(SOURCE.read_text())
    trainer = next(n for n in module.body if isinstance(n, ast.ClassDef)
                   and n.name == "RayPPOTrainer")
    fit = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    epoch = next(n for n in fit.body if isinstance(n, ast.For))
    batches = next(n for n in epoch.body if isinstance(n, ast.For))
    step = next(n for n in batches.body if isinstance(n, ast.With))
    update = next(i for i, n in enumerate(step.body) if isinstance(n, ast.If)
                  and "critic_warmup" in ast.unparse(n.test))
    body = ast.parse("batch = object(); metrics = {}; timing_raw = {}; logger = self.logger").body
    body += step.body[update:] + batches.body[batches.body.index(step) + 1:]
    function = ast.FunctionDef(
        name="iteration_tail",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[])
    compiled = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "metrics": {}, "timing_raw": {}, "batch": object(), "json": json,
        "_timer": lambda *args: nullcontext(), "reduce_metrics": lambda x: x,
        "compute_data_metrics": lambda **kwargs: {},
        "compute_timing_metrics": lambda **kwargs: {}, "pprint": lambda x: None,
    }
    exec(compile(compiled, str(SOURCE), "exec"), namespace)
    return namespace["iteration_tail"]


class FakeTrainer:
    def __init__(self, step, total, save_freq=100, test_freq=50):
        self.global_steps = step
        self.total_training_steps = total
        self.use_critic = False
        self.config = SimpleNamespace(
            do_search=False,
            trainer=SimpleNamespace(critic_warmup=0, save_freq=save_freq, test_freq=test_freq))
        self.weight = 0
        self.events = []
        self.val_reward_fn = object()
        self.actor_rollout_wg = SimpleNamespace(update_actor=self.update)
        self.logger = SimpleNamespace(log=lambda **kwargs: None)

    def update(self, batch):
        self.weight += 1
        self.events.append(("update", self.global_steps, self.weight))
        return SimpleNamespace(meta_info={"metrics": {}})

    def _save_checkpoint(self):
        self.events.append(("save", self.global_steps, self.weight))

    def _validate(self):
        self.events.append(("validate", self.global_steps, self.weight))
        return {"em": self.weight}

    def _prepare_final_validation(self):
        self.events.append(("final_data",self.global_steps,self.weight))


class FinalCheckpointTest(unittest.TestCase):
    def test_off_boundary_final_save_matches_final_validation(self):
        run = real_iteration_tail()
        trainer = FakeTrainer(998, 1005)
        for _ in range(7):
            run(trainer)
        saves = [e for e in trainer.events if e[0] == "save"]
        self.assertEqual([e[1] for e in saves], [1000, 1004])
        self.assertEqual(saves[-1], ("save", 1004, 7))
        self.assertEqual(trainer.events[-1], ("validate", 1005, 7))
        self.assertLess(trainer.events.index(saves[-1]), len(trainer.events) - 1)

    def test_terminal_frequency_boundary_is_saved_once(self):
        trainer = FakeTrainer(1000, 1001)
        real_iteration_tail()(trainer)
        self.assertEqual([e for e in trainer.events if e[0] == "save"], [("save", 1000, 1)])
        self.assertEqual(trainer.events[-1], ("validate", 1001, 1))

    def test_nonterminal_off_boundary_remains_unsaved(self):
        trainer = FakeTrainer(999, 1005)
        real_iteration_tail()(trainer)
        self.assertEqual(trainer.events, [("update", 999, 1)])

    def test_explicit_save_disabled_remains_disabled(self):
        trainer = FakeTrainer(1004, 1005, save_freq=0)
        real_iteration_tail()(trainer)
        self.assertEqual([e for e in trainer.events if e[0] == "save"], [])
        self.assertEqual(trainer.events[-1], ("validate", 1005, 1))

    def test_step_five_saves_without_stopping_training(self):
        trainer=FakeTrainer(1,501)
        trainer.config.trainer.acceptance_step=5
        run=real_iteration_tail()
        for _ in range(6):
            run(trainer)
        self.assertEqual([e for e in trainer.events if e[0]=='save'], [('save',5,5)])
        self.assertEqual(trainer.global_steps,7)
        self.assertFalse(any(e[0]=='final_data' for e in trainer.events))


if __name__ == "__main__":
    unittest.main()
