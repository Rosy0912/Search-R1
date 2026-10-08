"""Tiny CPU probe of exact source functions; no FSDP/model/asset execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def source_function(source_root, relative, name):
    path = Path(source_root) / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], ast.dump(node, include_attributes=False)


def fresh():
    params = [torch.nn.Parameter(torch.tensor([1., -2.])),
              torch.nn.Parameter(torch.tensor([3., 4.]))]
    optimizer = torch.optim.AdamW(params, lr=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    return params, optimizer, scheduler


def snapshot(optimizer):
    return {param: {key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()}
            for param, state in optimizer.state.items()}


def functions(root):
    return [source_function(root, "verl/utils/fsdp_utils.py", name)[0]
            for name in ("offload_fsdp_optimizer", "load_fsdp_optimizer")]


def invoke(function, optimizer):
    with patch.object(torch.cuda, "empty_cache"):
        if function.__name__ == "load_fsdp_optimizer":
            function(optimizer, torch.device("cpu"))
        else:
            function(optimizer)


class TinyCPU(unittest.TestCase):
    source_root = None

    def assert_unchanged(self, optimizer, before):
        self.assertEqual(set(optimizer.state), set(before))
        for param, state in before.items():
            self.assertEqual(set(optimizer.state[param]), set(state))
            for key, value in state.items():
                actual = optimizer.state[param][key]
                if isinstance(value, torch.Tensor):
                    self.assertEqual(actual.device, torch.device("cpu"))
                    self.assertEqual(actual.dtype, value.dtype)
                    self.assertTrue(torch.equal(actual, value))
                else:
                    self.assertEqual(actual, value)

    def test_fresh_offload_and_load_each_preserve_entire_empty_state_zero_steps(self):
        for function in functions(self.source_root):
            with self.subTest(function=function.__name__):
                params, optimizer, scheduler = fresh()
                before_params = [p.detach().clone() for p in params]
                before_lrs = scheduler.get_last_lr()
                self.assertEqual(len(optimizer.state), 0)
                self.assertEqual(scheduler.last_epoch, 0)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    invoke(function, optimizer)
                    self.assertEqual(step.call_count, 0)
                self.assertEqual(len(optimizer.state), 0)
                self.assertEqual(scheduler.last_epoch, 0)
                self.assertEqual(scheduler.get_last_lr(), before_lrs)
                for param, before in zip(params, before_params):
                    self.assertTrue(torch.equal(param.detach(), before))

    def test_one_real_adam_write_preserves_moments_and_skips_unused_parameter(self):
        params, optimizer, scheduler = fresh()
        params[0].grad = torch.tensor([.25, -.5])
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            optimizer.step()
            self.assertEqual(step.call_count, 1)
            self.assertEqual(set(optimizer.state), {params[0]})
            self.assertEqual(float(optimizer.state[params[0]]["step"]), 1.)
            self.assertEqual(set(optimizer.state[params[0]]), {"step", "exp_avg", "exp_avg_sq"})
            # Existing non-tensor entries use the unchanged migration branch.
            marker = object()
            optimizer.state[params[0]]["cpu_probe_marker"] = marker
            before = snapshot(optimizer)
            before_params = [p.detach().clone() for p in params]
            state_object = optimizer.state[params[0]]
            for function in functions(self.source_root):
                invoke(function, optimizer)
                self.assert_unchanged(optimizer, before)
                self.assertIs(optimizer.state[params[0]], state_object)
                self.assertIs(optimizer.state[params[0]]["cpu_probe_marker"], marker)
                self.assertNotIn(params[1], optimizer.state)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(scheduler.last_epoch, 0)
            for param, before_param in zip(params, before_params):
                self.assertTrue(torch.equal(param.detach(), before_param))

    def test_existing_empty_mapping_is_retained_not_cleared_or_fabricated_fresh(self):
        for function in functions(self.source_root):
            with self.subTest(function=function.__name__):
                params, optimizer, _ = fresh()
                original = {}
                optimizer.state[params[0]] = original
                invoke(function, optimizer)
                self.assertEqual(set(optimizer.state), {params[0]})
                self.assertIs(optimizer.state[params[0]], original)
                self.assertEqual(len(optimizer.state), 1)
                self.assertNotIn(params[1], optimizer.state)

    def test_existing_none_or_nonmapping_still_raises_and_remains_present(self):
        for function in functions(self.source_root):
            for invalid in (None, 7):
                with self.subTest(function=function.__name__, invalid=invalid):
                    params, optimizer, _ = fresh()
                    optimizer.state[params[0]] = invalid
                    with self.assertRaises(AttributeError):
                        invoke(function, optimizer)
                    self.assertEqual(set(optimizer.state), {params[0]})
                    self.assertIs(optimizer.state[params[0]], invalid)

    def test_actual_worker_inspector_preserves_whole_state_and_initial_scheduler_gate(self):
        inspector, _ = source_function(self.source_root, "verl/workers/fsdp_workers.py",
                                       "inspect_fresh_worker_state")
        sampling = ModuleType("search_r1.baselines.sampling_consistency")
        sampling.collective_device = lambda: torch.device("cpu")
        def sync_prepare(error, device):
            if error is not None:
                raise RuntimeError(error)
        sampling.sync_prepare = sync_prepare
        sampling.check_signature = lambda values, device: None
        protocol = ModuleType("search_r1.baselines.tree_fresh200")
        protocol.refuse_recovery_environment = lambda: None
        env = {"SEARCH_R1_BASELINE_COMMIT": "tiny-cpu-fixture",
               "SEARCH_R1_BASELINE_CONFIG_SHA256": "tiny-cpu-fixture",
               "SEARCH_R1_BASELINE_DATA_IDENTITY": "tiny-cpu-fixture",
               "SEARCH_R1_MODEL_IDENTITY": "tiny-cpu-fixture",
               "SLURM_JOB_ID": "tiny-cpu-fixture", "SEARCH_R1_BASELINE_RUN_ID": "tiny-cpu-fixture"}
        # Construct real optimizers before patch.dict(sys.modules), whose restore
        # would otherwise remove lazily imported torch modules after registration.
        initial_fixture, scheduler_fixture = fresh(), fresh()
        # Only import dependencies and rank collectives are fixture substitutions.
        # The gate body is compiled directly from this candidate's actual source.
        with patch.dict(sys.modules, {sampling.__name__: sampling, protocol.__name__: protocol}), \
                patch.dict(os.environ, env), \
                patch.object(torch.distributed, "get_rank", return_value=0), \
                patch.object(torch.distributed, "get_world_size", return_value=1):
            params, optimizer, scheduler = initial_fixture
            worker = SimpleNamespace(_is_actor=True, actor_optimizer=optimizer,
                actor_lr_scheduler=scheduler, config=SimpleNamespace(
                    actor={"fresh200_protocol": "tree_fresh200_five_leaf_v1"},
                    model=SimpleNamespace(path="/tiny-cpu-fixture/model")))
            record = inspector(worker, "/tiny-cpu-fixture/model")
            self.assertEqual(record["optimizer_state_entries"], 0)
            self.assertEqual(record["scheduler_step"], 0)
            for invalid in ({}, None, {"step": torch.tensor(0.)}):
                optimizer.state[params[0]] = invalid
                with self.assertRaisesRegex(RuntimeError, "entire state mapping to be empty"):
                    inspector(worker, "/tiny-cpu-fixture/model")
                self.assertIs(optimizer.state[params[0]], invalid)
            params, optimizer, scheduler = scheduler_fixture
            worker.actor_optimizer = optimizer
            worker.actor_lr_scheduler = scheduler
            scheduler.last_epoch = 1
            with self.assertRaisesRegex(RuntimeError, "initial step zero"):
                inspector(worker, "/tiny-cpu-fixture/model")


def base_counterexample(root):
    records = []
    for function in functions(root):
        params, optimizer, scheduler = fresh()
        before = len(optimizer.state)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            invoke(function, optimizer)
            records.append(dict(function=function.__name__, state_entries_before=before,
                state_entries_after=len(optimizer.state),
                every_created_entry_empty=all(state == {} for state in optimizer.state.values()),
                optimizer_step_calls=step.call_count, scheduler_last_epoch=scheduler.last_epoch,
                params_count=len(params)))
    assert all(r["state_entries_before"] == 0 and r["state_entries_after"] == 2
               and r["every_created_entry_empty"] and r["optimizer_step_calls"] == 0
               and r["scheduler_last_epoch"] == 0 for r in records)
    return dict(status="BASE_SOURCE_ZERO_STEP_STATE_CREATION_REPRODUCED", records=records,
        python_executable=sys.executable, torch_version=torch.__version__,
        source_path=str(Path(root).resolve()), actual_gpu_executed=False,
        actual_remote_executed=False, official_assets_loaded=False,
        source_fsdp_utils_sha256=hashlib.sha256((Path(root)/"verl/utils/fsdp_utils.py").read_bytes()).hexdigest(),
        execution_scope="Exact AST-extracted source functions; real torch AdamW; CPU tensors; empty_cache stub only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--base-counterexample", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.base_counterexample:
        result = base_counterexample(args.source)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.json_output:
            Path(args.json_output).write_text(encoded)
        print(encoded, end="")
    else:
        TinyCPU.source_root = str(Path(args.source).resolve())
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TinyCPU)
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if args.json_output:
            Path(args.json_output).write_text(json.dumps(dict(
                status="TINY_CPU_PASSED" if result.wasSuccessful() else "TINY_CPU_FAILED",
                actual_test_count=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                exit_code=0 if result.wasSuccessful() else 1,
                wall_seconds=time.perf_counter()-started_wall,
                process_cpu_seconds=time.process_time()-started_cpu,
                python_executable=sys.executable, torch_version=torch.__version__,
                source_path=TinyCPU.source_root,
                source_files={relative: hashlib.sha256((Path(TinyCPU.source_root)/relative).read_bytes()).hexdigest()
                    for relative in ("verl/utils/fsdp_utils.py", "verl/workers/fsdp_workers.py")},
                actual_gpu_executed=False, actual_remote_executed=False, actual_slurm_executed=False,
                official_assets_loaded=False, actual_3b_optimizer_calls=0, actual_3b_outer_updates=0,
                actual_tiny_cpu_adamw_step_calls=1 if result.wasSuccessful() else None,
                cpu_device_scope="CPU->CPU Tensor.to only; no CUDA, accelerator, FSDP or distributed device migration",
                execution_scope="5 exact-source AST function tests with real torch AdamW; worker import/collective fixtures and empty_cache stub"),
                sort_keys=True, indent=2)+"\n")
        sys.exit(0 if result.wasSuccessful() else 1)
