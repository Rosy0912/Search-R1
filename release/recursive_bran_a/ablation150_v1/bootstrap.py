"""Process-local hooks, applied independently in the entry and every Ray worker.

The published original modules remain untouched. Imports are intercepted lazily,
so installing this loader does not create Torch/CUDA state before Ray starts.
"""
import copy
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

from .contracts import projected_original, read_receipt, validate_config_dict

_INSTALLED = False
PREFIX = 'RECURSIVE_ABLATION_'


def active():
    return bool(os.environ.get(PREFIX + 'RECEIPT'))


def receipt():
    return read_receipt(os.environ[PREFIX + 'RECEIPT'])


def _native_contract(module):
    original = module.validate_recursive_config

    def validate(config):
        if not active():
            return original(config)
        from omegaconf import OmegaConf
        actual = OmegaConf.to_container(config, resolve=True)
        validate_config_dict(actual)
        return original(OmegaConf.create(projected_original(actual)))
    module.validate_recursive_config = validate


def _runtime_environment(module):
    original = module.runtime_env_vars

    def runtime_env_vars():
        result = original()
        if active():
            result.update({key: value for key, value in os.environ.items() if key.startswith(PREFIX)})
            if 'PYTHONPATH' in os.environ:
                result['PYTHONPATH'] = os.environ['PYTHONPATH']
        return result
    module.runtime_env_vars = runtime_env_vars


def _rollout(module):
    if active():
        from search_r1.baselines.recursive_ablation import install
        condition = receipt()['condition']
        install({key: condition[key] for key in ('variant', 'position', 'max_generation', 'selection_seed')})


def _native_resume(module):
    original_enabled, original_validate = module.enabled, module.validate_entry
    original_load, original_migrate = module.load_worker, module.migrate_trainer

    def enabled(environ=None):
        values = os.environ if environ is None else environ
        return bool(values.get(PREFIX + 'RESUME_MANIFEST')) or original_enabled(environ)

    def validate(config=None, environ=None):
        values = os.environ if environ is None else environ
        if not values.get(PREFIX + 'RESUME_MANIFEST'):
            return original_validate(config, environ)
        from .native_resume import manifest
        manifest()
        if config is not None:
            from omegaconf import OmegaConf
            validate_config_dict(OmegaConf.to_container(config, resolve=True))
        return True

    def load(worker):
        if not os.environ.get(PREFIX + 'RESUME_MANIFEST'):
            return original_load(worker)
        from .native_resume import load_worker
        return load_worker(worker)

    def migrate(trainer):
        if not os.environ.get(PREFIX + 'RESUME_MANIFEST'):
            return original_migrate(trainer)
        from .native_resume import migrate_trainer
        return migrate_trainer(trainer)

    module.enabled, module.validate_entry = enabled, validate
    module.load_worker, module.migrate_trainer = load, migrate


class _TrainingLogger:
    def __init__(self, trainer, original):
        self.trainer, self.original = trainer, original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def log(self, data, step, **kwargs):
        result = self.original.log(data=data, step=step, **kwargs)
        trainer = self.trainer
        if data.get('training/actual_outer_updates') == step and trainer.actual_outer_updates == step:
            from .progress import completed_round
            completed_round(trainer, data, step)
        return result


def _trainer(module):
    cls = module.RayPPOTrainer
    create, fit, validate = cls._create_dataloader, cls.fit, cls._validate

    def create_dataloader(self):
        result = create(self)
        if active():
            from omegaconf import open_dict
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = 200
                self.config.critic.optim.total_training_steps = 201
            validate_config_dict(__import__('omegaconf').OmegaConf.to_container(self.config, resolve=True))
        return result

    def validate_once(self, *args, **kwargs):
        if active() and self.global_steps == 151:
            # Native fit asks for a final evaluation after incrementing to151.
            # The actual raw150 scheduled dev512 already ran and was persisted.
            if not hasattr(self, '_ablation_dev150'):
                raise ValueError('Missing actual scheduled150 result; cannot manufacture terminal evaluation')
            print('ABLATION_TERMINAL_REUSES_ACTUAL_SCHEDULED_DEV150', flush=True)
            return copy.deepcopy(self._ablation_dev150)
        result = validate(self, *args, **kwargs)
        if active() and self.global_steps == 150:
            self._ablation_dev150 = copy.deepcopy(result)
        return result

    def prepare_final(self):
        if not active():
            return original_prepare_final(self)
        # No unrequested final51713 scan for every mechanism screening group.
        return None

    def fit_with_rolling(self):
        if not active():
            return fit(self)
        self.logger = _TrainingLogger(self, self.logger)
        try:
            result = fit(self)
            if self.actual_outer_updates != 150:
                raise ValueError('Native trainer returned before150 complete sampling rounds')
            from .progress import finished
            finished(self)
            return result
        finally:
            self.logger = self.logger.original

    original_prepare_final = cls._prepare_final_validation
    cls._create_dataloader, cls.fit, cls._validate = create_dataloader, fit_with_rolling, validate_once
    cls._prepare_final_validation = prepare_final


CALLBACKS = {
    'search_r1.baselines.tree_fresh200': _native_contract,
    'search_r1.baselines.runtime_audit': _runtime_environment,
    'search_r1.baselines.tree_current_chain_resume': _native_resume,
    'search_r1.baselines.rollout': _rollout,
    'verl.trainer.ppo.ray_trainer': _trainer,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, original, callback):
        self.original, self.callback = original, callback

    def create_module(self, spec):
        method = getattr(self.original, 'create_module', None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.original.exec_module(module)
        self.callback(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in CALLBACKS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader, CALLBACKS[fullname])
        return spec


def install():
    global _INSTALLED
    if not active() or _INSTALLED:
        return
    receipt()  # Bind process startup to the frozen configuration before imports.
    _INSTALLED = True
    sys.meta_path.insert(0, _Finder())
    for name, callback in CALLBACKS.items():
        if name in sys.modules:
            callback(sys.modules[name])


def worker_setup():
    install()

