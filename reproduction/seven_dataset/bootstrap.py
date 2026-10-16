"""Install frozen hooks lazily, after Ray has assigned the actor's GPU mask."""
import importlib
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = 'verl.workers.fsdp_workers'
_INSTALLED = False


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        method = getattr(self.original, 'create_module', None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.original.exec_module(module)
        importlib.import_module('reproduction.seven_dataset.worker')

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    global _INSTALLED
    if os.environ.get('TREE_SEVEN_EVAL_MODE') != 'frozen_milestone_v1':
        raise RuntimeError('Lazy frozen hooks require the dedicated evaluator')
    if _INSTALLED:
        return
    if _TARGET in sys.modules:
        raise RuntimeError('FSDP module was imported before lazy worker setup')
    sys.meta_path.insert(0, _Finder())
    _INSTALLED = True
