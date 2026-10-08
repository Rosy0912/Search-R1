"""Keep Click sentinel identity during Ray 2.40 CLI command alias deepcopy.

This shim is confined to this launcher process. Dependency files are unchanged.
"""
import copy
import runpy
from click._utils import Sentinel


def preserve_sentinel(value, memo):
    memo[id(value)] = value
    return value


if __name__ == '__main__':
    assert all(copy.copy(value) is value for value in Sentinel)
    copy._deepcopy_dispatch[Sentinel] = preserve_sentinel
    assert all(copy.deepcopy(value) is value for value in Sentinel)
    runpy.run_module('ray.scripts.scripts', run_name='__main__')
