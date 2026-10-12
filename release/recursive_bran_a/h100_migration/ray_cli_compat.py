"""Process-local Click sentinel identity fix for Ray CLI; dependencies unchanged."""
import copy
import importlib
import runpy


def preserve_sentinel(value, memo):
    memo[id(value)] = value
    return value


def prepare_click():
    # Older Click lacks this sentinel and its deepcopy regression entirely.
    try:
        utility = importlib.import_module('click._utils')
    except ModuleNotFoundError as error:
        if error.name != 'click._utils': raise
        return False
    sentinel = getattr(utility, 'Sentinel', None)
    if sentinel is None: return False
    if not all(copy.copy(value) is value for value in sentinel):
        raise RuntimeError('Unexpected Click sentinel copy behavior')
    copy._deepcopy_dispatch[sentinel] = preserve_sentinel
    if not all(copy.deepcopy(value) is value for value in sentinel):
        raise RuntimeError('Click sentinel identity repair failed')
    return True


if __name__ == '__main__':
    prepare_click()
    runpy.run_module('ray.scripts.scripts', run_name='__main__')
