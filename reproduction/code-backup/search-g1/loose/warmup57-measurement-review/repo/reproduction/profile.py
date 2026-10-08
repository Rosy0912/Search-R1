"""Explicit profile routing; absent profile preserves historical A800 behavior."""
import importlib
import os


def module(kind):
    profile = os.environ.get('SEARCH_G1_PROFILE', 'a800')
    if profile not in ('a800', 'common8'):
        raise ValueError('Unknown G1 runtime profile: ' + profile)
    return importlib.import_module('reproduction.' + profile + '.' + kind)


def bind_validation_dataframe(*args, **kwargs):
    return module('validation_identity').bind_validation_dataframe(*args, **kwargs)


def record_validation(*args, **kwargs):
    return module('validation_identity').record_validation(*args, **kwargs)


def checkpoint_complete(*args, **kwargs):
    return module('runtime').checkpoint_complete(*args, **kwargs)


def accept_five_updates(*args, **kwargs):
    return module('runtime').accept_five_updates(*args, **kwargs)


def refresh_after_checkpoint(*args, **kwargs):
    return module('runtime').refresh_after_checkpoint(*args, **kwargs)
