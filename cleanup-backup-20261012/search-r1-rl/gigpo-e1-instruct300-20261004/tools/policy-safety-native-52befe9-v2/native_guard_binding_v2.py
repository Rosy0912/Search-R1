"""Require the isolated native guard and module to be in the SHA binding."""
from pathlib import Path


def validate_guard_helper_binding(helpers, value, require, sha):
    helpers = Path(helpers).resolve()
    require(Path(value['helpers']).resolve() == helpers, 'Native helper directory differs from binding')
    # The original full file-SHA validation remains in the caller. Additionally,
    # reject an old binding that omits this versioned executable/module/tests.
    names = ('native_run_safety_v1.py', 'native_guard_binding_v2.py',
             'resource_guard_v2.py', 'resource_ownership_v2.py',
             'test_resource_ownership_v2.py', 'test_native_guard_binding_v2.py')
    for name in names:
        path = helpers / name
        entries = [item for item in value['files'] if Path(item['path']).resolve() == path]
        require(len(entries) == 1, 'Versioned native guard/helper missing or duplicate binding: ' + name)
        require(sha(path) == entries[0]['sha256'], 'Versioned native guard/helper SHA mismatch: ' + name)
    identity = {}
    for key, name in (('resource_guard', 'resource_guard_v2.py'),
                      ('resource_ownership', 'resource_ownership_v2.py')):
        require(key in value, 'Explicit versioned native guard identity missing: ' + key)
        item = value[key]
        path = helpers / name
        require(Path(item['path']).resolve() == path, 'Explicit versioned native guard path mismatch: ' + key)
        actual_sha = sha(path)
        require(item['sha256'] == actual_sha, 'Explicit versioned native guard SHA mismatch: ' + key)
        identity[key] = dict(path=str(path), sha256=actual_sha)
    return identity
