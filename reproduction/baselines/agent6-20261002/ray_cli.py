"""Ray 2.40 CLI with immutable Click 8.5 sentinels on Python 3.10.

The pinned environment stays untouched. This affects only the CLI process;
Ray worker environments and all training dependencies retain their versions.
"""
import copy
from click import _utils


def preserve_click_sentinel_identity():
    sentinel = getattr(_utils, 'Sentinel', None)
    if sentinel is None:
        return False
    try:
        copy.deepcopy(sentinel.UNSET)
    except ValueError:
        # Python 3.10 Enum reconstruction deep-copies the object-valued member,
        # producing a value not present in the enum. Sentinels are singletons.
        sentinel.__deepcopy__ = lambda self, memo: self
    assert all(copy.deepcopy(item) is item for item in sentinel)
    return True


if __name__ == '__main__':
    preserve_click_sentinel_identity()
    from ray.scripts.scripts import main
    main()
