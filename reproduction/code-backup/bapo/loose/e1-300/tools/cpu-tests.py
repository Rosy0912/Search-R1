"""Run repository tests without installed packages shadowing its namespace."""
from pathlib import Path
import sys
import os
import types
import pytest
root = Path.cwd()
# CPU tests assert default composition and use local loopback for Gloo ranks.
os.environ.pop("SEARCH_R1_ASSETS", None)
os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
for name, directory in [('tests', root/'tests'), ('tests.baselines', root/'tests/baselines')]:
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module
if __name__ == '__main__':
    raise SystemExit(pytest.main(['-q', 'tests/baselines', 'tests/reproduction']))
