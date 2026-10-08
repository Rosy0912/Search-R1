"""Replay reviewed V2 CPU tests against only-source-rebound runtime files."""
import importlib.util,sys
from pathlib import Path
base=Path(__file__).resolve().parent.parent
for name in ('receipt_payload','hf_cpu_reload'):
 p=base/'rebound-a0968c2-candidate/first5'/(name+'.py');spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sys.modules[name]=m
original=base/'first5-tools-service-marker-live-state-v2/test_first5_tools.py'
# __file__ stays at original fixture directory; production imports above point
# to pinned rebound files. No deployed tool/source is changed.
namespace={'__file__':str(original),'__name__':__name__}
exec(compile(original.read_bytes(),str(original),'exec'),namespace)
for name,value in namespace.items():
 if name.startswith('test_') or name in ('r','hf','manifest','actual_metrics','frozen_gate','log_fixture','tiny_hf'):globals()[name]=value
