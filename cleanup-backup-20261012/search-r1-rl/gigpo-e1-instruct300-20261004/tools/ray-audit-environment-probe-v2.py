"""Real Ray2.40 nested env merge, CPU actors only; no policy/GPU computation."""
import json,os,sys,subprocess,sysconfig
from pathlib import Path
import ray
from search_r1.baselines.runtime_audit import runtime_env_vars
expected=runtime_env_vars()
assert len(expected)==5,expected
@ray.remote(num_cpus=1,num_gpus=0)
class Probe:
    def inspect(self):
        expected=os.environ['SEARCH_R1_PYTHON_HEADER_CPATH']
        assert os.environ.get('CPATH')==expected
        compiler=subprocess.run(['gcc','-x','c','-fsyntax-only','-'],input='#include <Python.h>\n#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 10 || PY_MICRO_VERSION != 12\n#error Wrong Python headers\n#endif\nint main(void) {return 0;}\n',text=True,capture_output=True)
        assert compiler.returncode==0,compiler.stderr
        return dict(environment=runtime_env_vars(),rank=os.environ['RANK'],pid=os.getpid(),cpath=expected,python=sys.version,sysconfig_include=sysconfig.get_path('include'),compiler_header_probe='PASS')
@ray.remote(num_cpus=0)
def driver():
    actors=[Probe.options(runtime_env={'env_vars':{'RANK':str(rank),'WG_BACKEND':'ray'}}).remote() for rank in range(8)]
    try:
        return dict(driver=runtime_env_vars(),workers=ray.get([x.inspect.remote() for x in actors]))
    finally:
        for x in actors:ray.kill(x,no_restart=True)  # Only handles created by this probe.
ray.init(address=os.environ['RAY_ADDRESS'])
try:
    result=ray.get(driver.options(runtime_env={'env_vars':expected}).remote())
    assert result['driver']==expected
    assert len({x['pid'] for x in result['workers']})==8
    assert all(x['environment']==expected and x['rank']==str(rank) for rank,x in enumerate(result['workers']))
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
    print('REAL_RAY_DRIVER_AND_EIGHT_CPU_ACTOR_BASELINE_IDENTITY_MERGE_PASS')
finally:ray.shutdown()
