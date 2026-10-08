"""Published node-side CPU fixtures and frozen source/dependency checks; no job launch."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import unittest

try:
    from .checkpoint import checked_json, checked_file, require
    from .prepare import verify_runtime
except ImportError:
    from checkpoint import checked_json, checked_file, require
    from prepare import verify_runtime


def preflight(root, helper_ref, runtime, runtime_ref):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['OPSD_ONLINE_TEST_RUNTIME']=str(Path(runtime).resolve())
    sys.dont_write_bytecode=True
    root=Path(root).resolve()
    manifest, proof=checked_json(helper_ref)
    def verify_helpers():
        for name, entry in manifest['files'].items():
            path=Path(name)
            require(not path.is_absolute() and '..' not in path.parts,'Unsafe helper source path')
            checked_file(dict(path=str(root/path),sha256=entry['sha256']))
        require({str(p.relative_to(root)) for p in root.rglob('*.py')} ==
                {name for name in manifest['files'] if name.endswith('.py')}, 'Helper Python inventory changed')
    verify_helpers()
    import torch
    require(not torch.cuda.is_initialized(),'CPU fixtures must not initialize CUDA')
    versions={name:importlib.metadata.version(name) for name in
              ('torch','transformers','peft','vllm','xformers','numpy','rapidfuzz')}
    require((versions['torch'].split('+')[0],versions['transformers'],versions['peft'])==
            ('2.4.0','4.46.3','0.13.2'),'Node dependency versions differ from frozen native training')
    frozen=verify_runtime(dict(paths={'runtime':str(runtime)},runtime_manifest=runtime_ref))
    sys.path.insert(0,str(root))
    suite=unittest.TestLoader().discover(str(root),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    verify_helpers()
    require(not torch.cuda.is_initialized(),'A CPU fixture unexpectedly initialized CUDA')
    require(result.wasSuccessful() and not result.skipped,'Node CPU fixtures failed or skipped')
    return dict(schema='opsd-online-node-cpu-preflight-v1',accepted=True,
                scope='CPU source/math/restore fixtures only; no GPU launch or five-update acceptance',
                helper_manifest=proof, frozen_runtime=frozen,python_version=sys.version,
                dependency_versions=versions,tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skipped=len(result.skipped),cuda_initialized=False,
                production_training_started=False,launch_allowed=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--helper-root',required=True)
    for name in ('helper-manifest','runtime-manifest'):
        parser.add_argument('--'+name,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--runtime',required=True)
    args=parser.parse_args()
    result=preflight(args.helper_root,dict(path=args.helper_manifest,sha256=args.helper_manifest_sha256),
        args.runtime,dict(path=args.runtime_manifest,sha256=args.runtime_manifest_sha256))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
