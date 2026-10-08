"""Subprocess UUID-query compatibility. Startup installs only a standard-library import hook."""
import importlib.abc,importlib.machinery,sys

class _VllmUuidLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        import observe_worker
        observe_worker.install_vllm_uuid_mapping(module)
        import os
        if os.environ.get('COSEARCH_NATIVE_EVIDENCE_ROOT'):
            observe_worker.write_record('vllm_uuid_compat_import',dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),subprocess_compatible=True))

class _VllmUuidFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!='vllm.platforms.interface':return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.loader is not None:
            spec.loader=_VllmUuidLoader(spec.loader)
        return spec

if not any(isinstance(x,_VllmUuidFinder) for x in sys.meta_path):
    sys.meta_path.insert(0,_VllmUuidFinder())
