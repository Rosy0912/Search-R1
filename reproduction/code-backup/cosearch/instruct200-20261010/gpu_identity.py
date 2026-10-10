"""Resolve only owned four-GPU UUIDs for vLLM without changing visibility."""
import functools,hashlib,json,os
from pathlib import Path

def install(platform_module=None):
    if platform_module is None:
        import vllm.platforms.interface as platform_module
    if hashlib.sha256(Path(platform_module.__file__).read_bytes()).hexdigest()!='ff142312173c173dae20a8303cd10f9c29fa255862396655b6888d2fdb73fae5':
        raise RuntimeError('Unreviewed vLLM platform source')
    cls=platform_module.Platform;original=cls.device_id_to_physical_device_id.__func__
    if getattr(original,'_cosearch_uuid_mapping',False):return
    @functools.wraps(original)
    def resolve(platform,device_id):
        mask=os.environ.get(platform.device_control_env_var,'');values=mask.split(',') if mask else []
        if values and values[device_id].startswith('GPU-'):
            mapping=json.loads(os.environ['COSEARCH_LEASED_UUID_TO_INDEX'])
            leased=set(os.environ['COSEARCH_LEASED_UUIDS'].split(','))
            if len(mapping)!=4 or set(mapping)!=leased or len(set(mapping.values()))!=4:raise RuntimeError('Invalid owned four-GPU mapping')
            if not all(isinstance(v,int) and v>=0 for v in mapping.values()):raise RuntimeError('Invalid GPU indices')
            if not all(v in leased for v in values):raise RuntimeError('Foreign visible GPU UUID')
            return mapping[values[device_id]]
        return original(platform,device_id)
    resolve._cosearch_uuid_mapping=True;cls.device_id_to_physical_device_id=classmethod(resolve)
