import os,sys,ray,ray._private.utils as u
import observe_worker as o
assert 'torch' not in sys.modules and 'verl' not in sys.modules
original=u.set_visible_accelerator_ids
calls=[];sentinel={}
from types import SimpleNamespace
saved_context=ray.get_runtime_context
ray.get_runtime_context=lambda:SimpleNamespace(get_accelerator_ids=lambda:{'GPU':['3']})
assert not ray.is_initialized()
def assign():
 calls.append('assign');os.environ['CUDA_VISIBLE_DEVICES']='3';return sentinel
u.set_visible_accelerator_ids=assign
o.install=lambda:calls.append(('install',os.environ.get('CUDA_VISIBLE_DEVICES')))
o.write_record=lambda *args,**kwargs:None
o.defer_install()
assert calls==[] and 'torch' not in sys.modules and 'verl' not in sys.modules
assert u.set_visible_accelerator_ids() is sentinel
assert calls==['assign',('install','3')]
assert u.set_visible_accelerator_ids() is sentinel
assert calls==['assign',('install','3'),'assign']
u.set_visible_accelerator_ids=original
ray.get_runtime_context=saved_context
assert not ray.is_initialized()
print('DEFERRED_CALLBACK_CPU_PASS exact_original_calls/return_identity/one_install; synthetic0RL')
