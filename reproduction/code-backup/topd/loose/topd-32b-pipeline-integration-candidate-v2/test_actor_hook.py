import sys
import types
import textwrap
from pathlib import Path
import pytest
code=textwrap.dedent(Path(__file__).with_name('controller-hook.py').read_text())
namespace={};exec(code,namespace)
hook=namespace['before_actor_update']


@pytest.mark.parametrize('method,backend',[('grpo',None),('topd','cpu'),('topd','local_gpu_fp32_parallel')])
def test_unaffected_paths_do_not_require_new_barrier(method,backend):
    controller=types.SimpleNamespace(method=method,config={'topd':{'teacher_backend':backend}})
    assert hook(controller,types.SimpleNamespace(meta_info={}),1) is None


def test_actual_outer_mismatch_precedes_any_actor_or_barrier(monkeypatch):
    controller=types.SimpleNamespace(method='topd',config={'topd':{'teacher_backend':'local_gpu_fp32_shared_pipeline32b'}},teacher=object())
    with pytest.raises(RuntimeError,match='outer identity'):hook(controller,types.SimpleNamespace(meta_info={'topd_outer_step':1}),2)


def test_failed_ack_propagates_before_actor(monkeypatch):
    calls=[]
    def guard(client,step):calls.append((client,step));raise RuntimeError('actual ACK missing')
    monkeypatch.setitem(sys.modules,'topd32b_shared_pipeline.backend',types.SimpleNamespace(require_global_ack_before_actor=guard))
    teacher=object();controller=types.SimpleNamespace(method='topd',config={'topd':{'teacher_backend':'local_gpu_fp32_shared_pipeline32b'}},teacher=teacher)
    with pytest.raises(RuntimeError,match='ACK missing'):hook(controller,types.SimpleNamespace(meta_info={'topd_outer_step':5}),5)
    assert calls==[(teacher,5)]
