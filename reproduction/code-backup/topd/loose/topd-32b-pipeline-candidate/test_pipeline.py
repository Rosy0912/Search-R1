"""CPU-only prototype contracts; never native/shared-GPU readiness evidence."""
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

source=Path(__file__).resolve().parents[1]/'topd'
if not source.is_dir():
    import os
    source=Path(os.environ['AGENT_ROOT'])/'repo'
sys.path.insert(0,str(source))
from search_r1.baselines.distillation import FrozenTeacher
sys.path.insert(0,str(Path(__file__).parent))
import pipeline as scorer


class CpuFixtureRuntime:
    device='cpu';kind='CPU_ONLY_FIXTURE_NOT_HARDWARE_EVIDENCE'
    def __init__(self,fail=None):self.calls=[];self.fail=fail
    def verify(self):self.calls.append('verify')
    def synchronize_device(self,device):self.calls.append('device_sync')
    def thread_context(self):return dict(cpu_fixture=True,thread=__import__('threading').get_ident())
    def flags(self):self.calls.append('flags');return dict(cpu_fixture=True)
    def synchronize(self):self.calls.append('synchronize')
    def reset_peaks(self):self.calls.append('reset_peaks')
    def peaks(self):return [dict(cpu_fixture=True) for _ in range(8)]
    def dispatch(self,model,mapping):
        self.calls.append('dispatch')
        if self.fail=='stage':raise RuntimeError('deliberate partial staging failure')
        return model
    def verify_partition(self,model,inventory):self.calls.append('fixture_partition_no_cuda')
    def unhook(self,model):self.calls.append('unhook')
    def gpu_identity(self):return [dict(cpu_fixture=True)]
    def release_allocators(self):
        self.calls.append('release_allocators')
        return [dict(device=i,allocated_bytes=0,reserved_bytes=32 if self.fail=='release' and i==7 else 0) for i in range(8)]


@pytest.fixture
def fixture():
    torch.manual_seed(42);torch.set_num_threads(1)
    cfg=Qwen2Config(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=8192,pad_token_id=0)
    cfg._attn_implementation='eager'
    teacher=FrozenTeacher.__new__(FrozenTeacher);teacher.tokenizer=SimpleNamespace(pad_token_id=0);teacher.device=torch.device('cpu')
    teacher.model=Qwen2ForCausalLM(cfg).float().eval().requires_grad_(False);teacher.scored_tokens=0
    runtime=CpuFixtureRuntime();mapping={name:'cuda:0' for name in ['model.embed_tokens','model.rotary_emb','model.norm','lm_head','model.layers.0','model.layers.1']}
    shared=scorer.PipelineScorer(teacher,runtime,mapping,range(128),expected_parameters=sum(p.numel() for p in teacher.model.parameters()))
    inputs=dict(prompts=torch.tensor([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]),responses=torch.tensor([[15,16,0]]*5),attention=torch.tensor([[1,1,0]]*5),row_indices=torch.arange(5))
    return shared,inputs,runtime



@pytest.mark.parametrize('workers',[1,2,4,8])
def test_original_tiny_qwen_two_rounds_math_mask_order_frozen_and_barrier(fixture,workers):
    shared,inputs,runtime=fixture
    expected=torch.cat([shared.original(shared.teacher, inputs['prompts'][i:i+1], inputs['responses'][i:i+1], inputs['attention'][i:i+1], targets=None,chunk=16) for i in range(5)])
    initial=scorer.snapshot(shared.teacher.model)
    for _ in range(2):
        output,record=shared.score_round(inputs,workers=workers)
        assert torch.equal(output,expected)
        assert record['scored_tokens']==24 and record['warmup_tokens']==4*workers
        assert len({t['thread'] for t in record['threads']})==workers
        assert min(t['first_delivered_row_started_epoch'] for t in record['threads'] if t['rows'])>=max(t['warmup_completed_epoch'] for t in record['threads'])
        assert sorted(i for t in record['threads'] for i in t['rows'])==list(range(5))
        assert record['all_threads_joined'] and record['status']=='RELEASED'
        assert record['released_epoch']>=record['threads_joined_epoch']
        assert scorer.reference.frozen_equal(initial,scorer.snapshot(shared.teacher.model))
        assert not any(p.grad is not None for p in shared.teacher.model.parameters())
    assert runtime.calls.count('dispatch')==2


def test_unwrap_source_identity_is_teacher_not_torch_contextlib(fixture):
    shared,_,_=fixture
    assert shared.original_file.name=='distillation.py'
    assert scorer.reference.method_sha(shared.original)==scorer.reference.ORIGINAL_METHOD_SHA


@pytest.mark.parametrize('failure',['stage','release'])
def test_partial_stage_or_release_failure_refuses_output_and_preserves_failure(fixture,failure):
    shared,inputs,runtime=fixture;runtime.fail=failure
    with pytest.raises(RuntimeError):shared.score_round(inputs,workers=2)
    assert shared.last_record['status']=='FAILED'
    assert 'release_allocators' in runtime.calls
    if failure=='release':assert shared.state=='FAILED'


def test_worker_exception_joins_before_unhook_and_stops_admission(fixture):
    shared,inputs,runtime=fixture
    real=shared.teacher.model.model.layers[0].forward
    def fail(*a,**kw):raise RuntimeError('deliberate real-layer failure')
    shared.teacher.model.model.layers[0].forward=fail
    with pytest.raises(RuntimeError,match='Row scoring failed'):shared.score_round(inputs,workers=4)
    assert shared.last_record['all_threads_joined'] and shared.state=='RELEASED'
    assert all(not t.name.startswith('teacher-pipeline-') for t in __import__('threading').enumerate())
    assert 'unhook' in runtime.calls
    shared.teacher.model.model.layers[0].forward=real


def test_module_permit_is_exclusive_and_sync_on_forward_failure():
    import threading,time
    model=torch.nn.Sequential(torch.nn.Linear(2,2));events=[];active=[0];maximum=[0]
    original=model[0].forward
    def forward(x):
        active[0]+=1;maximum[0]=max(maximum[0],active[0]);time.sleep(.01)
        try:return original(x)
        finally:active[0]-=1
    model[0].forward=forward
    runtime=SimpleNamespace(synchronize_device=lambda device:events.append(device))
    permits=scorer.DevicePermits(model,{'0':'cuda:0'},runtime)
    threads=[threading.Thread(target=lambda:model(torch.ones(1,2))) for _ in range(4)]
    for t in threads:t.start()
    for t in threads:t.join()
    permits.close()
    assert maximum[0]==1 and len(events)==4 and model[0].forward is forward


def test_no_gold_unknown_worker_and_dynamic_rope_refused(fixture):
    shared,inputs,_=fixture
    with pytest.raises(RuntimeError):shared.score_round(dict(inputs,gold=torch.tensor([1])))
    with pytest.raises(RuntimeError):shared.score_round(inputs,workers=3)
    shared.teacher.model.config.rope_scaling={'type':'dynamic','factor':2}
    with pytest.raises(RuntimeError,match='RoPE'):shared.score_round(inputs,workers=2)
    assert shared.state=='RELEASED'


def test_out_of_order_completion_restores_original_row_indices(fixture):
    import time
    shared,inputs,_=fixture
    original=shared.original
    expected=torch.cat([original(shared.teacher,inputs['prompts'][i:i+1],inputs['responses'][i:i+1],inputs['attention'][i:i+1],targets=None,chunk=16) for i in range(5)])
    def delayed(view,p,r,a,**kwargs):
        if int(p[0,0])==0:time.sleep(.05)
        return original(view,p,r,a,**kwargs)
    shared.original=delayed
    output,record=shared.score_round(inputs,workers=4)
    assert record['completion_order']!=list(range(5))
    assert torch.equal(output,expected)


def test_streamed_hash_exactly_matches_original_ordered_identity(fixture):
    shared,_,_=fixture
    assert scorer.snapshot(shared.teacher.model)==scorer.reference.snapshot(shared.teacher.model)
