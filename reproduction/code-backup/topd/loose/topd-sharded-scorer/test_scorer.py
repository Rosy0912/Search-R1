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
spec=importlib.util.spec_from_file_location('shared_scorer',Path(__file__).with_name('scorer.py'))
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)


class CpuFixtureRuntime:
    device='cpu';kind='CPU_ONLY_FIXTURE_NOT_HARDWARE_EVIDENCE'
    def __init__(self,fail=None):self.calls=[];self.fail=fail
    def verify(self):self.calls.append('verify')
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
    shared=scorer.SharedScorer(teacher,runtime,mapping,range(128),expected_parameters=sum(p.numel() for p in teacher.model.parameters()))
    inputs=dict(prompts=torch.tensor([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]),responses=torch.tensor([[15,16,0]]*5),attention=torch.tensor([[1,1,0]]*5),row_indices=torch.arange(5))
    return shared,inputs,runtime


def test_original_method_exact_source_and_unequal_full_model_vocab_allowed():
    assert scorer.method_sha(FrozenTeacher.log_probs)==scorer.ORIGINAL_METHOD_SHA
    assert 'F.log_softmax(logits, -1)' in inspect.getsource(FrozenTeacher.log_probs)
    assert scorer.VOCAB_32B==152064  # No student151936 clipping/renormalization.


def test_balanced_complete64layer_map():
    mapping=scorer.device_map_32b();counts=[sum(value==f'cuda:{i}' for key,value in mapping.items() if key.startswith('model.layers.')) for i in range(8)]
    assert counts==[6,9,9,8,8,8,8,8]
    assert all(mapping[key]=='cuda:0' for key in ['model.embed_tokens','model.norm','model.rotary_emb','lm_head'])
    assert len(mapping)==68


@pytest.mark.parametrize('batch',[1,4,8])
def test_two_rounds_preserve_original_function_row_mask_hash_and_tokens(fixture,batch):
    shared,inputs,runtime=fixture;teacher=shared.teacher
    original=FrozenTeacher.log_probs(teacher,inputs['prompts'],inputs['responses'],inputs['attention'],temperature=1.,targets=None,chunk=16)
    before=scorer.snapshot(teacher.model)
    for round in [1,2]:
        output,record=shared.score_round(inputs,batch_size=batch)
        assert torch.allclose(output,original,atol=1e-4,rtol=1e-4)
        assert record['round']==round and record['status']=='RELEASED' and record['runtime_kind'].startswith('CPU_ONLY')
        assert record['scored_tokens']==24 and bool((output[inputs['attention']==0]==0).all())
        assert shared.state=='RELEASED' and scorer.frozen_equal(before,scorer.snapshot(teacher.model))
    assert runtime.calls.count('dispatch')==runtime.calls.count('unhook')==runtime.calls.count('release_allocators')==2


@pytest.mark.parametrize('change',['gold','float','row_order','mask','unmapped','long_prompt','long_response','empty_prompt','batch'])
def test_bad_requests_fail_before_staging(fixture,change):
    shared,inputs,runtime=fixture;inputs={k:v.clone() for k,v in inputs.items()};batch=4
    if change=='gold':inputs['targets']=torch.tensor([1])
    elif change=='float':inputs['responses']=inputs['responses'].float()
    elif change=='row_order':inputs['row_indices']=torch.arange(4,-1,-1)
    elif change=='mask':inputs['attention'][0,0]=2
    elif change=='unmapped':inputs['responses'][0,0]=128
    elif change=='long_prompt':inputs['prompts']=torch.ones((5,2049),dtype=torch.long)
    elif change=='long_response':inputs['responses']=torch.ones((5,4097),dtype=torch.long);inputs['attention']=torch.ones_like(inputs['responses'])
    elif change=='empty_prompt':inputs['prompts'][0]=0
    else:batch=2
    with pytest.raises(RuntimeError):shared.score_round(inputs,batch_size=batch)
    assert 'dispatch' not in runtime.calls


@pytest.mark.parametrize('fail',['stage','release'])
def test_partial_stage_always_cleans_and_nonzero8thallocator_never_passes(fixture,fail):
    shared,inputs,runtime=fixture;runtime.fail=fail
    with pytest.raises(RuntimeError):shared.score_round(inputs)
    assert 'unhook' in runtime.calls and 'release_allocators' in runtime.calls
    assert shared.rounds==0
    if fail=='release':assert shared.state=='FAILED'


def test_concurrent_request_rejected(fixture):
    shared,inputs,_=fixture;shared.lock.acquire()
    try:
        with pytest.raises(RuntimeError,match='concurrent'):shared.score_round(inputs)
    finally:shared.lock.release()


def test_missing_parameter_coverage_rejected(fixture):
    shared,_,_=fixture;mapping=dict(shared.mapping);mapping.pop('model.norm')
    with pytest.raises(RuntimeError,match='Unmapped'):scorer.parameter_map(shared.teacher.model,mapping)


@pytest.mark.parametrize('correct',[25,26,64])
def test_same64_quality_cutoff_without_rounding(correct):
    receipt=dict(status='PASS',teacher_kind='QWEN2_5_32B_BASE_QUALITY_TEST',questions=64,correct=correct,greedy=True,pass_at_k=1,metric='EM',teacher_model='Qwen/Qwen2.5-32B',question_sha256='a'*64,teacher_weights_inventory_sha256='b'*64,student_comparison_sha256='c'*64)
    if correct<26:
        with pytest.raises(RuntimeError):scorer.quality_gate(receipt)
    else:assert scorer.quality_gate(receipt)=='b'*64


def test_scoring_exception_still_offloads_and_does_not_publish_output(fixture):
    shared,inputs,runtime=fixture
    def fail(*args,**kwargs):raise RuntimeError('deliberate score failure')
    shared.original=fail
    with pytest.raises(RuntimeError):shared.score_round(inputs)
    assert 'unhook' in runtime.calls and 'release_allocators' in runtime.calls and shared.rounds==0


def test_bfloat16_parameter_is_rejected_before_native_stage(fixture):
    shared,_,runtime=fixture
    shared.teacher.model.bfloat16()
    with pytest.raises(RuntimeError,match='Non-FP32'):scorer.snapshot(shared.teacher.model)
    assert 'dispatch' not in runtime.calls
