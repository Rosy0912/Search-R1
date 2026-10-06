"""Local tiny-Qwen producer/consumer joins; never official GPU acceptance."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2ForCausalLM

from search_r1.baselines.distillation import FrozenTeacher
from tests.baselines import test_models_and_driver as fixtures

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT.parent/'e1-unclipped-multitraj200-teacherrecompute/formal-tools-candidate'


def load_source(path, name):
    import types
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def capacity_fixture(formal, numeric):
    # This declared local fixture does not claim a job,8GPU,Adam or RL evidence.
    return dict(passed=True, source_commit=formal.SOURCE, job_id='LOCAL_CPU_FIXTURE',
        world_size=8, optimizer_steps=2, component_clip=None, response_only_logits=False,
        teacher_forward_micro_batch=4,teacher_logits_recompute=True, local_micro=4, input_width=6144,
        response_width=4096, minimum_required_free_fraction=.15,
        estimated_peak_device_used_bytes=60., gpu_total_bytes=100.,
        short_native_teacher_logits_recompute_equivalence=numeric,
        old_logprob_capacity=dict(passed=True,local_micro=8,global_micro=64,
            logprob_shape=[8,4096],entropy_shape=[8,4096],logprob_finite=True,
            entropy_finite=True,estimated_peak_device_used_bytes=40.))


@pytest.fixture
def formal(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    return load_source(TOOLS/'formal_supervisor.py', 'local_cpu_native_head_formal')


@pytest.fixture
def numeric():
    return dict(comparison_kind='NATIVE_FULL_HEAD_FULL_TEACHER4_STORED_VS_RECOMPUTED_TARGETS',
        response_only_logits=False, stored_teacher_forward_micro_batch=4,
        recomputed_teacher_forward_micro_batch=4, passed=True,
        global_native_stored_squared_gradient_norm=1.,global_native_recomputed_squared_gradient_norm=1.,
        shard_gradient_relative_l2=0.,shard_gradient_cosine=1.,
        loss_relative_difference=0.,teacher_q_mean_probability_l1_difference=0.,stored_teacher_logits_recompute=False,recomputed_teacher_logits_recompute=True,teacher_projection_calls=dict(stored_forward=2,stored_backward=0,recomputed_forward=2,recomputed_backward=2),teacher_projection_forward_feature_sha256=['a'*64,'b'*64],teacher_projection_backward_feature_sha256=['a'*64,'b'*64],shard_gradient_cosine_raw=1.,shard_gradient_cosine_roundoff_applied=False,maximum_cosine_roundoff=1e-12,teacher_q_forward_mean_probability_l1_difference=0.,teacher_q_backward_mean_probability_l1_difference=0.)


@pytest.mark.parametrize('checkpoint', [True])
def test_real_cpu_qwen_probe_writer_to_formal_consumer(formal, monkeypatch, checkpoint):
    """Run the real comparison math; only CUDA sync/collective are CPU fixtures."""
    fixtures.TinyModels.setUpClass()
    try:
        tok = fixtures.TinyModels.tokenizer
        tok.chat_template = (
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + "
            "message['content'] + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
        # Explicit tiny-model ChatML fixture: control-token boundaries cannot
        # merge with adjacent BPE text as ordinary vocabulary strings.
        tok.add_special_tokens({'additional_special_tokens':['<|im_start|>','<|im_end|>']})
        tok.save_pretrained(fixtures.TinyModels.tmp.name)
        seed_model=Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name)
        seed_model.resize_token_embeddings(len(tok))
        seed_model.save_pretrained(fixtures.TinyModels.tmp.name)
        teacher = FrozenTeacher(fixtures.TinyModels.tmp.name, tok)
        student = Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
        probe = load_source(ROOT/'reproduction/baselines/e1-multitraj200/microbatch_probe.py',
                            'local_cpu_native_head_probe')
        monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
        monkeypatch.setattr(torch.distributed,'all_reduce',lambda *a,**k:None)
        settings = SimpleNamespace(chunk_size=16,beta=0.,temperature=1.,component_clip=None,
            teacher_prompt_mode='user_context',max_student_logits_gib=8.,
            loss_checkpointing=checkpoint,response_only_logits=False,teacher_forward_micro_batch=4,teacher_logits_recompute=True)
        shape_calls=[]
        hook=student.lm_head.register_forward_pre_hook(lambda _,args:shape_calls.append(tuple(args[0].shape)))
        teacher_batches=[]
        teacher_hook=teacher.model.base_model.register_forward_pre_hook(
            lambda _,args,kwargs:teacher_batches.append(kwargs['input_ids'].shape[0]),with_kwargs=True)
        numeric=probe.gpu_teacher_logits_recompute_equivalence(student,teacher,tok,torch.device('cpu'),settings)
        hook.remove()
        teacher_hook.remove()
        assert shape_calls == [(4,96,16),(4,96,16)]
        assert teacher_batches == [4,4]  # One native whole teacher forward per call.
        assert numeric['passed'] is True
        assert numeric['response_only_logits'] is False
        assert numeric['comparison_kind']=='NATIVE_FULL_HEAD_FULL_TEACHER4_STORED_VS_RECOMPUTED_TARGETS'
        assert (numeric['stored_teacher_forward_micro_batch'],numeric['recomputed_teacher_forward_micro_batch'])==(4,4)
        assert numeric['teacher_FP32_logits_max_absolute_difference']==0
        assert numeric['teacher_q_mean_probability_l1_difference']==0
        assert numeric['shard_gradient_relative_l2']==0
        assert numeric['teacher_projection_calls']==dict(stored_forward=2,stored_backward=0,recomputed_forward=2,recomputed_backward=2)
        assert numeric['teacher_projection_forward_feature_sha256']==numeric['teacher_projection_backward_feature_sha256']
        assert len(numeric['teacher_projection_forward_feature_sha256'])==2
        assert numeric['teacher_cost_intervals'][0]['backward_teacher_host_wall_seconds']==0
        assert numeric['teacher_cost_intervals'][1]['backward_teacher_host_wall_seconds']>0
        assert numeric['maximum_relative_gradient_l2']==.02
        assert numeric['minimum_gradient_cosine']==.9999
        assert numeric['maximum_relative_loss']==.002
        assert numeric['maximum_teacher_q_mean_l1_difference']==1e-4
        assert numeric['checked_gradient_shards']==len(list(student.parameters()))
        assert all(p.grad is None for p in teacher.model.parameters())
        formal.validate_capacity_rank(capacity_fixture(formal,numeric),'LOCAL_CPU_FIXTURE')
    finally:
        fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('key,value',[
    ('response_only_logits',True),('response_only_logits',0),
    ('teacher_forward_micro_batch',True),('teacher_forward_micro_batch',4.),
    ('teacher_forward_micro_batch',2),('teacher_logits_recompute',False),('teacher_logits_recompute',1),
])
def test_capacity_flag_and_teacher_row_type_refused(formal,numeric,key,value):
    row=capacity_fixture(formal,numeric);row[key]=value
    with pytest.raises(ValueError):formal.validate_capacity_rank(row,'LOCAL_CPU_FIXTURE')


@pytest.mark.parametrize('key,value',[
    ('response_only_logits',True),('stored_teacher_forward_micro_batch',4.),
    ('recomputed_teacher_forward_micro_batch',4.),('recomputed_teacher_forward_micro_batch',2),
    ('comparison_kind','TEACHER4_VS2_EQUIVALENCE'),('global_native_stored_squared_gradient_norm',True),
    ('global_native_recomputed_squared_gradient_norm',float('inf')),('shard_gradient_relative_l2',-1.),
    ('shard_gradient_relative_l2',.0200001),('shard_gradient_cosine',.9998999),
    ('shard_gradient_cosine',1.01),('loss_relative_difference',-.001),
    ('loss_relative_difference',.0020001),('teacher_q_mean_probability_l1_difference',.000100001),
    ('teacher_q_mean_probability_l1_difference',float('nan')),
])
def test_numeric_typed_range_and_unchanged_thresholds_refused(formal,numeric,key,value):
    numeric[key]=value
    with pytest.raises(ValueError):formal.validate_capacity_rank(capacity_fixture(formal,numeric),'LOCAL_CPU_FIXTURE')


@pytest.mark.parametrize('legacy_field',['short_native_response_head_equivalence','short_native_teacher_row_equivalence','short_native_full_path_repeatability'])
def test_legacy_equivalence_objects_not_accepted(formal,numeric,legacy_field):
    row=capacity_fixture(formal,numeric)
    row[legacy_field]=row.pop('short_native_teacher_logits_recompute_equivalence')
    with pytest.raises(KeyError):formal.validate_capacity_rank(row,'LOCAL_CPU_FIXTURE')


def test_full_width_full_head_logits_conservative_upper_bound():
    #152064 is a conservative vocabulary upper bound; runtime reads the actual
    #official model config vocabulary size151936 and allocates its full head.
    assert 4*6144*152064*2/1024**3 == 6.9609375
    assert 4*6144*151936*2/1024**3 == 6.955078125
    assert 6.9609375 < 8


@pytest.mark.parametrize('raw,expected,rounded',[(1.,1.,False),(-1.,-1.,False),(.9,.9,False),
    (1.+4e-16,1.,True),(-1.-4e-16,-1.,True)])
def test_cosine_roundoff_preserves_actual_raw_and_only_normalizes_diagnostic_boundary(raw,expected,rounded):
    probe=load_source(ROOT/'reproduction/baselines/e1-multitraj200/microbatch_probe.py','cosine_local_fixture')
    assert probe.cosine_roundoff(raw)==(expected,rounded)


@pytest.mark.parametrize('raw',[True,float('nan'),float('inf'),1.+2e-12,-1.-2e-12])
def test_cosine_outside_tiny_roundoff_is_rejected(raw):
    probe=load_source(ROOT/'reproduction/baselines/e1-multitraj200/microbatch_probe.py','cosine_negative_fixture')
    with pytest.raises(ValueError):probe.cosine_roundoff(raw)


def test_formal_accepts_only_raw_preserved_cosine_roundoff(formal,numeric):
    numeric['shard_gradient_cosine_raw']=1.+4e-16
    numeric['shard_gradient_cosine_roundoff_applied']=True
    formal.validate_capacity_rank(capacity_fixture(formal,numeric),'LOCAL_CPU_FIXTURE')
    numeric['shard_gradient_cosine_raw']=1.+2e-12
    with pytest.raises(ValueError):formal.validate_capacity_rank(capacity_fixture(formal,numeric),'LOCAL_CPU_FIXTURE')


@pytest.mark.parametrize('field,value',[
    ('stored_teacher_logits_recompute',0),('recomputed_teacher_logits_recompute',False),
    ('teacher_projection_calls',dict(stored_forward=2.,stored_backward=0,recomputed_forward=2,recomputed_backward=2)),
    ('teacher_projection_calls',dict(stored_forward=2,stored_backward=0,recomputed_forward=2,recomputed_backward=0)),
    ('teacher_projection_backward_feature_sha256',['b'*64,'a'*64]),
    ('shard_gradient_cosine_raw',True),('shard_gradient_cosine_roundoff_applied',1),
    ('teacher_q_backward_mean_probability_l1_difference',.0001),
])
def test_recomputed_projection_phase_feature_and_typed_flags_required(formal,numeric,field,value):
    numeric[field]=value
    with pytest.raises(ValueError):formal.validate_capacity_rank(capacity_fixture(formal,numeric),'LOCAL_CPU_FIXTURE')
