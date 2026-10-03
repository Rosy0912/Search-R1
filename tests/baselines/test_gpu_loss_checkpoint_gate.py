"""Strict zero-tolerance new GPU diagnostic-result gate, CPU-only validator tests."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('loss_checkpoint_gate',ROOT/'reproduction/baselines/opsd-runtime-20261003/gpu_loss_checkpoint_equivalence.py')
GATE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(GATE)


def evidence():
    modes=[]
    for enabled in (False,True):
        modes.append(dict(loss_checkpointing=enabled,loss=.3797,teacher_head_chunk_lengths=[16,9],teacher_head_forward_calls=2,
            teacher_base_forward_calls=1,teacher_parameter_dtypes=['torch.float32'],teacher_attention='eager',teacher_active_device='cuda:0',
            teacher_head_grad_enabled=[False,False],generalized_jsd_calls=4 if enabled else 2,
            checkpoint_invocations=[False,False] if enabled else [],
            teacher_audit={'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0}))
    return dict(scope='production-eight',leased_gpu_count=8,all_student_parameter_elements_compared=3085938688,
        loss_absolute_difference=0.,student_gradient_max_abs=0.,student_gradient_difference_l2=0.,reference_gradient_l2=20.,
        reference_gradient_max_abs=2.,student_parameter_dtypes=['torch.bfloat16'],student_attention='flash_attention_2',
        student_activation_checkpointing=True,matmul_tf32=False,cudnn_tf32=False,teacher_returned_device='cpu',modes=modes,
        response_width=25,masked_observation_positions=8,supervised_positions=17,retriever_same_job_loopback=True)


def test_exact_complete_short_actual_model_result_is_accepted():
    assert GATE.validate_result(evidence())


@pytest.mark.parametrize('path,value',[
    (('scope',),'unknown'),(('leased_gpu_count',),1),
    (('all_student_parameter_elements_compared',),3085938687),
    (('loss_absolute_difference',),1e-15),
    (('student_gradient_max_abs',),1e-15),
    (('student_gradient_difference_l2',),float('nan')),
    (('reference_gradient_l2',),0.),
    (('student_parameter_dtypes',),['torch.float32']),
    (('student_activation_checkpointing',),False),
    (('matmul_tf32',),True),
    (('response_width',),32),
    (('masked_observation_positions',),0),
    (('modes',1,'teacher_head_forward_calls'),4),
    (('modes',1,'teacher_base_forward_calls'),2),
    (('modes',1,'teacher_head_chunk_lengths'),[9,16]),
    (('modes',1,'teacher_head_grad_enabled'),[True,False]),
    (('modes',1,'generalized_jsd_calls'),2),
    (('modes',1,'checkpoint_invocations'),[True,True]),
    (('modes',1,'teacher_audit','teacher/gradient_tensors_per_rank'),1),
    (('retriever_same_job_loopback',),False),
])
def test_any_nonzero_or_missing_execution_requirement_fails(path,value):
    j=evidence();target=j
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    with pytest.raises((ValueError,KeyError,TypeError)):GATE.validate_result(j)


def test_explicit_isolated_one_scope_does_not_claim_gpu_retriever_coexistence():
    j=evidence();j.update(scope='isolated-one',leased_gpu_count=1,retriever_same_job_loopback=False)
    assert GATE.validate_result(j)
