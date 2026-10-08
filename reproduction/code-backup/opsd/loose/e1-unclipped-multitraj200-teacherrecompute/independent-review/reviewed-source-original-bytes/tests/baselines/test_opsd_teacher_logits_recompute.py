"""Actual tiny-Qwen target recomputation, gradients and retained-target boundary."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2ForCausalLM

from search_r1.baselines import distillation
from tests.baselines import test_models_and_driver as fixtures
from tests.baselines.test_opsd_zero_initialization import tensors_and_settings


@pytest.mark.parametrize('chunk', [1, 3])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('padding', [False, True])
@pytest.mark.parametrize('beta,temperature', [(0.,1.), (.4,2.)])
def test_stored_recomputed_loss_all_parameter_gradients_masks_and_teacher_cost(chunk,checkpoint,padding,beta,temperature):
    fixtures.TinyModels.setUpClass()
    try:
        tok=fixtures.TinyModels.tokenizer
        teacher=distillation.FrozenTeacher(fixtures.TinyModels.tmp.name,tok)
        tensors,settings=tensors_and_settings(tok,chunk)
        settings.beta=beta;settings.temperature=temperature;settings.component_clip=None
        settings.loss_checkpointing=checkpoint;settings.teacher_forward_micro_batch=4
        if padding:
            tensors['responses'][1,-1]=tok.pad_token_id
            tensors['input_ids'][1,-1]=tok.pad_token_id
            tensors['attention_mask'][1,-1]=0;tensors['loss_mask'][1,-1]=0
        outputs=[];call_counts=[];saved_targets=[]
        for recompute in (False,True):
            settings.teacher_logits_recompute=recompute
            student=Qwen2ForCausalLM.from_pretrained(fixtures.TinyModels.tmp.name).eval()
            calls=[];retained=[]
            hook=teacher.model.get_output_embeddings().register_forward_hook(lambda *args:calls.append(1))
            def save(value):
                if value.ndim==3 and value.shape[-1]==len(tok) and not value.requires_grad:
                    retained.append((tuple(value.shape),value.dtype))
                return value
            with torch.autograd.graph.saved_tensors_hooks(save,lambda value:value):
                loss=distillation.opsd_loss(student,teacher,tensors,tok,[['Paris'],['Rome']],settings,
                                             tensors['loss_mask'].sum())
                forward_calls=len(calls);forward_cost=teacher.wall_seconds
                loss.backward()
            hook.remove()
            assert forward_calls>1  # Distinct chunks catch late-bound features/settings.
            assert len(calls)==forward_calls*(2 if checkpoint and recompute else 1)
            if checkpoint and recompute:assert teacher.wall_seconds>forward_cost
            assert all(p.grad is None and not p.requires_grad for p in teacher.model.parameters())
            outputs.append((loss.detach(),{name:p.grad.clone() for name,p in student.named_parameters()}))
            call_counts.append(len(calls));saved_targets.append(retained)
        torch.testing.assert_close(outputs[0][0],outputs[1][0],atol=0,rtol=0)
        for name in outputs[0][1]:
            torch.testing.assert_close(outputs[0][1][name],outputs[1][1][name],atol=0,rtol=0)
        if checkpoint:
            assert saved_targets[0]  # Stored checkpoint explicitly retains FP32 q logits.
            assert not saved_targets[1]  # Recomputed checkpoint retains compact features instead.
    finally:fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('checkpoint',[False,True])
@pytest.mark.parametrize('empty',[False,True])
def test_recomputed_masked_nan_and_connected_empty_zero(checkpoint,empty):
    fixtures.TinyModels.setUpClass()
    try:
        tok=fixtures.TinyModels.tokenizer
        teacher=distillation.FrozenTeacher(fixtures.TinyModels.tmp.name,tok)
        tensors,settings=tensors_and_settings(tok,3)
        settings.loss_checkpointing=checkpoint;settings.teacher_logits_recompute=True;settings.component_clip=None
        if empty:tensors['loss_mask'].zero_()
        values=torch.randn((*tensors['input_ids'].shape,len(tok)))
        offset=tensors['input_ids'].shape[-1]-tensors['responses'].shape[-1]-1
        values[1,offset,0]=float('nan');values[1,offset,-1]=float('inf')
        class Direct(torch.nn.Module):
            def __init__(self):super().__init__();self.logits=torch.nn.Parameter(values.clone())
            def forward(self,**kwargs):return SimpleNamespace(logits=self.logits)
        student=Direct()
        loss=distillation.opsd_loss(student,teacher,tensors,tok,[['Paris'],['Rome']],settings,
                                     tensors['loss_mask'].sum().clamp_min(1))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(student.logits.grad).all()
        if empty:assert loss==0 and torch.count_nonzero(student.logits.grad)==0
    finally:fixtures.TinyModels.tearDownClass()


@pytest.mark.parametrize('flag',[None,1,0,'true'])
def test_recompute_requires_actual_bool_before_teacher_work(flag):
    teacher=SimpleNamespace(model=SimpleNamespace(parameters=lambda:()))
    with pytest.raises(ValueError,match='must be a boolean'):
        distillation.opsd_loss(None,teacher,{},None,None,SimpleNamespace(teacher_logits_recompute=flag),1)


def test_recompute_refuses_trainable_teacher():
    teacher=SimpleNamespace(model=torch.nn.Linear(2,2))
    with pytest.raises(ValueError,match='requires a frozen teacher'):
        distillation.opsd_loss(None,teacher,{},None,None,SimpleNamespace(teacher_logits_recompute=True),1)


@pytest.mark.parametrize('checkpoint',[False,True])
def test_recompute_rejects_supervised_nan(checkpoint):
    fixtures.TinyModels.setUpClass()
    try:
        tok=fixtures.TinyModels.tokenizer;teacher=distillation.FrozenTeacher(fixtures.TinyModels.tmp.name,tok)
        tensors,settings=tensors_and_settings(tok,3)
        settings.loss_checkpointing=checkpoint;settings.teacher_logits_recompute=True
        values=torch.randn((*tensors['input_ids'].shape,len(tok)),requires_grad=True)
        offset=tensors['input_ids'].shape[-1]-tensors['responses'].shape[-1]-1
        with torch.no_grad():values[0,offset,-1]=float('nan')
        class Direct:
            def __call__(self,**kwargs):return SimpleNamespace(logits=values)
        with pytest.raises(ValueError,match='Nonfinite OPSD logits'):
            distillation.opsd_loss(Direct(),teacher,tensors,tok,[['Paris'],['Rome']],settings,tensors['loss_mask'].sum())
    finally:fixtures.TinyModels.tearDownClass()
