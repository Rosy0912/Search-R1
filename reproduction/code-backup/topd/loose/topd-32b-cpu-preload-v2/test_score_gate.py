import copy
import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location('gate',Path(__file__).with_name('score_gate.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

@pytest.fixture
def inputs():
    p=torch.ones((16,2048),dtype=torch.long)
    r=torch.ones((16,4096),dtype=torch.long)
    a=torch.ones_like(r);a[15,500:]=0;p[15,:1792]=0
    return dict(prompts=p,responses=r,attention=a,row_indices=torch.arange(16))

def test_actual_long_and_short_input(inputs):
    lengths=g.check_inputs(inputs,0)
    assert lengths[:15]==[6144]*15 and lengths[15]==756

@pytest.mark.parametrize('change',['gold','roworder','wide_padding_only','floatmask','nonbinary','no_mask'])
def test_refuse_fake_context_or_private_input(inputs,change):
    if change=='gold':inputs['targets']=torch.ones(16)
    if change=='roworder':inputs['row_indices']=torch.arange(15,-1,-1)
    if change=='wide_padding_only':inputs['attention'][:,1024:]=0
    if change=='floatmask':inputs['attention']=inputs['attention'].float()
    if change=='nonbinary':inputs['attention'][0,0]=2
    if change=='no_mask':inputs['attention'].fill_(1)
    with pytest.raises(RuntimeError):g.check_inputs(inputs,0)

@pytest.fixture
def record():
    cpu=dict(parameter_count=g.PARAMETERS,parameter_bytes=g.PARAMETERS*4,parameter_sha256='a'*64,
             buffer_sha256='b'*64,training_modules=0,requires_grad_parameters=0,grad_parameters=0,
             parameter_devices=['cpu'],buffer_devices=['cpu'])
    staged=copy.deepcopy(cpu);staged['parameter_devices']=['cuda:%d'%i for i in range(8)]
    return dict(status='RELEASED',original_method_sha256=g.METHOD_SHA,initial_cpu=cpu,staged_frozen=staged,
                released_cpu=copy.deepcopy(cpu),allocators_released=[dict(device=i,uuid='GPU-'+str(i),allocated_bytes=0,reserved_bytes=0) for i in range(8)],
                actual_flags=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,matmul_precision='highest',autocast_enabled=False))

def test_full_release_contract(record):g.assert_release(record,['GPU-'+str(i) for i in range(8)])

@pytest.mark.parametrize('change',['3b','reserved','wronguuid','hash','training','single_gpu','tf32'])
def test_failclosed_release(record,change):
    if change=='3b':record['initial_cpu']['parameter_count']=3_000_000_000
    if change=='reserved':record['allocators_released'][7]['reserved_bytes']=32*1024**2
    if change=='wronguuid':record['allocators_released'][7]['uuid']='foreign'
    if change=='hash':record['released_cpu']['parameter_sha256']='c'*64
    if change=='training':record['staged_frozen']['training_modules']=1
    if change=='single_gpu':record['staged_frozen']['parameter_devices']=['cuda:0']
    if change=='tf32':record['actual_flags']['matmul_allow_tf32']=True
    with pytest.raises(RuntimeError):g.assert_release(record,['GPU-'+str(i) for i in range(8)])

def test_old3b_proof_cannot_enter_native_teacher_gate(tmp_path):
    with pytest.raises(RuntimeError):
        g.check_identity(dict(status='REVIEWED',scope='HARDWARE_SCORER_READY_NO_TRAINING'),tmp_path)


def test_uuid_set_alone_does_not_prove_device_order(record):
    record['allocators_released'][0]['device']=1
    record['allocators_released'][1]['device']=0
    with pytest.raises(RuntimeError):g.assert_release(record,['GPU-'+str(i) for i in range(8)])

@pytest.mark.parametrize('fault',['nan','masked','dtype'])
def test_original_serial_output_checked_directly(inputs,fault):
    value=torch.zeros_like(inputs['responses'],dtype=torch.float32)
    if fault=='nan':value[0,0]=float('nan')
    if fault=='masked':value[15,500]=1
    if fault=='dtype':value=value.to(torch.float64)
    with pytest.raises(RuntimeError):g.check_logp(value,inputs)

def test_literal_parent_flags_cannot_replace_actual_thread_flags():
    values=dict(matmul_allow_tf32=False,cudnn_allow_tf32=False,matmul_precision='highest',autocast_enabled=False)
    rec=dict(threads=[dict(thread=i,context=dict(actual_flags=dict(values))) for i in range(8)])
    g.thread_flags(rec);rec['threads'][7]['context']['actual_flags']['autocast_enabled']=True
    with pytest.raises(RuntimeError):g.thread_flags(rec)
