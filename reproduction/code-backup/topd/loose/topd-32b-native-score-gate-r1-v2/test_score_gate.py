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

def test_real_shard_headers_allow_export_layout_and_require_every_tensor(tmp_path):
    import json
    from safetensors.torch import save_file
    save_file({'layer':torch.zeros(2,dtype=torch.bfloat16)},str(tmp_path/'a.safetensors'))
    save_file({'head':torch.zeros(3,dtype=torch.float32)},str(tmp_path/'b.safetensors'))
    index=tmp_path/'model.safetensors.index.json'
    index.write_text(json.dumps(dict(weight_map={'layer':'a.safetensors','head':'b.safetensors'})))
    g.check_hf_tensor_index(tmp_path,['a.safetensors','b.safetensors'],5)
    with pytest.raises(RuntimeError):g.check_hf_tensor_index(tmp_path,['a.safetensors'],5)
    with pytest.raises(RuntimeError):g.check_hf_tensor_index(tmp_path,['a.safetensors','b.safetensors'],6)
    index.write_text(json.dumps(dict(weight_map={'layer':'a.safetensors','head':'b.safetensors','missing':'b.safetensors'})))
    with pytest.raises(RuntimeError):g.check_hf_tensor_index(tmp_path,['a.safetensors','b.safetensors'],5)

@pytest.mark.parametrize('kind,extra,accepted',[
 ('SEARCH_R1_GRPO100_32B','teacher-manifest.json',True),
 ('SEARCH_R1_GRPO100_32B','special_tokens_map.json',True),
 ('SEARCH_R1_GRPO100_32B','added_tokens.json',True),
 ('SEARCH_R1_GRPO100_32B','unreviewed.json',False),
 ('OFFICIAL_QWEN2_5_32B_BASE','teacher-manifest.json',False),
])
def test_actual_export_metadata_complete_inventory(tmp_path,kind,extra,accepted):
    names=g.HF_METADATA|{'a.safetensors',extra}
    for n in names:(tmp_path/n).write_text('{}')
    inv=dict(shards=['a.safetensors'],files={n:{} for n in names})
    if accepted:g.check_hf_file_set(inv,tmp_path,kind)
    else:
        with pytest.raises(RuntimeError):g.check_hf_file_set(inv,tmp_path,kind)
    inv['files'].pop(extra)
    with pytest.raises(RuntimeError):g.check_hf_file_set(inv,tmp_path,kind)

@pytest.mark.parametrize('fault',['none','vocab','special','architecture','illegal_config_id'])
def test_actual_tokenizer_files_and_fsdp_special_overrides(tmp_path,monkeypatch,fault):
    import json
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    root=tmp_path/'root'; student=root/'assets-root/assets/qwen2.5-3b';base=root/'assets-root/assets/qwen2.5-32b';teacher=tmp_path/'teacher'
    base.mkdir(parents=True);teacher.mkdir()
    def save(path,vocab,eos='eos'):
        tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab=vocab,unk_token='unk')),unk_token='unk',eos_token=eos,pad_token='pad')
        tok.save_pretrained(path)
    vocab={'unk':0,'pad':1,'eos':2,'word':3};save(student,vocab)
    other=dict(vocab)
    if fault=='vocab':other['eos'],other['word']=3,2
    save(teacher,other,'word' if fault=='special' else 'eos')
    config=dict(model_type='qwen2',vocab_size=4,hidden_size=8,bos_token_id=3,eos_token_id=2)
    (base/'config.json').write_text(json.dumps(config));current=dict(config,torch_dtype='float32',_name_or_path='export',transformers_version='other')
    current['bos_token_id']=None  # Legal FSDP HF export overrides base config BOS.
    current['pad_token_id']=1
    if fault=='architecture':current['hidden_size']=16
    if fault=='illegal_config_id':current['eos_token_id']=99
    (teacher/'config.json').write_text(json.dumps(current));monkeypatch.setattr(g,'ROOT',root)
    if fault=='none':g.check_tokenizer_config_identity(teacher)
    else:
        with pytest.raises(RuntimeError):g.check_tokenizer_config_identity(teacher)

@pytest.fixture
def trained_measurement():
    raw={'status':'COMPLETE64_UNREVIEWED_NO_TRAINING','quality':{'denominator':64,'correct':0,'second_quality_cutoff_applied':False}}
    supervised={'status':'COMPLETE64_UNREVIEWED_NO_TRAINING','sampler_errors':[],
                'parent_cuda_initialized':False,'foreign_baseline_missing':[],
                'cleanup':{'actual_cgroup_empty':True},'terminal':{'monitor_stop':{'stopped':True}},'monitor_stop':{'stopped':True}}
    model=Path('/actual/trained100');quality={'correct':0,'teacher_model_directory':str(model)}
    return raw,supervised,quality,model

def test_trained64_has_no_second_quality_cutoff(trained_measurement):
    g.verify_trained_measurement(*trained_measurement)

@pytest.mark.parametrize('fault',['runtimeFAIL','samplerERROR','terminalNOTjoined','cleanupNOTempty','differentMODEL','differentCOUNT'])
def test_trained64_actual_failures_do_not_promote_teacher(trained_measurement,fault):
    raw,sup,quality,model=trained_measurement
    if fault=='runtimeFAIL':sup['status']='FAIL_NO_TRAINING'
    if fault=='samplerERROR':sup['sampler_errors']=['NVML6_UNKNOWN_NOT_ACCEPTED']
    if fault=='terminalNOTjoined':sup['terminal']['monitor_stop']['stopped']=False
    if fault=='cleanupNOTempty':sup['cleanup']['actual_cgroup_empty']=False
    if fault=='differentMODEL':quality['teacher_model_directory']='/official/base'
    if fault=='differentCOUNT':raw['quality']['correct']=1
    with pytest.raises(RuntimeError):g.verify_trained_measurement(raw,sup,quality,model)

def test_base_fp32_identity_is_never_claimed_as_trained():
    assert g.BASE_FP32_SHA=='a80140df56c39e1e0f99e29473bd1b6f3668d47cccfa05240514f560705d0d16'
    import ast
    source=Path(g.__file__).read_text();tree=ast.parse(source)
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and "kind == 'SEARCH_R1_GRPO100_32B'"==ast.unparse(n.test))
    checks=[n for n in ast.walk(branch) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require']
    check=next(n for n in checks if n.args and "expected_fp32_parameter_sha256" in ast.unparse(n.args[0]))
    # Evaluate the actual production guard, not a separately written condition.
    expr=compile(ast.Expression(check.args[0]),'<actual-trained-hash-guard>','eval')
    assert eval(expr,{'provenance':{'base_fp32_parameter_sha256':g.BASE_FP32_SHA,'expected_fp32_parameter_sha256':'b'*64},'BASE_FP32_SHA':g.BASE_FP32_SHA})
    assert not eval(expr,{'provenance':{'base_fp32_parameter_sha256':'c'*64,'expected_fp32_parameter_sha256':g.BASE_FP32_SHA},'BASE_FP32_SHA':g.BASE_FP32_SHA})
