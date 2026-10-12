"""Tiny CPU fixtures only: no actual formal submission, GPU reload or receipt."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('aligned_cpu',HERE/'aligned_first5_cpu_evidence.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


@pytest.fixture(autouse=True)
def hidden_cuda(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')


def test_cpu_guard_before_load_and_after_failed_native_inspection(monkeypatch,tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError,match='empty CUDA_VISIBLE'):helper.cpu_only()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    with pytest.raises(ValueError,match='CPU-only'):helper.cpu_only()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    initialized=[False]
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:initialized[0])
    def fail(*args,**kwargs):
        initialized[0]=True
        raise RuntimeError('DTensor attempted CUDA')
    monkeypatch.setattr(torch,'load',fail)
    with pytest.raises(ValueError,match='CPU-only'):
        helper.native_cpu_inspection([dict(rank=0,path=str(tmp_path/'native.pt'),sha256='a'*64)])


def test_borrowed_official_model_is_explicitly_pinned(monkeypatch,tmp_path):
    root=tmp_path/'root';target=tmp_path/'reviewed';target.mkdir();(root/'assets').mkdir(parents=True)
    logical=root/'assets/qwen2.5-3b-instruct-aa8e725';logical.symlink_to(target,target_is_directory=True)
    monkeypatch.setattr(helper,'ROOT',root);monkeypatch.setattr(helper,'BORROWED_MODEL',target)
    assert helper.official_model_path(logical)==target.resolve()
    other=tmp_path/'other';other.mkdir();logical.unlink();logical.symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError,match='borrowed'):helper.official_model_path(logical)


def submit(root=helper.ROOT):
    token='b'*32;job='9999'
    return dict(source=helper.SOURCE,root=str(root),job=job,token=token,
        run=str(root/'runs'/('formal-aligned-em200-'+token+'-'+job)),
        comment='BAPO_ALIGNED_EM200:formal:'+helper.SOURCE+':'+token,
        scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION')


@pytest.mark.parametrize('change',[{'job':'3315'},{'job':'3345'},{'source':'a'*40},{'run':'/old/run'},
    {'token':'unbound'},{'scope':'CPU_FIXTURE'},{'root':'/foreign'}])
def test_original_new_formal_receipt_only(change):
    record=submit();record.update(change)
    with pytest.raises(ValueError):helper.identity(record)


def test_actual_dynamic_new_job_no_historical_constraint():
    value=helper.identity(submit())
    assert value['job']=='9999'and value['run'].name.endswith('-9999')


def test_exclusive_evidence_and_symlink(tmp_path):
    root=tmp_path.resolve();path=root/'evidence.json'
    helper.write_new(path,{'acceptance':False,'release_receipt':False})
    with pytest.raises(FileExistsError):helper.write_new(path,{'acceptance':True})
    alias=root/'alias';alias.symlink_to(path)
    with pytest.raises(ValueError,match='noncanonical'):helper.binding(alias)
    assert json.loads(path.read_text())['acceptance']is False


def tensor_files(path, tensors):
    from safetensors.torch import save_file
    path.mkdir();save_file(tensors,str(path/'model.safetensors'))


def test_full_scan_rejects_rounding_only_and_nonfinite_later_tensor(tmp_path):
    base=tmp_path/'base';cp=tmp_path/'cp'
    a=torch.tensor([1.0001,2.0001]);b=a.half()
    tensor_files(base,{'first':a,'last':torch.tensor([3.])})
    tensor_files(cp,{'first':b,'last':torch.tensor([3.],dtype=torch.float16)})
    with pytest.raises(ValueError,match='no dtype-aligned'):helper.complete_weight_scan(base,cp)
    from safetensors.torch import save_file
    save_file({'first':b+1,'last':torch.tensor([float('nan')],dtype=torch.float16)},str(cp/'model.safetensors'))
    with pytest.raises(ValueError,match='nonfinite'):helper.complete_weight_scan(base,cp)
    save_file({'first':b+1,'last':torch.tensor([3.],dtype=torch.float16)},str(cp/'model.safetensors'))
    proof=helper.complete_weight_scan(base,cp)
    assert proof['changed_elements']==2 and proof['tensors_checked']==2 and proof['all_official_and_export_tensors_finite']


def tiny_hf(root):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
    base=root/'base';cp=root/'checkpoint';base.mkdir();cp.mkdir()
    tok=Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'[EOS]':2,'[BOS]':3,'The':4,'capital':5,'of':6,'France':7,'is':8},unk_token='[UNK]'))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='[PAD]',unk_token='[UNK]',eos_token='[EOS]',bos_token='[BOS]')
    tokenizer.model_input_names=['input_ids','attention_mask']
    tokenizer.chat_template="{% for message in messages %}{{message['content']}}{% endfor %}{% if add_generation_prompt %} is{% endif %}"
    config=Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32,tie_word_embeddings=True,pad_token_id=0,eos_token_id=2,bos_token_id=3)
    model=Qwen2ForCausalLM(config);model.save_pretrained(base);tokenizer.save_pretrained(base)
    with torch.no_grad():model.model.layers[0].mlp.up_proj.weight.add_(.1)
    model.save_pretrained(cp);tokenizer.save_pretrained(cp)
    return base,cp


def test_actual_tiny_cpu_hf_load_tokenizer_tied_finite_forward_and_delta(tmp_path):
    base,cp=tiny_hf(tmp_path)
    proof=helper.cpu_hf(base,cp,official=False)
    assert proof['scope']=='CPU_FIXTURE_ONLY'and proof['tied_input_output_storage']
    assert proof['loading_info']['missing_keys']==[]and proof['all_parameters_finite']
    assert proof['forward']['device']=='cpu'and proof['forward']['finite']
    assert proof['tokenizer']['full_vocabulary_equal']and proof['parameter_delta']['changed_elements']>0
    assert proof['cost']['wall_seconds']>0 and not torch.cuda.is_initialized()
    with pytest.raises(ValueError,match='official Qwen2.5'):helper.cpu_hf(base,cp,official=True)


def test_native_cpu_unavailable_never_fabricates_reload(tmp_path):
    path=tmp_path/'rank.pt';path.write_bytes(b'not a valid dtensor archive')
    row=helper.native_cpu_inspection([{'rank':0,'path':str(path),'sha256':helper.digest(path)}])[0]
    assert row['status']=='CPU_NATIVE_INSPECTION_UNAVAILABLE_NOT_RELOADED'and row['reloaded']is False
    assert row['error']and row['traceback']


def test_old_gate_rejected_before_any_cpu_state_load(tmp_path):
    root=tmp_path.resolve();receipt=submit(root);run=helper.identity(receipt,root)['run']
    manifest=run/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'version':2,'protocol':'BAPO_FRESH200_V1'}))
    with pytest.raises(ValueError,match='manifest identity'):
        helper.first5_evidence(manifest,receipt,root,SimpleNamespace(),root=root,processes=False)


def test_actual_five_driver_manifest_plumbing_never_signs_release(tmp_path,monkeypatch):
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    from search_r1.baselines.full_state import state_digest
    from search_r1.baselines import fresh200_gate
    root=tmp_path.resolve();submission=submit(root);run=helper.identity(submission,root)['run']
    directory=run/'checkpoints/fresh200_state/outer_5_COMPLETE';directory.mkdir(parents=True)
    marker=run/'source-commit.txt';marker.write_text(helper.SOURCE)
    config=directory/'resolved-config.json';config.write_text(json.dumps({'baselines':{'algorithm_source_commit':helper.SOURCE,'alignment':{'common_manifest_sha256':helper.COMMON_SHA}}}))
    paths=[marker,config];progress=Fresh200Progress(baseline_aligned=True);attempts=[];events=[]
    for outer in range(1,6):
        attempt_id=progress.begin_attempt(outer)
        ranks=[dict(rank=r,baseline_aligned=True,actual_outer=outer,attempt_id=attempt_id,status='COMPLETE',
            optimizer_steps=20,expected_optimizer_steps=20,optimizer_opportunities_total=20,
            processed_opportunities=20,skipped_opportunities=0,unprocessed_opportunities=0,
            postabort=False,update_complete=True,attempt_complete=True,outer_commit_phase='STAGE_COMMITTED',accepted_complete_outer=True)for r in range(8)]
        progress.record(ranks,outer)
        path=run/f'attempt_{outer}.json'
        attempt=dict(driver_pid=123,source_commit=helper.SOURCE,attempt_id=outer,candidate_outer=outer,
            accepted_complete_outer=True,status='COMPLETE',rank_write_ledger=ranks,progress=progress.state_dict(),ledger_path=str(path))
        path.write_text(json.dumps(attempt));paths.append(path);attempts.append(attempt)
        events.append(dict(step=outer,attempt_id=outer,accepted_complete_outer=True,metrics={'actor/optimizer_steps':20,'bapo/actual_outer_updates':outer}))
    state=dict(driver_pid=123,global_step=5,next_global_step=6,actual_outer_updates=5,source_commit=helper.SOURCE,
        progress=progress.state_dict(),rank_write_ledger=attempts[-1]['rank_write_ledger'])
    driver_path=directory/'driver.pt';torch.save(state,driver_path);paths.append(driver_path)
    reload_path=directory/'driver-reload.json';reload_path.write_text(json.dumps(dict(status='EXACT_DRIVER_RELOAD_OK',step=5,
        path=str(driver_path),sha256=helper.digest(driver_path),state_digest=state_digest(state),pid=123)));paths.append(reload_path)
    reports=[]
    for rank in range(8):
        path=directory/'ranks'/f'rank_{rank}.pt';path.parent.mkdir(exist_ok=True);path.write_bytes(b'CPU_FIXTURE_ONLY_NOT_GPU_STATE')
        report=dict(rank=rank,path=str(path),sha256=helper.digest(path),status='EXACT_FULL_STATE_RELOAD_OK',
            native_state_version=3,roundtrip=True,step=5,world_size=8,scheduler_last_epoch=5,
            model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=1,optimizer_state_entries=1,
            aligned_pending_outer=None,state_digest='b'*64,reloaded_state_digest='b'*64,
            generation_rng_before_digest='c'*64,generation_rng_after_digest='c'*64,pid=1000+rank,
            latest_rank_event=attempts[-1]['rank_write_ledger'][rank])
        side=path.with_suffix('.json');side.write_text(json.dumps(report));paths.extend([path,side]);reports.append(report)
    manifest_path=directory/'manifest.json'
    manifest=dict(version=3,protocol='BAPO_BASELINE_ALIGNED_FRESH200_V1',baseline_aligned=True,source_commit=helper.SOURCE,
        run_root=str(run),global_step=5,actual_outer_updates=5,horizon=200,warmup_steps=57,mini_batch_switch_outer=None,
        update_semantics='COMPLETE20_NO_SKIP_ACCEPTED_OUTERS_V1',optimizer_opportunities_before_switch=20,
        optimizer_opportunities_after_switch=20,pause_nonce='d'*64,files=[helper.binding(p)for p in paths],
        config_sha256=helper.digest(config),attempt_accounting=progress.state_dict(),rank_roundtrips=reports)
    manifest_path.write_text(json.dumps(manifest))
    pause=dict(pid=123,manifest_path=str(manifest_path),manifest_sha256=helper.digest(manifest_path))
    text=''.join('FRESH200_ATTEMPT_JSON='+json.dumps(a)+'\nBAPO_TRAIN_METRICS_JSON='+json.dumps(e)+'\n'for a,e in zip(attempts,events))
    (run/'train.log').write_text(text+'FRESH200_STEP5_PAUSED_JSON='+json.dumps(pause)+'\n')
    calls=[]
    gate=SimpleNamespace(validate_gate_config=lambda c:(200,57),
        validate_first_five_attempts=fresh200_gate.validate_first_five_attempts,
        validate_aligned_first_five_evidence=lambda *args:calls.append(args))
    output=root/'out';output.mkdir()
    result=helper.first5_evidence(manifest_path,submission,output,gate,root=root,processes=False)
    assert result['attempt_accounting']['actual_adam_writes_per_rank']==[100]*8 and result['driver_pid']==123
    assert calls and calls[0][2]==helper.SOURCE and calls[0][4]==123
    assert result['native_roundtrip_role']=='ORIGINAL_GPU_PRODUCER_REPORTS_HASH_BOUND_NOT_CPU_RELOADED'
    assert (output/'original-step5-manifest.json').exists()and (output/'train-step5-prefix.log').exists()
    assert not list(output.glob('*release*'))
    # This fixture does not execute actual aligned EM/native-process verification;
    # it tests plumbing into that real production validator without issuing PASS.


@pytest.mark.parametrize('changed',['manifest.json','aligned_first5_cpu_evidence.py','aligned_first5_math.py','aligned-first5-math-provenance.json'])
def test_cpu_postwork_rehash_closes_manifest_and_helper_bytes(tmp_path,changed):
    names=['manifest.json','aligned_first5_cpu_evidence.py','aligned_first5_math.py','aligned-first5-math-provenance.json']
    for name in names:(tmp_path/name).write_text('original immutable bytes')
    records=[helper.binding(tmp_path/name)for name in names]
    helper.verify_original_bindings(records)
    (tmp_path/changed).write_text('changed during costly CPU work')
    with pytest.raises(ValueError,match='original inputs changed'):
        helper.verify_original_bindings(records)
    assert records[names.index(changed)]['sha256']!=helper.digest(tmp_path/changed)
