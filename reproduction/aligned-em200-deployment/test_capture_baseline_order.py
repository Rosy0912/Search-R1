"""CPU archive-loader fixtures, not actual common169028/model evidence."""
import hashlib
import importlib.util
from pathlib import Path
import json
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('capture_order',HERE/'capture_baseline_order.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Tokenizer:
    chat_template=None
    pad_token_id=0
    def __call__(self, prompt, **kwargs):
        ids=torch.arange(1,len(prompt.split())+1).reshape(1,-1)
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}


def holder(tmp_path, count, batch_size=512, long=False):
    rows=[{'prompt':[{'role':'user','content':'one two three four five' if long else 'one two'}],
        'extra_info':{'index':f'original-q-{i+700000}'},'data_source':'nq',
        'reward_model':{'ground_truth':{'target':['answer']}}}for i in range(count)]
    train=tmp_path/'train.parquet';dev=tmp_path/'dev.parquet'
    pd.DataFrame(rows).to_parquet(train);pd.DataFrame(rows[:1]).to_parquet(dev)
    config=OmegaConf.create(dict(data=dict(train_files=[str(train)],val_files=[str(dev)],prompt_key='prompt',
        max_prompt_length=4,train_batch_size=batch_size,val_batch_size=1,shuffle_train_dataloader=True,
        train_data_num=None,val_data_num=None,return_raw_chat=False),trainer=dict(total_epochs=1,total_training_steps=201),
        actor_rollout_ref=dict(actor=dict(optim={})),critic=dict(optim={})))
    built,_=module.reference_loader(HERE/'reference',config,Tokenizer())
    return built,[row['extra_info']['index']for row in rows]


def test_actual_archive_loader_full330_batches_consumer_schema_and_rng(tmp_path):
    from search_r1.baselines.baseline_alignment_data import verify_order_manifest
    built,ids=holder(tmp_path,169028)
    torch.manual_seed(42);initial=torch.get_rng_state().clone()
    result=module.observe_epoch(built.train_dataloader,ids,initial)
    assert len(result['actual_batches'])==330 and len(result['epochs'][0])==168960
    assert len(ids)-len(result['epochs'][0])==68
    assert result['preiterator_cpu_rng_state']==initial.tolist()
    assert result['initial_rng_sha256']==hashlib.sha256(bytes(initial.tolist())).hexdigest()
    assert result['actual_batches'][0]['question_ids']==[ids[i]for i in result['epochs'][0][:512]]
    # Native sampler's observed positions, original IDs, and real collation agree.
    assert built.train_dataset.observed_indices==result['epochs'][0]
    result['order_sha256']=module.object_hash(result)
    assert verify_order_manifest(result,ids,512)==result
    result['actual_batches'][0]['question_ids'][0]='forged'
    with pytest.raises(ValueError,match='modified'):verify_order_manifest(result,ids,512)


def test_actual_boundary_rng_replay_and_different_rng(tmp_path):
    built,ids=holder(tmp_path,20,4)
    torch.manual_seed(42);state=torch.get_rng_state().clone()
    first=module.observe_epoch(built.train_dataloader,ids,state,4)
    torch.rand(100)  # Caller RNG drift does not substitute for captured boundary.
    replay=module.observe_epoch(built.train_dataloader,ids,state,4)
    assert first['epochs']==replay['epochs']
    torch.manual_seed(43)
    changed=module.observe_epoch(built.train_dataloader,ids,torch.get_rng_state(),4)
    assert first['epochs']!=changed['epochs']


def test_actual_archive_truncation_error_not_silent_filter(tmp_path):
    built,ids=holder(tmp_path,8,4,long=True)
    assert len(built.train_dataset)==8 # archive filtering is commented out
    with pytest.raises(NotImplementedError,match='larger'):
        module.observe_epoch(built.train_dataloader,ids,torch.get_rng_state(),4)


def test_reference_tamper_and_immutable_output(tmp_path):
    import shutil
    reference=tmp_path/'reference';shutil.copytree(HERE/'reference',reference)
    (reference/'create_dataloader.py').write_text('def _create_dataloader(self): pass\n')
    with pytest.raises(ValueError,match='Modified'):
        module.reference_loader(reference,OmegaConf.create({}),Tokenizer())
    output=tmp_path/'order.json';module.write_exclusive(output,{'test':1})
    with pytest.raises(FileExistsError):module.write_exclusive(output,{'test':2})
    assert json.loads(output.read_text())=={'test':1}


def test_goal_guard_rejects_unpinned_manifest_before_dataset_or_model(tmp_path):
    from types import SimpleNamespace
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    with pytest.raises(ValueError,match='User-pinned'):
        module.capture(SimpleNamespace(aligned_root=str(Path.cwd()),common_manifest=str(manifest)))
