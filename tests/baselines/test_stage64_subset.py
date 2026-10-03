from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.holdout import prepare_common_holdout, select_stage_subset, question_key
from search_r1.baselines.controller import Controller
from search_r1.baselines.bapo import BoundaryStage


def frame(n=400):
    return pd.DataFrame([dict(prompt=[dict(content=f'q{i}')], extra_info={'index':i}, reward_model={'ground_truth':str(i)}) for i in range(n)])


def keys(f):
    return list(f.prompt.map(question_key))


def test_stage64_selection_unique_order_and_gold_independent():
    source=frame()
    source=pd.concat([source, source.iloc[:20]],ignore_index=True)
    selected=select_stage_subset(source,64)
    assert len(selected)==len(set(keys(selected)))==64
    shuffled=source.sample(frac=1,random_state=8).copy()
    shuffled['reward_model']=[{'ground_truth':'changed'}]*len(shuffled)
    assert keys(selected)==keys(select_stage_subset(shuffled,64))
    assert keys(selected)!=keys(select_stage_subset(source,64,seed=43))
    for size in (True,0,-1,1.2,401):
        with pytest.raises(ValueError):select_stage_subset(source,size)


def test_stage_subset_applied_after_common_split_and_grpo_preserved(tmp_path):
    source=frame(4000);source.to_parquet(tmp_path/'train.parquet',index=False)
    pd.DataFrame([dict(prompt=[dict(content='final')])]).to_parquet(tmp_path/'final.parquet',index=False)
    def cfg(method='bapo',eval_only=False):
        return OmegaConf.create(dict(data=dict(train_files=str(tmp_path/'train.parquet'),val_files=str(tmp_path/'final.parquet'),prompt_key='prompt'),baselines=dict(method=method,holdout_eval_only=eval_only,holdout=dict(enabled=True,fraction=.02,seed=42,output_dir=str(tmp_path/'split')),bapo=dict(stage_validation_files='/ignored',stage_validation_size=64))))
    c=cfg();manifest=prepare_common_holdout(c)
    subset_path=Path(c.baselines.bapo.stage_validation_files)
    selected=pd.read_parquet(subset_path)
    assert len(selected)==64 and manifest['holdout_rows']==80
    assert not set(keys(selected)) & set(keys(pd.read_parquet(c.data.train_files)))
    d=cfg();assert prepare_common_holdout(d)==manifest
    assert d.baselines.bapo.stage_validation_files==str(subset_path)
    for method,eval_only in [('grpo',False),('bapo',True)]:
        full=cfg(method,eval_only);assert prepare_common_holdout(full)==manifest
        assert len(pd.read_parquet(full.baselines.bapo.stage_validation_files))==80
        assert full.data.train_files==c.data.train_files
    # Even if an attacker updates the cached byte hash, semantic selection differs.
    selected.iloc[:63].to_parquet(subset_path,index=False)
    from search_r1.baselines.holdout import file_hash
    record=json.loads(subset_path.with_name('manifest.json').read_text())
    record['sha256']=file_hash(subset_path)
    subset_path.with_name('manifest.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='modified'):prepare_common_holdout(cfg())


def test_partial_stage_never_advances_and_restores_loader():
    c=Controller.__new__(Controller);c.method='bapo'
    c.config=SimpleNamespace(bapo=SimpleNamespace(stage_validation_freq=1,stage_validation_size=64))
    c.stage=BoundaryStage();c.stage_loader='stage';c.stage_dataset='stage'
    t=SimpleNamespace(val_reward_fn=None,tokenizer=None,val_dataloader='periodic',val_dataset='periodic')
    c.trainer=t
    for count,qa in [(63,63),(64,63),(0,64)]:
        t._validate=lambda:{'val/test_score/nq':.5,'val/evaluated_count/nq':count,'val/qa_count/all':qa}
        with pytest.raises(ValueError,match='complete'):c.stage_validation(1)
        assert c.stage.best == float('-inf') and c.stage.stalled==0
        assert t.val_dataloader==t.val_dataset=='periodic' and t.val_reward_fn is None
    t._validate=lambda:{'val/test_score/nq':.5,'val/evaluated_count/nq':64,'val/qa_count/all':64}
    assert c.stage_validation(1)['bapo/stage_evaluated_count']==64
