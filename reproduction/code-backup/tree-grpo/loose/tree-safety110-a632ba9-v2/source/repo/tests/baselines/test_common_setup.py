"""Shared deployment regression tests."""
from copy import deepcopy
import pandas as pd
import pytest
from search_r1.baselines.holdout import split_frames, content_identity
from search_r1.baselines.select_teacher import select, CANDIDATES
from search_r1.baselines.evaluation import aggregate_answer_metrics


def test_content_identity_ignores_paths_detects_rows():
    frame=pd.DataFrame([dict(prompt=[dict(role='user',content=str(i))],extra_info=dict(index=i)) for i in range(100)])
    a,b=split_frames(frame,frame.iloc[:3])
    spec=dict(version=1,train=[dict(path='/a/train',sha256='source')],final=[dict(path='/a/test',sha256='test')],fraction=.02,seed=42,prompt_key='prompt')
    other=deepcopy(spec);other['train'][0]['path']='/b/train';other['final'][0]['path']='/b/test'
    assert content_identity(spec,a,b)==content_identity(other,a,b)
    altered=b.copy(deep=True);altered.at[0,'extra_info']={'index':999}
    assert content_identity(spec,a,b)!=content_identity(other,a,altered)


def records():
    return [dict(outer_update=s,split_content_identity='abc',evaluation_split='common-holdout',seed=42,evaluated_count=100,holdout_rows=100,mean_answer_f1=.5) for s in CANDIDATES]


def test_selector_requires_complete_same_holdout_prefers_earliest_tie():
    rs=records();assert select(rs,dict(content_identity=dict(sha256='abc'),holdout_rows=100))['outer_update']==5
    rs[3]['mean_answer_f1']=.6;assert select(rs,dict(content_identity=dict(sha256='abc'),holdout_rows=100))['outer_update']==300
    for key,value in [('evaluation_split','final-test'),('split_content_identity','other'),('evaluated_count',99),('mean_answer_f1',float('nan')),('seed',123)]:
        bad=deepcopy(rs);bad[0][key]=value
        with pytest.raises(ValueError):select(bad,dict(content_identity=dict(sha256='abc'),holdout_rows=100))
    with pytest.raises(ValueError):select(rs[:-1],dict(content_identity=dict(sha256='abc'),holdout_rows=100))


def test_global_f1_is_sample_weighted():
    rows=[dict(source=s,f1=f,em=f,answer_format=1.,format_rate=1.,generated_tokens=2,search_calls=1) for s,f in [('nq',1.),('hotpotqa',0.),('hotpotqa',0.)]]
    result=aggregate_answer_metrics(rows)
    assert result['val/f1/all']==pytest.approx(1/3)
    assert result['val/qa_count/all']==3


def test_smoke_skips_final_set_saves_checkpoint():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduction'))
    from test_final_checkpoint import real_iteration_tail, FakeTrainer
    t=FakeTrainer(2,3);t.config.trainer.smoke_only=True
    real_iteration_tail()(t)
    assert t.events==[('update',2,1),('save',2,1)]


def test_hf_index_without_weights_refused(tmp_path):
    import json
    from search_r1.baselines.select_teacher import model_manifest
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'tokenizer_config.json').write_text('{}')
    (tmp_path/'pytorch_model.bin.index.json').write_text(json.dumps(dict(weight_map={'x':'missing.bin'})))
    with pytest.raises(ValueError):model_manifest(tmp_path)
    (tmp_path/'pytorch_model-00001.bin').write_bytes(b'weight')
    with pytest.raises(ValueError,match='shard'):model_manifest(tmp_path)
