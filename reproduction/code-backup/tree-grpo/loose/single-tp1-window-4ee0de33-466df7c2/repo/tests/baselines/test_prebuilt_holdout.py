"""Real parquet/DataProto fixtures; no mocked split selection, GPU or network."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.holdout import (E1_INSTRUCTION, E1_SOURCES, E1_FINAL_COUNTS,
    frame_identity, ordered_question_ids_hash, file_hash, _logical_path, _prebuilt_frame,
    prepare_common_holdout, assert_prebuilt_unchanged, prebuilt_overlap_audit, canonical_question_key, canonical_questions_hash)


def frame(source='nq', count=2, first=0, split='train', label='train'):
    return pd.DataFrame([dict(data_source=source,
        prompt=[dict(role='user',content=E1_INSTRUCTION+'\nQuestion: '+source+'-'+split+'-'+str(i))],
        ability='fact-reasoning',reward_model=dict(style='rule',ground_truth={'target':['answer-'+str(i)]}),
        extra_info=dict(index=source+':'+split+':'+str(i),split=label)) for i in range(first,first+count)])


def record(path, data):
    data.to_parquet(path,index=False)
    loaded=pd.read_parquet(path)
    return dict(path=path.name,sha256=file_hash(path),rows=len(loaded),
        source_counts={str(k):int(v) for k,v in loaded.data_source.value_counts().items()},
        question_ids_sha256=ordered_question_ids_hash(loaded),content_identity=frame_identity(loaded))


def test_prebuilt_frame_actual_ids_content_and_protocol(tmp_path):
    path=tmp_path/'small.parquet';data=frame();r=record(path,data)
    loaded,ids,questions=_prebuilt_frame(path,r,'prompt','train')
    assert len(loaded)==len(ids)==len(questions)==2
    for field,value in [('rows',1),('question_ids_sha256','a'*64),('content_identity',{}),('source_counts',{'nq':1})]:
        bad=copy.deepcopy(r);bad[field]=value
        with pytest.raises(ValueError):_prebuilt_frame(path,bad,'prompt','train')
    for kind in ('id_collision','wrong_prompt','old_idk_prompt','wrong_source','wrong_target'):
        changed=data.copy(deep=True)
        if kind=='id_collision':changed.at[1,'extra_info']={'index':'nq:train:0'}
        if kind in ('wrong_prompt','old_idk_prompt'):
            changed.at[0,'prompt']=[dict(role='user',content='Question: plain' if kind=='wrong_prompt' else E1_INSTRUCTION+'\n<answer>I don\'t know</answer>\nQuestion: q')]
        if kind=='wrong_source':changed.at[0,'data_source']='hotpotqa'
        if kind=='wrong_target':changed.at[0,'reward_model']={'style':'rule','ground_truth':{'target':[]}}
        bad=record(path,changed)
        with pytest.raises(ValueError):_prebuilt_frame(path,bad,'prompt','train')


def test_logical_paths_refuse_escape_and_external_symlink(tmp_path):
    (tmp_path/'ok').write_text('data')
    assert _logical_path(tmp_path,'ok')==tmp_path/'ok'
    for p in ('/etc/passwd','../ok','missing','bad\\name'):
        with pytest.raises(ValueError):_logical_path(tmp_path,p)
    outside=tmp_path.parent/(tmp_path.name+'-outside');outside.write_text('outside')
    (tmp_path/'link').symlink_to(outside)
    with pytest.raises(ValueError):_logical_path(tmp_path,'link')


def test_real_dataset_to_dataproto_preserves_prebuilt_ids(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    backend=Tokenizer(models.WordLevel({'<unk>':0,'<pad>':1,'Answer':2},unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>',pad_token='<pad>')
    tokenizer.chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
    path=tmp_path/'fixture.parquet';data=frame();r=record(path,data)
    dataset=RLHFDataset(str(path),tokenizer,max_prompt_length=256,cache_dir=str(tmp_path/'cache'))
    batch=DataProto.from_single_dict(collate_fn([dataset[0],dataset[1]]))
    assert len(batch)==2 and batch.batch['input_ids'].shape==(2,256)
    assert [x['index'] for x in batch.non_tensor_batch['extra_info']]==['nq:train:0','nq:train:1']
    assert batch.non_tensor_batch['reward_model'][0]['ground_truth']['target'][0]=='answer-0'
    assert file_hash(path)==r['sha256']


def full_fixture(tmp_path):
    train=pd.concat([frame('nq',78880,288),frame('hotpotqa',90159,288)],ignore_index=True)
    train=train.iloc[1:].reset_index(drop=True)  # Real retained row reduction after the approved exclusions.
    validation=pd.concat([frame('nq',256,label='validation'),frame('hotpotqa',256,label='validation')],ignore_index=True)
    stage=pd.concat([frame('nq',32,256,label='stage'),frame('hotpotqa',32,256,label='stage')],ignore_index=True)
    final=pd.concat([frame(s,n,split='test' if s in ('nq','triviaqa','popqa','bamboogle') else 'dev',label='test') for s,n in E1_FINAL_COUNTS.items()],ignore_index=True)
    records={name:record(tmp_path/(name+'.parquet'),data) for name,data in [('train',train),('validation',validation),('stage',stage),('final',final)]}
    raw=[]
    for s,split in [('nq','train'),('hotpotqa','train')]+[(s,'test' if s in ('nq','triviaqa','popqa','bamboogle') else 'dev') for s in E1_SOURCES]:
        p=tmp_path/(s+'-'+split+'.raw');p.write_text('raw-source-fixture:'+s+':'+split)
        raw.append(dict(source=s,split=split,path=p.name,sha256=file_hash(p),rows=({'nq':79168,'hotpotqa':90447}[s] if split=='train' else E1_FINAL_COUNTS[s])))
    manifest=dict(version=1,protocol='search-e1-common-300',seed=42,prompt_key='prompt',
        preparation=dict(dataset_id='RUC-NLPIR/FlashRAG_datasets',dataset_revision='fixture-version',prepare_sha256='a'*64,core_sha256='b'*64,
            selection='source-order-seed42-first256-dev-next32-stage',source_order=list(E1_SOURCES),raw_sources=raw),splits=records)
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    config=OmegaConf.create(dict(data=dict(prompt_key='prompt',train_files='/ignored',val_files='/ignored',final_val_files=None),
        baselines=dict(method='bapo',holdout_eval_only=False,holdout=dict(enabled=True,input_mode='prebuilt_manifest',seed=42,
            manifest_path=str(mp),manifest_sha256=file_hash(mp),data_root=str(tmp_path)),bapo=dict(stage_validation_size=64,stage_validation_files=None))))
    return config,manifest


def test_full_prebuilt_immutable_approved_counts_and_runtime_paths(tmp_path,monkeypatch):
    # Production-sized real parquet fixtures: fixed512/64/51713 plus actual retained169038, no size mocks.
    cfg,manifest=full_fixture(tmp_path)
    before={p.name:file_hash(p) for p in tmp_path.iterdir() if p.is_file()}
    from search_r1.baselines import holdout
    def forbidden(*args,**kwargs):raise AssertionError('Prebuilt mode must not split/sample')
    monkeypatch.setattr(holdout,'split_frames',forbidden);monkeypatch.setattr(holdout,'select_stage_subset',forbidden)
    result=prepare_common_holdout(cfg)
    assert result['train_rows']==169038 and result['holdout_rows']==512
    assert all(x['ids']==x['questions']==0 for x in result['overlaps'].values())
    assert_prebuilt_unchanged(cfg,result)
    after={p.name:file_hash(p) for p in tmp_path.iterdir() if p.is_file()}
    assert before==after and not any(p.is_dir() for p in tmp_path.iterdir())
    cfg.data.train_files=str(tmp_path/'validation.parquet')
    with pytest.raises(ValueError,match='paths changed'):assert_prebuilt_unchanged(cfg,result)
    cfg.data.train_files=result['effective_paths']['train']
    mp=Path(cfg.baselines.holdout.manifest_path);mp.write_text(mp.read_text()+'\n')
    with pytest.raises(ValueError,match='byte hash'):assert_prebuilt_unchanged(cfg,result)


def test_unknown_mode_and_invalid_manifest_before_any_output(tmp_path):
    cfg=OmegaConf.create({'baselines':{'holdout':{'input_mode':'prebuilt_typo'}}})
    with pytest.raises(ValueError,match='Unknown'):prepare_common_holdout(cfg)


def test_real_prompt_leakage_across_distinct_source_ids_is_rejected(tmp_path):
    first=frame(first=1,count=1);other=frame(first=2,count=1)
    other.at[0,'prompt']=first.iloc[0]['prompt']
    records={}
    for name,data in [('train',first),('stage',other)]:
        path=tmp_path/(name+'.parquet');r=record(path,data)
        _,ids,questions=_prebuilt_frame(path,r,'prompt',name)
        records[name]=(ids,questions)
    assert not records['train'][0]&records['stage'][0]
    with pytest.raises(ValueError,match='no automatic resplit'):
        prebuilt_overlap_audit({k:v[0] for k,v in records.items()},{k:v[1] for k,v in records.items()})


def test_prebuilt_subsample_refused_before_reading_data(tmp_path):
    cfg=OmegaConf.create(dict(data=dict(train_data_num=1,val_data_num=None),baselines=dict(holdout=dict(enabled=True,input_mode='prebuilt_manifest'),bapo=dict(stage_validation_size=64))))
    with pytest.raises(ValueError,match='cannot be subsampled'):prepare_common_holdout(cfg)


def test_canonical_case_space_and_trailing_question_overlap(tmp_path):
    first=frame(first=1,count=1);other=frame(first=2,count=1)
    first.at[0,'prompt']=[{'role':'user','content':E1_INSTRUCTION+'\nQuestion: Paris CAPITAL?'}]
    other.at[0,'prompt']=[{'role':'user','content':E1_INSTRUCTION+'\nQuestion: paris   capital'}]
    assert canonical_question_key(first.iloc[0]['prompt'])=='paris capital'
    assert canonical_questions_hash(first)==canonical_questions_hash(other)
    records={}
    for name,data in [('train',first),('stage',other)]:
        path=tmp_path/(name+'.parquet');r=record(path,data)
        _,ids,questions=_prebuilt_frame(path,r,'prompt',name);records[name]=(ids,questions)
    with pytest.raises(ValueError,match='no automatic resplit'):
        prebuilt_overlap_audit({k:v[0] for k,v in records.items()},{k:v[1] for k,v in records.items()})


def test_manifest_schema_negative_cases_before_any_data_read(tmp_path):
    base=dict(version=1,protocol='search-e1-common-300',seed=42,prompt_key='prompt',preparation=dict(
        dataset_id='RUC-NLPIR/FlashRAG_datasets',dataset_revision='fixture-revision',prepare_sha256='a'*64,core_sha256='b'*64,
        selection='source-order-seed42-first256-dev-next32-stage',source_order=list(E1_SOURCES),raw_sources=[]))
    mp=tmp_path/'manifest.json'
    def cfg(manifest):
        mp.write_text(json.dumps(manifest))
        return OmegaConf.create(dict(data=dict(prompt_key='prompt',train_data_num=None,val_data_num=None),
            baselines=dict(holdout=dict(enabled=True,input_mode='prebuilt_manifest',seed=42,manifest_path=str(mp),
                manifest_sha256=file_hash(mp),data_root=str(tmp_path)),bapo=dict(stage_validation_size=64))))
    for field,value in [('version',True),('version',2),('protocol','old-protocol'),('seed',43),('prompt_key','different')]:
        bad=copy.deepcopy(base);bad[field]=value
        with pytest.raises(ValueError,match='protocol/seed/prompt'):prepare_common_holdout(cfg(bad))
    for field,value in [('dataset_revision',''),('dataset_id','other'),('source_order',list(reversed(E1_SOURCES))),('selection','random-resplit')]:
        bad=copy.deepcopy(base);bad['preparation'][field]=value
        with pytest.raises(ValueError,match='provenance/selection'):prepare_common_holdout(cfg(bad))
    for field in ('prepare_sha256','core_sha256'):
        bad=copy.deepcopy(base);bad['preparation'][field]='not-a-sha'
        with pytest.raises(ValueError,match='SHA256'):prepare_common_holdout(cfg(bad))
    with pytest.raises(ValueError,match='two train and seven final'):prepare_common_holdout(cfg(base))
    assert {p.name for p in tmp_path.iterdir()}=={'manifest.json'}
