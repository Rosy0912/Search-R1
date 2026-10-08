import json,sys,pathlib
import pandas as pd
import pytest
import assemble

def frame(n,source='nq',prefix='h'):
 return pd.DataFrame([dict(prompt=[dict(role='user',content=prefix+str(i))],data_source=source,extra_info=dict(index=i)) for i in range(n)])
def key(p):return json.dumps(p,sort_keys=True)

def test_actual_select300_preserves_train_and_disjoint_fixed_order():
 train=frame(8,prefix='t');h=frame(350);final=frame(3,prefix='f');val=frame(2,prefix='v')
 before=train.copy(deep=True);selected,keys=assemble.select_calibration(h,train,final,val,key)
 assert len(selected)==len(set(keys))==300 and keys==sorted(keys)
 pd.testing.assert_frame_equal(train,before)

@pytest.mark.parametrize('contaminate',['train','final','val'])
def test_real_question_contamination_rejected(contaminate):
 h=frame(350);others={n:frame(2,prefix=n) for n in ('train','final','val')};others[contaminate]=h.iloc[:1]
 with pytest.raises(ValueError,match='contamination'):assemble.select_calibration(h,others['train'],others['final'],others['val'],key)

def test_real_unique_question_count_rejected():
 h=frame(299);h=pd.concat([h,h.iloc[:10]],ignore_index=True)
 with pytest.raises(ValueError,match='300 unique'):assemble.select_calibration(h,frame(1,prefix='t'),frame(1,prefix='f'),frame(1,prefix='v'),key)

def test_actual_command_matches_bound_production_layout_and_original_math():
 c=dict(train_file='COMMON98_TRAIN',val_file='ORIGINAL64',base_model='OFFICIAL32BASE',retriever_url='http://127.0.0.1:18085/retrieve',checkpoint_root='OWN_CHECKPOINT',experiment='search_g1')
 d={x.lstrip('+').split('=',1)[0]:x.split('=',1)[1] for x in assemble.command(c)[3:]}
 assert d['data.train_batch_size']=='128' and d['actor_rollout_ref.rollout.n_agent']=='5'
 assert d['actor_rollout_ref.actor.ppo_micro_batch_size']=='64' and d['actor_rollout_ref.actor.ppo_mini_batch_size']=='256'
 assert d['actor_rollout_ref.rollout.tensor_model_parallel_size']=='8'
 assert d['actor_rollout_ref.actor.optim.lr']=='1e-6' and d['actor_rollout_ref.actor.optim.lr_warmup_steps_ratio']=='0.285'
 assert d['trainer.total_training_steps']=='101' and d['data.train_files']=='COMMON98_TRAIN'
 assert d['actor_rollout_ref.ref.log_prob_micro_batch_size']=='16'
 assert [len(range(i,min(i+256,640))) for i in range(0,640,256)]==[256,256,128]
 assert sum((rows//8+7)//8 for rows in [256,256,128])==10
