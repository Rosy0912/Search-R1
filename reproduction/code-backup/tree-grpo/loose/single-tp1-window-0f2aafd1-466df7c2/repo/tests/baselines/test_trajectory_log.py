import gzip,json
import numpy as np
import pytest
import torch
from verl import DataProto
from search_r1.baselines.trajectory_log import save_rollouts
from search_r1.baselines.training_state import file_sha


class Tokenizer:
    def decode(self,tokens,skip_special_tokens=False):return ' '.join(map(str,tokens))


def example():
    t={k:torch.ones(6,4) for k in ('input_ids','responses','attention_mask','info_mask',
       'old_log_probs','ref_log_prob','advantages','policy_weights','token_level_rewards','loss_mask')}
    t['input_ids']=torch.arange(24).reshape(6,4)
    t['old_log_probs'][2,3]=-123.5
    a=np.empty(6,dtype=object)
    a[:]=[[dict(text='search',observation_text='original tool result',action_token_ids=[1],observation_token_ids=[2])] for _ in range(6)]
    return DataProto.from_dict(tensors=t,non_tensors={'trajectory_uid':np.array(list('abcdef'),dtype=object),'action_records':a})


def test_all_leaves_tool_observations_and_loss_tensors_survive(tmp_path):
    b=example();r=save_rollouts(b,Tokenizer(),101,tmp_path)
    d=tmp_path/'outer-000101'
    assert r['rows']==6
    for name,sha in r['payloads'].items():assert file_sha(d/name)==sha
    with gzip.open(d/'trajectories.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    assert len(rows)==6 and [v['metadata']['trajectory_uid'] for v in rows]==list('abcdef')
    assert rows[5]['token_ids']==[20,21,22,23]
    assert rows[0]['metadata']['action_records'][0]['observation_text']=='original tool result'
    p=torch.load(d/'actor-inputs.pt',weights_only=False)
    for k in b.batch.keys():assert torch.equal(p['tensors'][k],b.batch[k])
    before=file_sha(d/'actor-inputs.pt')
    with pytest.raises(FileExistsError):save_rollouts(b,Tokenizer(),101,tmp_path)
    assert file_sha(d/'actor-inputs.pt')==before


def test_missing_loss_inputs_cannot_publish_success_manifest(tmp_path):
    b=example();del b.batch['old_log_probs']
    with pytest.raises(ValueError,match='Incomplete'):save_rollouts(b,Tokenizer(),101,tmp_path)
    assert not (tmp_path/'outer-000101/manifest.json').exists()
