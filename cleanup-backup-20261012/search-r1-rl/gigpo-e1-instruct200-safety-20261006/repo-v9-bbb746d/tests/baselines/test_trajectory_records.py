import gzip
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from search_r1.baselines.trajectory_records import RolloutTrace, atomic_rows, publish_trace_completion, record_scored_batch


class Batch(SimpleNamespace):
    def __len__(self): return self.batch['prompts'].shape[0]


class Tokenizer:
    def decode(self,tokens,**kwargs):return ' '.join(str(x) for x in tokens)


def test_all_responses_turns_masks_and_row_reordering(tmp_path):
    random.seed(3);np.random.seed(3);torch.manual_seed(3)
    before=(random.getstate(),np.random.get_state()[1].copy(),torch.get_rng_state().clone())
    trace=RolloutTrace(tmp_path,55,['q0']*5+['q1']*5,5)
    trace.record_turn(['<search>Q</search>']*10,['full wiki observation']*10,[True]*10,[False]*10,[True]*10,[True]*10)
    batch=Batch(batch={'prompts':torch.ones(10,4,dtype=torch.long),
        'responses':torch.ones(10,4,dtype=torch.long), 'info_mask':torch.tensor([[1,0,1,1]]*10),
        'advantages':torch.arange(10).float().unsqueeze(1).expand(10,4),
        'loss_mask':torch.ones(10,4), 'policy_weights':torch.ones(10,4)/4,
        'step_ids':torch.zeros(10,4,dtype=torch.long),'token_level_scores':torch.ones(10,4)},
        non_tensor_batch={'audit_trajectory_row':np.arange(9,-1,-1),'uid':np.array(['u']*10),'index':np.array(['q1']*5+['q0']*5)})
    snapshots={k:v.clone() for k,v in batch.batch.items()}
    trace.record_generation(batch,Tokenizer());record_scored_batch(tmp_path,55,batch)
    receipt=publish_trace_completion(tmp_path,55,10,'ACTUAL_OUTER_UPDATE_COMPLETE',10)
    assert receipt['actual_optimizer_steps']==10 and len(receipt['files'])==3
    assert all(torch.equal(v,batch.batch[k]) for k,v in snapshots.items())
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1])
    assert torch.equal(torch.get_rng_state(),before[2])
    with gzip.open(trace.root/'scored.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    assert rows[0]['trajectory_row']==9 and rows[0]['response_index']==4 and rows[0]['response_group']==1
    with pytest.raises(ValueError):publish_trace_completion(tmp_path,55,10,'ACTUAL_OUTER_UPDATE_COMPLETE',10)


def test_partial_or_missing_evidence_never_publishes_completion(tmp_path):
    p=tmp_path/'iter_000055';p.mkdir()
    atomic_rows(p/'scored.jsonl.gz',[{'row':0}])
    with pytest.raises(ValueError):publish_trace_completion(tmp_path,55,1,'ACTUAL_OUTER_UPDATE_COMPLETE',10)
    assert not (p/'complete.json').exists()
    atomic_rows(p/'generated.jsonl.gz',[{'row':0}]);atomic_rows(p/'turn_01.jsonl.gz',[{'row':0}])
    (p/'bad.partial').write_text('incomplete')
    with pytest.raises(ValueError):publish_trace_completion(tmp_path,55,1,'ACTUAL_OUTER_UPDATE_COMPLETE',10)
    assert not (p/'complete.json').exists()


def test_refuses_partial_replacement_and_incomplete_response_groups(tmp_path):
    p=tmp_path/'row.jsonl.gz';p.with_name(p.name+'.partial').write_text('prior failed write')
    with pytest.raises(FileExistsError):atomic_rows(p,[{'row':0}])
    with pytest.raises(ValueError):RolloutTrace(tmp_path,1,['q']*4,5)
    with pytest.raises(ValueError):RolloutTrace(tmp_path,1,['q']*4+['other'],5)
