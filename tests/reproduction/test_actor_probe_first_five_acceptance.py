"""First-five orchestration with scalar files and RAM-only tensor stand-ins.

Native rank/calibration/checkpoint decoders are separately substituted; these
fixtures verify evidence selection/serialization, never production acceptance.
"""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from reproduction.common8.actor_probe_first_five_acceptance import verify_root_five
from search_g1.actor_probe_recovery import identity
from tests.reproduction.test_actor_probe_recovery import documents


def fixture(root):
    base=Path(root).resolve();run=documents()[0];run.update(initial_model=str(base/'official'),
        checkpoint_dir=str(base/'checkpoints'),initial_probe_version=run['run_id']+':0:'+'7'*32)
    records=[dict(step=s,run_id=run['run_id'],probe_version=run['initial_probe_version'],
        optimizer_writes_after=s*20,metrics={'critic/rewards/mean':.3}) for s in range(1,6)]
    gates={str(s):{'proof_sha256':str(s)*64} for s in range(1,6)}
    schedulers={str(rank):dict(selected_rows=[dict(logical_update=s,kind='COMPLETE',
        actual_optimizer_writes=20,opportunities_processed=20) for s in range(1,6)]) for rank in range(4)}
    # Actual writer returns tuples; JSON turns them into lists. Equality of
    # canonical serialized content must preserve all fields across this change.
    safety=dict(worker_process_births=[(1,2),(3,4)],selected_evidence_sha256='1'*64)
    readouts=dict(accepted=True,receipts=[{'feature_sha256':'2'*64}],retained_prefix_files=None)
    pair=dict(file_sha256={'0':'3'*64,'5':'4'*64})
    segment=dict(identity=identity(run),base=str(base),first=1,last=50,actual_interval_accepted=True,
        formal_inline_plan_sha256='5'*64,counter_segment={'retained_updates':records},
        behavior_probability_gates=gates,checkpoint_proofs={'5':dict(receipt={'fixture':'canonical'},finite_tensor_count=435)},
        full_states={'5':{'fixture':'full-state'}})
    entry=dict(run=run,base=str(base),origin=dict(completed_outer=0,actual_optimizer_writes=0,attempt_index=0,aborted_attempts=0))
    five=dict(accepted=True,run_id=run['run_id'],policy_safety_mode=run['policy_safety_mode'],
        formal_inline_plan_sha256='5'*64,actual_updates=records,behavior_probability_gates=gates,
        scheduler_receipts=schedulers,formal_inline_first_five=safety,actual_actor_readouts=readouts,
        paired_development=pair,checkpoint={'fixture':'canonical'},independently_verified_saved_tensor_count=435,
        changed_parameters=['a'],actual_first_five_gpu_occupancy={'fixture':'historical'})
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
    write(base/'logs/five-updates-acceptance.json',five)
    for directory in (base/'official',base/'checkpoints/actor/global_step_5'):
        write(directory/'model.safetensors.index.json',{'weight_map':{'a':'scalar-shard-stand-in'}})
    class Reader:
        def __init__(self,path,**kwargs):self.path=Path(path)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get_tensor(self,name):return torch.tensor([0 if self.path.parent==base/'official' else 1])
    return base,entry,segment,five,schedulers,safety,readouts,pair,Reader


class RootFirstFiveTests(unittest.TestCase):
    def test_json_birth_normalization_and_longer_feature_ledgers_keep_original_selected_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            base,entry,segment,five,schedulers,safety,readouts,pair,reader=fixture(root)
            prefix={'selected_updates':segment['counter_segment']['retained_updates']}
            with ExitStack() as stack:
                for name,value in [('load_prefix',prefix),('verify_scheduler_receipts',schedulers),
                    ('verify_first_five',safety),('verify_pair',pair),('verify_roundtrip_receipt',{'accepted':True}),
                    ('verify_historical_training_gpu_samples',{'accepted':True,'current_liveness_claimed':False})]:
                    stack.enter_context(patch('reproduction.common8.actor_probe_first_five_acceptance.'+name,return_value=value))
                stack.enter_context(patch('reproduction.common8.actor_probe_first_five_acceptance.verify_actor_readouts',
                    return_value=dict(readouts,retained_prefix_files=[{'fixture':'later stable ledger'}])))
                stack.enter_context(patch.dict('sys.modules',{'safetensors':SimpleNamespace(safe_open=reader)}))
                proof=verify_root_five(entry,segment)
                self.assertTrue(proof['accepted']);self.assertFalse(proof['current_liveness_claimed'])
                self.assertFalse(proof['full_training_or_final_acceptance'])
                changed=copy.deepcopy(safety);changed['selected_evidence_sha256']='9'*64
                with patch('reproduction.common8.actor_probe_first_five_acceptance.verify_first_five',return_value=changed):
                    with self.assertRaises(AssertionError):verify_root_five(entry,segment)


if __name__=='__main__':unittest.main()
