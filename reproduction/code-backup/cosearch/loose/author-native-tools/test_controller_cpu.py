"""Native CPU serialization/read-only gate checks, zero real RL/GPU acceptance."""
import hashlib, json, os, tempfile
from pathlib import Path
import torch, ray.cloudpickle
from unittest.mock import patch
from verl import DataProto
from launch_observed import ObservedPolicyGroup,validate_acks,reject_context_before_ppo

class Group:
    def update_actor(self,batch):
        self.received=batch
        return 'original-result'
    def compute_log_prob(self,*args,**kwargs):return (args,kwargs)
    def compute_ref_log_prob(self,*args,**kwargs):return (args,kwargs)

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']=temporary
    batch=DataProto.from_dict(tensors={
      'responses':torch.arange(8).reshape(8,1),
      'response_mask':torch.ones(8,1,dtype=torch.long),
      'token_level_rewards':torch.tensor([[1.],[0.],[1.],[0.],[0.],[0.],[0.],[0.]]),
      'advantages':torch.tensor([[1.],[-1.],[1.],[-1.],[0.],[0.],[0.],[0.]])},
      non_tensors={'uid':['group-a']*4+['group-b']*4})
    original={key:val.clone() for key,val in batch.batch.items()}
    rng=torch.get_rng_state().clone()
    group=Group();proxy=ObservedPolicyGroup(group,1,'reranker')
    restored=ray.cloudpickle.loads(ray.cloudpickle.dumps(proxy))
    assert restored.update_actor(batch)=='original-result'
    assert restored.group.received is batch
    assert all(torch.equal(value,batch.batch[key]) for key,value in original.items())
    assert torch.equal(rng,torch.get_rng_state())
    records=[json.loads(line) for path in root.glob('controller-*.jsonl') for line in path.read_text().splitlines()]
    assert records[0]['informative_reward_groups']==1
    assert records[0]['masked_nonzero_advantage_tokens']==4
    (root/'context-refusals').mkdir();(root/'context-refusals/oversized.json').write_text('{}')
    try:reject_context_before_ppo(1,False)
    except RuntimeError:pass
    else:raise AssertionError('Context marker must reject before dispatch')
    (root/'workers').mkdir()
    for index in range(8):
        row=dict(kind='ack',role='main' if index<4 else 'reranker',rank=index%4,
          world_size=4,uuid=f'synthetic-gpu-{index}',job='123',hostname='h100-16',
          process_cgroup='/slurm/uid_1/job_123/step_0',
          hook_sha256=hashlib.sha256(Path(__file__).with_name('observe_worker.py').read_bytes()).hexdigest(),
          source_manifest_sha256=hashlib.sha256(Path(__file__).with_name('source-manifest.json').read_bytes()).hexdigest(),
          original_actor_file_sha256=json.loads(Path(__file__).with_name('source-manifest.json').read_text())['sha256']['verl/verl/workers/actor/dp_actor.py'])
        (root/'workers'/f'{index}.jsonl').write_text(json.dumps(row)+'\n')
    os.environ['COSEARCH_LEASED_UUIDS']=','.join(f'synthetic-gpu-{index}' for index in range(8))
    with patch('launch_observed._verify_live_ack',return_value=None):
        assert len(validate_acks(root,'123'))==8
        (root/'workers/7.jsonl').unlink()
        try:validate_acks(root,'123')
        except AssertionError:pass
        else:raise AssertionError('Incomplete ACK must reject')
print(json.dumps(dict(synthetic_cpu_only=True,real_rl_updates=0,proxy_cloudpickle=True,
                     tensor_rng_unchanged=True,global_read_stats=True,context_marker_rejects=True,
                     incomplete_ack_rejects=True,actual_ray_cluster_test=False,passed=True)))
