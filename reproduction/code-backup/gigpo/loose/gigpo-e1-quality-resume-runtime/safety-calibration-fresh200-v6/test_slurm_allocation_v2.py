"""UNEXECUTED schema/failure fixtures; no scheduler command or allocation created."""
import pytest
from slurm_allocation_v2 import validate_exclusive_allocation

JOB = ('JobId=999 JobState=RUNNING Requeue=0 Restarts=0 NodeList=h100-8 BatchHost=h100-8 '
       'NumNodes=1 OverSubscribe=NO NumCPUs=192 TresPerNode=gres:gpu:h100:8')
NODE = ('NodeName=h100-8 CPUTot=192 CPUAlloc=192 Gres=gpu:h100:8(S:0-1) '
        'CfgTRES=cpu=192,gres/gpu=8 GresUsed=gpu:h100:8(IDX:0-7)')
ENV = dict(SLURM_JOB_ID='999', SLURM_JOB_NODELIST='h100-8', SLURM_JOB_GPUS='0-7',
           CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')


def test_actual_effective_scheduler_allocation():
    result = validate_exclusive_allocation(JOB, NODE, '999|h100-8\n', ENV, 'h100-8')
    assert result['job_id'] == '999' and result['header_is_not_allocation_evidence']


@pytest.mark.parametrize('job,node,allocations', [
    (JOB.replace('OverSubscribe=NO', 'OverSubscribe=YES'), NODE, '999|h100-8'),
    (JOB.replace('OverSubscribe=NO', ''), NODE, '999|h100-8'),
    (JOB.replace('JobId=999', 'JobId=998'), NODE, '999|h100-8'),
    (JOB.replace('NumCPUs=192', 'NumCPUs=128'), NODE, '999|h100-8'),
    (JOB, NODE.replace('IDX:0-7', 'IDX:0-3'), '999|h100-8'),
    (JOB, NODE, '999|h100-8\n998|h100-8'),
    (JOB, NODE.replace('NodeName=h100-8', 'NodeName=h100-7'), '999|h100-8'),
])
def test_missing_ambiguous_or_shared_allocation_refused(job, node, allocations):
    with pytest.raises(ValueError):
        validate_exclusive_allocation(job, node, allocations, ENV, 'h100-8')
