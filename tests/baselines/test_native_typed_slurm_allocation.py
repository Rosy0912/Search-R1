"""Slurm text parsing only; fixture text is not a native GPU proof."""
import pytest
from search_r1.baselines.fresh200_native_evidence import allocation_has_eight_gpus

@pytest.mark.parametrize('resource',[
    'TRES=cpu=32,gres/gpu=8',
    'TRES=cpu=32,gres/gpu:h100=8',
    'TRES=gres/gpu=8,gres/gpu:h100=8',
    'AllocTRES=gres/gpu=8',
    'TresPerNode=gres:gpu:8',
    'TresPerNode=gres:gpu:h100:8',
    'TRES=cpu=32,mem=512G,node=1,billing=32 Comment=BAPO:formal StdErr=/own/error Power= TresPerNode=gres:gpu:h100:8',
])
def test_exact_eight_standard_slurm_fields_accepted(resource):
    assert allocation_has_eight_gpus('JobId=3356 JobState=RUNNING '+resource+'\n')

@pytest.mark.parametrize('resource',[
    'TresPerNode=gres:gpu:h100:7','TresPerNode=gres:gpu:h100:80','TRES=gres/gpu=80',
    'TRES=gres/gpu:h100=7','TRES=gres/gpu:h100=80','TresPerNode=gres:gpu:h100:8junk',
    'Comment=gres/gpu=8','Comment=gres/gpu:h100:8','Comment="TresPerNode=gres:gpu:h100:8"',
    'Comment=free text TresPerNode=gres:gpu:h100:8',
    'ReqTRES=gres/gpu=8','Other=gres/gpu=8','TresPerNode=gres:gpu:h100:7 Comment=gres/gpu:h100:8',
    'TRES=gres/gpu=7 TresPerNode=gres:gpu:h100:8','TRES=gres/gpu=8 TRES=gres/gpu=7',
])
def test_wrong_count_or_free_text_cannot_supply_allocation(resource):
    assert not allocation_has_eight_gpus('JobId=3356 JobState=RUNNING '+resource+'\n')
