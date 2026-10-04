import os
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.local_runtime import check_local_runtime


@pytest.fixture
def local(monkeypatch):
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('SLURM_JOB_NODELIST','h100-8')
    monkeypatch.setenv('RAY_ADDRESS','127.0.0.1:16381')
    monkeypatch.setattr('socket.gethostname',lambda:'h100-8')
    monkeypatch.setattr('socket.getaddrinfo',lambda *a:[(2,1,6,'',('10.33.0.8',0))])
    return OmegaConf.create({'baselines':{'local_runtime_only':True,'local_runtime_host':'h100-8'},
        'retriever':{'url':'http://127.0.0.1:18081/retrieve'}})


def test_local_host_and_ray_are_accepted(local):
    check_local_runtime(local,[{'Alive':True,'NodeManagerAddress':'10.33.0.8'}])


@pytest.mark.parametrize('url',['http://a800:18081/retrieve','http://h100-16:18081/retrieve',
    'https://127.0.0.1:18081/retrieve','http://127.0.0.1:18081/remote'])
def test_online_foreign_retrieval_is_rejected(local,url):
    local.retriever.url=url
    with pytest.raises(ValueError,match='Retriever'):check_local_runtime(local)


def test_foreign_ray_node_is_rejected(local):
    with pytest.raises(ValueError,match='Ray contains'):
        check_local_runtime(local,[{'Alive':True,'NodeManagerAddress':'10.33.0.16'}])


def test_proxy_or_wrong_allocation_is_rejected(local,monkeypatch):
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:18971')
    with pytest.raises(ValueError,match='proxy'):check_local_runtime(local)
    monkeypatch.delenv('HTTPS_PROXY')
    monkeypatch.setenv('SLURM_JOB_NODELIST','h100-16')
    with pytest.raises(ValueError,match='Slurm'):check_local_runtime(local)


@pytest.fixture
def opsd_local(local, monkeypatch):
    local.baselines.method = 'opsd'
    local.baselines.distillation = {'teacher_device': 'cuda'}
    local.baselines.local_runtime_host = 'h100-10'
    monkeypatch.setenv('SEARCH_R1_OPSD_LOCAL_HOST', 'h100-10')
    monkeypatch.setenv('SLURM_JOB_NODELIST', 'h100-10')
    monkeypatch.setattr('socket.gethostname', lambda: 'h100-10')
    monkeypatch.setattr('socket.getaddrinfo', lambda *a: [(2,1,6,'',('10.33.0.10',0))])
    return local


def test_opsd_task_policy_accepts_local_cuda(opsd_local):
    check_local_runtime(opsd_local, [{'Alive': True, 'NodeManagerAddress': '10.33.0.10'}])


@pytest.mark.parametrize('field,value', [
    ('method', 'grpo'), ('local_runtime_only', False),
    ('local_runtime_host', 'h100-16'), ('teacher_device', 'cpu')])
def test_opsd_task_policy_cannot_be_disabled_by_overrides(opsd_local, field, value):
    if field == 'teacher_device':
        opsd_local.baselines.distillation.teacher_device = value
    else:
        opsd_local.baselines[field] = value
    with pytest.raises(ValueError, match='OPSD task runtime'):
        check_local_runtime(opsd_local)
