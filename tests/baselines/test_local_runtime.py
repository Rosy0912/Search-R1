import os
import socket
from types import SimpleNamespace
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
    monkeypatch.setattr('psutil.net_if_addrs',lambda:{'bond1':[SimpleNamespace(family=socket.AF_INET,address='10.33.0.8')]})
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


def test_secondary_local_interface_is_accepted_but_not_second_node(local,monkeypatch):
    monkeypatch.setattr('psutil.net_if_addrs',lambda:{'ib0':[SimpleNamespace(family=socket.AF_INET,address='10.32.0.8')]})
    check_local_runtime(local,[{'Alive':True,'NodeManagerAddress':'10.32.0.8'}])
    with pytest.raises(ValueError,match='Ray contains'):
        check_local_runtime(local,[{'Alive':True,'NodeManagerAddress':'10.32.0.8'},
                                  {'Alive':True,'NodeManagerAddress':'10.33.0.9'}])
