"""Exact scheduler reads shared by an explicit submit and one external publisher."""
import importlib.util
import json
from .contract import ALLOCATION


def cluster():
    s=importlib.util.spec_from_file_location('pt','/plm-shared/zhangjunming/Workspace/mmsearch-hybrid-explore/runs/msjev-pt-probe-20260924/pt_client.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.get_cluster()


def exact(client,name):
    assert len(name)<=32
    r=client.list_training_jobs(filter_str='name = '+json.dumps(name),page_size=100)
    assert isinstance(r,dict) and not(set(r)-{'training_jobs','total_size','next_page_token'})
    rows=r.get('training_jobs',[])
    assert not r.get('next_page_token') and r.get('total_size',0)==len(rows)
    assert all(j['name']==name for j in rows)
    return rows


def zero(client,name):
    assert not exact(client,name), 'Existing exact slot; inspect intent/output, never retry'
    try:client.get_training_job(name)
    except Exception as e:assert getattr(getattr(e,'response',None),'status_code',None)==404
    else:raise AssertionError('Existing exact allocation')


def actual(job):
    tasks=job['spec']['vc_job']['tasks'];assert len(tasks)==1 and tasks[0]['replicas']==1
    r=tasks[0]['resource_spec']
    resources=dict(gpus=r['accelerate_device_count'],cpus=r['cpu_count'],memory_gib=r['memory_gib'],
                   machine_type=r['machine_types'][0],gpu_model=r['accelerate_device_model'],gpu_memory_gib=r['accelerate_device_memory_gib'])
    assert resources==ALLOCATION
    return dict(name=job['name'],uid=job['uid'],state=job['status']['state'],resources=resources)
