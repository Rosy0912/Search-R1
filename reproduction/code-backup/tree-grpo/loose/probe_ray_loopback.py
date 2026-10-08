"""Actual target Ray CLI head + remote-worker loopback proof in own CPU Slurm job."""
import json, os, socket, sys, time
from pathlib import Path
from tree_shared_owned_runtime import ROOT, OwnedProcesses, cgroup_scope, available

job=os.environ['SLURM_JOB_ID']
assert job.isdigit() and os.environ['SLURM_JOB_NODELIST']=='h100-9'
assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
os.environ['RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER']='0'
os.environ['RAY_ADDRESS']='127.0.0.1:16382'
run=ROOT/'logs'/f'ray-loopback-{job}';run.mkdir()
alias=Path('/tmp')/f'sst-tree-loopback-{job}';assert not alias.exists();alias.mkdir()
scope,relative=cgroup_scope(job);owned=OwnedProcesses(scope,relative)
available(16382)
try:
    head=owned.spawn([str(ROOT/'env/bin/ray'),'start','--head','--node-ip-address=127.0.0.1','--port=16382','--num-cpus=2','--num-gpus=0','--include-dashboard=false','--object-store-memory=536870912','--temp-dir='+str(alias),'--min-worker-port=22082','--max-worker-port=23081','--disable-usage-stats','--block'],run/'head.log')
    deadline=time.monotonic()+90
    while True:
        assert head.poll() is None and time.monotonic()<deadline
        try:
            with socket.create_connection(('127.0.0.1',16382),timeout=1):break
        except OSError:time.sleep(1)
    import ray
    from omegaconf import OmegaConf
    from search_r1.baselines.local_runtime import check_local_runtime
    ray.init(address='127.0.0.1:16382',runtime_env={'env_vars':{'RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER':'0'}})
    nodes=ray.nodes()
    config=OmegaConf.create({'baselines':{'local_runtime_only':True,'local_runtime_host':'h100-9'},'retriever':{'url':'http://127.0.0.1:18082/retrieve'}})
    check_local_runtime(config,nodes)
    assert len(nodes)==1 and nodes[0]['NodeManagerAddress']=='127.0.0.1'
    @ray.remote
    def actual_worker():
        import os,socket,ray
        from ray._private.services import get_node_ip_address
        return {'pid':os.getpid(),'hostname':socket.gethostname(),'node_ip':get_node_ip_address(),'flag':os.environ.get('RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER'),'node_id':ray.get_runtime_context().get_node_id()}
    child=ray.get(actual_worker.remote())
    assert child['pid']!=os.getpid() and child['node_ip']=='127.0.0.1' and child['hostname']=='h100-9' and child['flag']=='0' and child['node_id']==nodes[0]['NodeID']
    result={'status':'ACTUAL_LOCAL_RAY_HEAD_WORKER_PASS','job_id':job,'ray_version':ray.__version__,'driver_pid':os.getpid(),'head_pid':head.pid,'worker':child,'node_manager_address':nodes[0]['NodeManagerAddress'],'cgroup':str(scope),'gpu_count':0,'local_boundary_pass':True}
    (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    try:
        if 'ray' in globals():ray.shutdown()
    finally:owned.cleanup()
