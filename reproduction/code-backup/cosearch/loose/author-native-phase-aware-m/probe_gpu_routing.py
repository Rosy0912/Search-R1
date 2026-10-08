"""Owned Ray GPU routing diagnostic. No model, loss, Adam, training, or acceptance."""
import json,os,time,sys,subprocess,hashlib
from pathlib import Path
import ray
from ray.util.placement_group import placement_group,remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def early_hook():
    import observe_worker
    observe_worker.write_record('early_hook',dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),rank=os.environ.get('RANK'),process_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),process_cgroup=Path('/proc/self/cgroup').read_text()))
    observe_worker.install()

@ray.remote
class Probe:
    def collective(self,address):
        import torch
        from datetime import timedelta
        torch.distributed.init_process_group('nccl',init_method=address,rank=int(os.environ['PROBE_BUNDLE_RANK']),world_size=8,timeout=timedelta(seconds=90))
        value=torch.tensor([1.],device='cuda:0')
        torch.distributed.all_reduce(value)
        result=float(value.cpu().item())
        torch.distributed.destroy_process_group()
        assert result==8.
        return result
    def inspect(self):
        import torch
        before_mask=os.environ.get('CUDA_VISIBLE_DEVICES')
        from verl.single_controller.base.worker import Worker
        original_worker=Worker.__new__(Worker)
        Worker.__init__(original_worker)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==before_mask
        props=torch.cuda.get_device_properties(0)
        return dict(worker_constructor_mask_before=before_mask,worker_constructor_mask_after=os.environ.get('CUDA_VISIBLE_DEVICES'),local_rank=os.environ.get('LOCAL_RANK'),original_worker_rank=original_worker.rank,pid=os.getpid(),rank=os.environ.get('RANK'),mask=os.environ.get('CUDA_VISIBLE_DEVICES'),
            assigned=ray.get_runtime_context().get_accelerator_ids(),uuid='GPU-'+str(props.uuid).removeprefix('GPU-'),raw_uuid=str(props.uuid),pci_name=props.name,
            device_count=torch.cuda.device_count(),current_device=torch.cuda.current_device(),
            startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),cgroup=Path('/proc/self/cgroup').read_text())

@ray.remote(num_cpus=1,num_gpus=0)
def cpu_probe():
    import torch
    return dict(pid=os.getpid(),mask=os.environ.get('CUDA_VISIBLE_DEVICES'),assigned=ray.get_runtime_context().get_accelerator_ids(),
        uuid=str(torch.cuda.get_device_properties(0).uuid),device_count=torch.cuda.device_count())

if __name__=='__main__':
    if len(sys.argv)==1:
        for mode in ('old_early','deferred'):
            subprocess.run([sys.executable,__file__,mode],env=os.environ.copy(),check=True)
        print('OWNED_RAY_ROUTING_PROBE_PASS synthetic0RL not_training_acceptance',flush=True)
        raise SystemExit(0)
    label=sys.argv[1]
    hook={'old_early':'probe_gpu_routing.early_hook','deferred':'observe_worker.defer_install'}[label]
    ray.init(address='127.0.0.1:16386',runtime_env={'worker_process_setup_hook':hook})
    results=[]
    for label,hook in [(label,hook)]:
        pg=placement_group([{'CPU':2,'GPU':1} for _ in range(8)],strategy='STRICT_PACK')
        ray.get(pg.ready())
        actors=[Probe.options(num_gpus=.5,num_cpus=1,scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,placement_group_bundle_index=i),
            runtime_env={'env_vars':{'RANK':str(i%4),'WORLD_SIZE':'4','PROBE_BUNDLE_RANK':str(i),'MASTER_ADDR':'10.33.0.16','MASTER_PORT':str(28000+i//4),'GLOO_SOCKET_IFNAME':'bond1','NCCL_SOCKET_IFNAME':'bond1'}}).remote() for i in range(8)]
        rows=ray.get([actor.inspect.remote() for actor in actors])
        value=dict(mode=label,rows=rows,distinct_uuids=len({row['uuid'] for row in rows}),real_rl_updates=0)
        if label=='deferred' and value['distinct_uuids']==8:
            import socket
            with socket.socket() as sock:
                sock.bind(('10.33.0.16',0));port=sock.getsockname()[1]
            value['eight_gpu_nccl_allreduce']=ray.get([actor.collective.remote('tcp://10.33.0.16:'+str(port)) for actor in actors])
        results.append(value)
        Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']).joinpath('routing-'+label+'.json').write_text(json.dumps(value,indent=2))
        print(json.dumps(value),flush=True)
        for actor in actors:ray.kill(actor)
        remove_placement_group(pg)
        time.sleep(3)
    files=list(Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']).joinpath('workers').glob('*.jsonl'))
    evidence=[json.loads(line) for p in files for line in p.read_text().splitlines()]
    expected_kind='early_hook' if label=='old_early' else 'device_assignment'
    assert {row['pid'] for row in results[-1]['rows']} <= {row['pid'] for row in evidence if row['kind']==expected_kind}, 'Actual job hook did not execute on all GPU actors'
    hook_sha=hashlib.sha256(Path(__file__).with_name('observe_worker.py').read_bytes()).hexdigest()
    for row in results[-1]['rows']:
        ack=[x for x in evidence if x['pid']==row['pid'] and x['kind']==expected_kind]
        assert len(ack)==1
        ack=ack[0]
        assert str(ack['job'])==os.environ['SLURM_JOB_ID'] and ack['hook_sha256']==hook_sha
        assert ack['source_commit']=='763bf8c4d7caa363ad845d39ddfd53b81ae377bd'
        assert ack['process_start_ticks']==row['startticks'] and ack['process_cgroup']==row['cgroup']
        assert 'job_'+os.environ['SLURM_JOB_ID'] in row['cgroup']
        if label=='deferred':
            assert ack['assigned_accelerator_ids']==row['assigned'] and ack['cuda_visible_devices']==row['mask']
    results[-1]['actual_hook_all8']=True
    if label=='deferred':
        cpu_rows=[ray.get(cpu_probe.remote()) for _ in range(8)]
        Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']).joinpath('routing-cpu-reuse.json').write_text(json.dumps(cpu_rows,indent=2))
        assert len({row['pid'] for row in cpu_rows})<len(cpu_rows), 'No actual CPU process reuse was observed'
        assert all(row['mask']==os.environ['COSEARCH_LEASED_UUIDS'] and row['device_count']==8 and not row['assigned'].get('GPU') for row in cpu_rows)
        assert results[-1]['distinct_uuids']==8
        assert set(row['uuid'] for row in results[-1]['rows'])==set(os.environ['COSEARCH_LEASED_UUIDS'].split(','))
    Path(os.environ['COSEARCH_NATIVE_EVIDENCE_ROOT']).joinpath('routing-'+label+'.json').write_text(json.dumps(results[-1],indent=2))
    print('ACTUAL_JOB_HOOK_MODE_PASS '+label+' synthetic0RL',flush=True)
    ray.shutdown()
