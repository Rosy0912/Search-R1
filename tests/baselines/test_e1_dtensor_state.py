"""Real CPU DTensor serialization; no claim of native CUDA/FSDP execution."""
from pathlib import Path
import types
import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import DTensor, Replicate, Shard
from search_r1.baselines.full_state import state_digest, atomic_save, rng_state

@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    tmp_path=tmp_path_factory.mktemp("dtensor-gloo")
    assert not dist.is_initialized()
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'gloo'),rank=0,world_size=1)
    try:
        yield (DeviceMesh('cpu',[0],mesh_dim_names=('fsdp',)),
               DeviceMesh('cpu',[0],mesh_dim_names=('different',)))
    finally:dist.destroy_process_group()

def make(mesh, placements=None, local=None):
    return DTensor.from_local(torch.arange(4,dtype=torch.float32) if local is None else local,
        mesh, [Shard(0)] if placements is None else placements,run_check=False)

def test_real_cpu_dtensor_disk_roundtrip_and_rng(meshes,tmp_path):
    dt=make(meshes[0]);before=rng_state();expected=state_digest(dt)
    path=tmp_path/'dt.pt';atomic_save(dict(model={'weight':dt}),path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)['model']['weight']
    assert isinstance(loaded,DTensor) and loaded.device.type=='cpu'
    assert loaded.device_mesh.mesh_dim_names==('fsdp',)
    assert state_digest(loaded)==expected
    assert state_digest(rng_state())==state_digest(before)

@pytest.mark.parametrize('change',['mesh-names','placements','dtype','shape','stride','local-bytes','mesh-ranks','mesh-device','coordinate'])
def test_digest_layout_change_cannot_pass(meshes,change):
    dt=make(meshes[0]);expected=state_digest(dt)
    if change=='mesh-names':altered=make(meshes[1])
    elif change=='placements':altered=make(meshes[0],[Replicate()])
    elif change=='dtype':altered=make(meshes[0],local=torch.arange(4,dtype=torch.float64))
    elif change=='shape':altered=make(meshes[0],local=torch.arange(4,dtype=torch.float32).reshape(2,2))
    elif change=='local-bytes':altered=make(meshes[0],local=torch.arange(4,dtype=torch.float32)+1)
    else:
        # Metadata counterexamples without manufacturing invalid process groups.
        mesh=types.SimpleNamespace(device_type=dt.device_mesh.device_type,
            mesh=dt.device_mesh.mesh.clone(),mesh_dim_names=dt.device_mesh.mesh_dim_names,
            get_coordinate=lambda:[0])
        stride=tuple(dt.stride())
        if change=='stride':stride=(2,)
        elif change=='mesh-ranks':mesh.mesh=torch.tensor([9])
        elif change=='mesh-device':mesh.device_type='cuda'
        elif change=='coordinate':mesh.get_coordinate=lambda:[1]
        altered=types.SimpleNamespace(shape=dt.shape,dtype=dt.dtype,placements=dt.placements,
            stride=lambda:stride,device_mesh=mesh,to_local=lambda:dt.to_local())
    assert state_digest(altered)!=expected

def test_digest_uses_no_collective_and_no_full_tensor(meshes,monkeypatch):
    dt=make(meshes[0]);expected=state_digest(dt)
    def forbidden(*args,**kwargs):raise AssertionError('digest initiated collective/full tensor')
    monkeypatch.setattr(DTensor,'full_tensor',forbidden)
    monkeypatch.setattr(DTensor,'redistribute',forbidden)
    monkeypatch.setattr(dist,'all_gather',forbidden)
    monkeypatch.setattr(dist,'all_gather_into_tensor',forbidden)
    assert state_digest(dt)==expected
